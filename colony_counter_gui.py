#!/usr/bin/env python3
"""
colony_counter_gui.py — Bacterial Colony Counter · GUI launcher

Idle animations
───────────────
Place three 1:1 MP4 files (H.264) in the  assets/  folder next to this file:

    Colony Algorithm/
    ├── colony_counter.py
    ├── colony_counter_gui.py
    └── assets/
        ├── anim_light.mp4
        ├── anim_default.mp4
        └── anim_dark.mp4

Requirements:
    pip install opencv-python scikit-image numpy matplotlib scipy pillow
"""
from __future__ import annotations

import sys
import threading
import subprocess
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog, messagebox, scrolledtext
from PIL import Image, ImageTk

sys.path.insert(0, str(Path(__file__).parent))
import colony_counter as cc

# ─────────────────────────────────────────────────────────────────────────────
# ASSETS  (resolved relative to this file — portable when folder is shared)
# ─────────────────────────────────────────────────────────────────────────────
_ASSETS = Path(__file__).parent / "assets"

VIDEOS: dict[str, Path] = {
    "light":   _ASSETS / "anim_light.mp4",
    "default": _ASSETS / "anim_default.mp4",
    "dark":    _ASSETS / "anim_dark.mp4",
}

ANIM_SIZE = 280   # square canvas — videos are 1:1 so no distortion

# ─────────────────────────────────────────────────────────────────────────────
# COLOUR THEMES
#   light   → #ffffff background
#   default → #555f71 background
#   dark    → #3a3a3a background
# ─────────────────────────────────────────────────────────────────────────────
THEMES: dict[str, dict[str, str]] = {
    "light": {
        "bg":          "#ffffff",   # settings panel
        "fg":          "#1e293b",
        "hint_fg":     "#64748b",
        "entry_bg":    "#f1f5f9",
        "entry_fg":    "#1e293b",
        "sep_fg":      "#dde3ee",
        "btn_bg":      "#e2e8f0",
        "btn_fg":      "#334155",
        "btn_act":     "#c8d3e3",
        "run_bg":      "#2563eb",
        "run_fg":      "#334155",
        "run_act":     "#1d4ed8",
        "title_fg":    "#1d4ed8",
        "log_bg":      "#f8fafc",
        "log_fg":      "#334155",
        "canvas_bg":   "#eef2f7",   # animation panel — matches video bg
        "tog_track":   "#dde3ee",
        "tog_sel":     "#2563eb",
        "tog_sel_fg":  "#ffffff",
        "tog_fg":      "#94a3b8",
    },
    "default": {
        "bg":          "#464f61",   # settings panel
        "fg":          "#f1f5f9",
        "hint_fg":     "#c0cad8",
        "entry_bg":    "#3d4655",
        "entry_fg":    "#334155",
        "sep_fg":      "#68778a",
        "btn_bg":      "#68778a",
        "btn_fg":      "#334155",
        "btn_act":     "#78889a",
        "run_bg":      "#60a5fa",
        "run_fg":      "#1e3a5f",
        "run_act":     "#3b82f6",
        "title_fg":    "#93c5fd",
        "log_bg":      "#3d4655",   
        "log_fg":      "#c0cad8",
        "canvas_bg":   "#555f71",   # animation panel — matches video bg
        "tog_track":   "#3a4352",
        "tog_sel":     "#60a5fa",
        "tog_sel_fg":  "#1e3a5f",
        "tog_fg":      "#8896ab",
    },
    "dark": {
        "bg":          "#2e2e2e",   # settings panel
        "fg":          "#f1f5f9",
        "hint_fg":     "#909090",
        "entry_bg":    "#2a2a2a",
        "entry_fg":    "#334155",
        "sep_fg":      "#505050",
        "btn_bg":      "#4a4a4a",
        "btn_fg":      "#334155",
        "btn_act":     "#5a5a5a",
        "run_bg":      "#5b8dee",
        "run_fg":      "#334155",
        "run_act":     "#4070d4",
        "title_fg":    "#7eb8f7",
        "log_bg":      "#222222",
        "log_fg":      "#c0c0c0",
        "canvas_bg":   "#3a3a3a",   # animation panel — matches video bg
        "tog_track":   "#252525",
        "tog_sel":     "#5b8dee",
        "tog_sel_fg":  "#334155",
        "tog_fg":      "#666666",
    },
}


# ─────────────────────────────────────────────────────────────────────────────
# LIVE LOG — redirects stdout to a ScrolledText widget
# ─────────────────────────────────────────────────────────────────────────────
class _LiveLog:
    def __init__(self, widget: scrolledtext.ScrolledText, root: tk.Tk) -> None:
        self._w, self._root = widget, root

    def write(self, text: str) -> None:
        self._root.after(0, self._append, text)

    def _append(self, text: str) -> None:
        self._w.config(state="normal")
        self._w.insert("end", text)
        self._w.see("end")
        self._w.config(state="disabled")

    def flush(self) -> None:
        pass


# ─────────────────────────────────────────────────────────────────────────────
# MODE TOGGLE — Canvas-rendered three-segment pill control
# ─────────────────────────────────────────────────────────────────────────────
class ModeToggle(tk.Canvas):
    """Segmented pill control: ☀ light / ◑ default / ☾ dark."""

    MODES = ["light", "default", "dark"]
    ICONS = {"light": "☀", "default": "◑", "dark": "☾"}
    W, H, R = 117, 32, 8

    def __init__(self, parent: tk.Widget, initial: str = "default",
                 on_change=None, **kw) -> None:
        t = THEMES[initial]
        super().__init__(parent, width=self.W, height=self.H,
                         bg=t["canvas_bg"], highlightthickness=0, bd=0,
                         cursor="hand2", **kw)
        self._mode     = initial
        self._callback = on_change
        self.bind("<Button-1>", self._click)
        self._redraw()

    @property
    def mode(self) -> str:
        return self._mode

    def sync(self, mode: str) -> None:
        """Update appearance without triggering the callback."""
        self._mode = mode
        self.config(bg=THEMES[mode]["canvas_bg"])
        self._redraw()

    def _seg(self) -> int:
        return self.W // 3

    def _click(self, e: tk.Event) -> None:
        idx = min(int(e.x // self._seg()), 2)
        new = self.MODES[idx]
        if new != self._mode:
            self._mode = new
            self._redraw()
            if self._callback:
                self._callback(new)

    def _pill(self, x1: float, y1: float, x2: float, y2: float,
              r: float, **kw) -> int:
        pts = [
            x1+r, y1,   x2-r, y1,
            x2,   y1,   x2,   y1+r,
            x2,   y2-r, x2,   y2,
            x2-r, y2,   x1+r, y2,
            x1,   y2,   x1,   y2-r,
            x1,   y1+r, x1,   y1,
        ]
        return self.create_polygon(pts, smooth=True, **kw)

    def _redraw(self) -> None:
        t   = THEMES[self._mode]
        seg = self._seg()
        idx = self.MODES.index(self._mode)
        self.delete("all")

        # track (pill background)
        self._pill(1, 1, self.W - 1, self.H - 1, self.R,
                   fill=t["tog_track"], outline="")

        # active-segment highlight (inset 2 px)
        sx = idx * seg + 2
        self._pill(sx, 2, sx + seg - 4, self.H - 2, max(1, self.R - 2),
                   fill=t["tog_sel"], outline="")

        # icons
        cy = self.H // 2
        for i, m in enumerate(self.MODES):
            cx = i * seg + seg // 2
            fg = t["tog_sel_fg"] if i == idx else t["tog_fg"]
            self.create_text(cx, cy, text=self.ICONS[m],
                             fill=fg, font=("Helvetica", 13))


# ─────────────────────────────────────────────────────────────────────────────
# MAIN APPLICATION
# ─────────────────────────────────────────────────────────────────────────────
class ColonyCounterApp:

    def __init__(self, root: tk.Tk) -> None:
        self.root    = root
        self.mode    = "default"
        self._caps:  dict[str, cv2.VideoCapture] = {}     # all three open in parallel
        self._photo: ImageTk.PhotoImage | None  = None   # GC guard
        self._anim_id: str | None               = None
        self._result: dict | None              = None
        self._themed: list[tuple]              = []
        self._option_menus: list[tuple]        = []

        root.title("Bacterial Colony Counter")
        root.resizable(False, False)

        self._init_vars()
        self._build()
        self._apply_theme()
        self._start_animation()

    # ── Variables ─────────────────────────────────────────────────────────────

    def _init_vars(self) -> None:
        self.v_image   = tk.StringVar()
        self.v_manual  = tk.StringVar()
        self.v_plate   = tk.StringVar(value="standard")
        self.v_size    = tk.StringVar(value="medium")
        self.v_grid    = tk.BooleanVar(value=False)
        self.v_save    = tk.BooleanVar(value=True)
        self.v_sat     = tk.StringVar(value="0.0")
        self.v_circ    = tk.StringVar(value="0.30")
        self.v_sigma   = tk.StringVar(value="0")
        self.v_offset  = tk.StringVar(value="0")
        self.v_status  = tk.StringVar(value="Ready.")

    # ── Layout ────────────────────────────────────────────────────────────────

    def _build(self) -> None:
        t = THEMES[self.mode]

        outer = tk.Frame(self.root, bg=t["bg"])
        outer.pack(fill="both", expand=True)
        self._reg(outer, "bg")

        # ── TOP: settings (left) + animation (right) ──────────────────────────
        top = tk.Frame(outer, bg=t["bg"])
        top.pack(fill="x", side="top")
        self._reg(top, "bg")

        self._left = tk.Frame(top, bg=t["bg"], padx=22, pady=18)
        self._left.pack(side="left", fill="y")
        self._reg(self._left, "bg")
        self._build_settings()

        vsep = tk.Frame(top, width=1, bg=t["sep_fg"])
        vsep.pack(side="left", fill="y")
        self._reg(vsep, "sep")

        self._right = tk.Frame(top, bg=t["canvas_bg"],
                               width=ANIM_SIZE + 24, pady=12, padx=12)
        self._right.pack(side="left", fill="y")
        self._right.pack_propagate(False)
        self._reg(self._right, "canvas_panel")
        self._build_right_panel()

        # ── HORIZONTAL SEPARATOR ──────────────────────────────────────────────
        hsep = tk.Frame(outer, height=1, bg=t["sep_fg"])
        hsep.pack(fill="x")
        self._reg(hsep, "sep")

        # ── BOTTOM: full-width log + run ──────────────────────────────────────
        bot = tk.Frame(outer, bg=t["bg"], padx=22, pady=14)
        bot.pack(fill="both", expand=True)
        self._reg(bot, "bg")
        self._build_log(bot)

    # ── Settings panel (left) ─────────────────────────────────────────────────

    def _build_settings(self) -> None:
        f   = self._left
        PAD = {"pady": (0, 10)}

        title = tk.Label(f, text="Bacterial Colony Counter",
                         font=("", 15, "bold"), anchor="w")
        title.pack(fill="x", pady=(2, 14))
        self._reg(title, "title")

        self._section(f, "Image file")
        row_img = tk.Frame(f)
        row_img.pack(fill="x", **PAD)
        self._reg(row_img, "bg")

        img_ent = tk.Entry(row_img, textvariable=self.v_image,
                           relief="flat", bd=4, font=("", 10))
        img_ent.pack(side="left", fill="x", expand=True, ipady=4)
        self._reg(img_ent, "entry")

        browse = tk.Button(row_img, text="Browse…", relief="flat",
                           font=("", 10), padx=10, command=self._browse)
        browse.pack(side="left", padx=(8, 0), ipady=4)
        self._reg(browse, "btn")

        # Three-column row
        g3 = tk.Frame(f)
        g3.pack(fill="x", **PAD)
        self._reg(g3, "bg")

        col0 = tk.Frame(g3)
        col0.pack(side="left", fill="x", expand=True, padx=(0, 14))
        self._reg(col0, "bg")
        self._section(col0, "Manual count")
        mc = tk.Entry(col0, textvariable=self.v_manual,
                      relief="flat", bd=4, font=("", 10), width=10)
        mc.pack(anchor="w", ipady=4, pady=(3, 2))
        self._reg(mc, "entry")
        self._hint(col0, "Leave blank if unknown")

        col1 = tk.Frame(g3)
        col1.pack(side="left", fill="x", expand=True, padx=(0, 14))
        self._reg(col1, "bg")
        self._section(col1, "Plate type")
        pt_om = self._option_menu(col1, self.v_plate,
                                  ["standard", "light_agar"])
        pt_om.pack(anchor="w", pady=(3, 2))
        self._plate_hint_lbl = self._hint(col1, self._plate_hint_text())
        self.v_plate.trace_add("write",
            lambda *_: self._plate_hint_lbl.config(
                text=self._plate_hint_text()))

        col2 = tk.Frame(g3)
        col2.pack(side="left", fill="x", expand=True)
        self._reg(col2, "bg")
        self._section(col2, "Colony size")
        sz_om = self._option_menu(col2, self.v_size,
                                  ["tiny", "medium", "large", "mixed"])
        sz_om.pack(anchor="w", pady=(3, 2))
        self._size_hint_lbl = self._hint(col2, self._size_hint_text())
        self.v_size.trace_add("write",
            lambda *_: self._size_hint_lbl.config(
                text=self._size_hint_text()))

        # Checkboxes
        cb_row = tk.Frame(f)
        cb_row.pack(fill="x", **PAD)
        self._reg(cb_row, "bg")
        for txt, var in [("  Grid lines visible", self.v_grid),
                          ("  Save result image",  self.v_save)]:
            cb = tk.Checkbutton(cb_row, text=txt, variable=var,
                                font=("", 10), relief="flat", bd=0)
            cb.pack(side="left", padx=(0, 18))
            self._reg(cb, "check")

        # Satellite exclusion
        self._section(f, "Exclude colonies smaller than")
        row_sat = tk.Frame(f)
        row_sat.pack(fill="x", **PAD)
        self._reg(row_sat, "bg")
        sat_ent = tk.Entry(row_sat, textvariable=self.v_sat,
                           relief="flat", bd=4, font=("", 10), width=6)
        sat_ent.pack(side="left", ipady=4)
        self._reg(sat_ent, "entry")
        sat_lbl = tk.Label(row_sat, text="  mm   (0.0 = off)", font=("", 9))
        sat_lbl.pack(side="left")
        self._reg(sat_lbl, "hint")

        # Advanced toggle
        self._adv_open = False
        self._adv_btn  = tk.Button(f, text="▶  Advanced settings",
                                   relief="flat", font=("", 10), anchor="w",
                                   command=self._toggle_advanced)
        self._adv_btn.pack(fill="x", pady=(6, 2))
        self._reg(self._adv_btn, "btn")

        self._adv_frame = tk.Frame(f, padx=12, pady=8)
        self._reg(self._adv_frame, "adv_panel")
        adv_inner = tk.Frame(self._adv_frame)
        adv_inner.pack(fill="x")
        self._reg(adv_inner, "adv_panel")

        for i, (lbl, var, tip) in enumerate([
            ("Min circularity",    self.v_circ,   "0.30 permissive · 0.55 strict"),
            ("BG sigma (0=auto)",  self.v_sigma,  "Larger = more bg removed"),
            ("Threshold offset",   self.v_offset, "+ fewer  · − more"),
        ]):
            col = tk.Frame(adv_inner)
            col.grid(row=0, column=i, padx=(0, 16), sticky="w")
            self._reg(col, "adv_panel")
            lw = tk.Label(col, text=lbl, font=("", 9, "bold"))
            lw.pack(anchor="w")
            self._reg(lw, "adv_label")
            ew = tk.Entry(col, textvariable=var, relief="flat", bd=3,
                          font=("", 10), width=9)
            ew.pack(anchor="w", pady=(2, 2))
            self._reg(ew, "entry")
            tw = tk.Label(col, text=tip, font=("", 8))
            tw.pack(anchor="w")
            self._reg(tw, "hint")

    # ── Right panel (animation) ───────────────────────────────────────────────

    def _build_right_panel(self) -> None:
        t = THEMES[self.mode]

        wrap = tk.Frame(self._right, bg=t["canvas_bg"])
        wrap.pack(pady=(0, 10))
        self._reg(wrap, "canvas_panel")

        self._toggle = ModeToggle(wrap, initial=self.mode,
                                  on_change=self._on_mode_change)
        self._toggle.pack()

        self._canvas = tk.Canvas(
            self._right,
            width=ANIM_SIZE, height=ANIM_SIZE,
            bg=t["canvas_bg"], highlightthickness=0,
        )
        self._canvas.pack()
        self._reg(self._canvas, "canvas_widget")

    # ── Bottom log + run ──────────────────────────────────────────────────────

    def _build_log(self, parent: tk.Frame) -> None:
        hdr = tk.Label(parent, text="Output log",
                       font=("", 10, "bold"), anchor="w")
        hdr.pack(fill="x")
        self._reg(hdr, "label")

        self._log = scrolledtext.ScrolledText(
            parent, height=9,
            state="disabled",
            font=("Courier", 10),
            relief="flat", borderwidth=0,
        )
        self._log.pack(fill="both", expand=True, pady=(4, 10))
        self._live_log = _LiveLog(self._log, self.root)

        run_row = tk.Frame(parent)
        run_row.pack(fill="x")
        self._reg(run_row, "bg")

        status = tk.Label(run_row, textvariable=self.v_status,
                          font=("", 9), anchor="w")
        status.pack(side="left")
        self._reg(status, "hint")

        self._run_btn = tk.Button(run_row, text="Run Analysis",
                                  font=("", 11, "bold"),
                                  relief="flat", padx=18, pady=8,
                                  command=self._run)
        self._run_btn.pack(side="right")
        self._reg(self._run_btn, "run")

    # ── Theming ───────────────────────────────────────────────────────────────

    def _reg(self, widget: tk.Widget, role: str) -> None:
        self._themed.append((widget, role))

    def _apply_theme(self) -> None:
        t = THEMES[self.mode]

        ROLES: dict[str, dict] = {
            "bg":            {"bg": t["bg"]},
            "canvas_panel":  {"bg": t["canvas_bg"]},
            "canvas_widget": {"bg": t["canvas_bg"]},
            "sep":           {"bg": t["sep_fg"]},
            "title":         {"bg": t["bg"],       "fg": t["title_fg"]},
            "section":       {"bg": t["bg"],       "fg": t["fg"]},
            "label":         {"bg": t["bg"],       "fg": t["fg"]},
            "hint":          {"bg": t["bg"],       "fg": t["hint_fg"]},
            "entry":         {"bg": t["entry_bg"], "fg": t["entry_fg"],
                              "insertbackground": t["entry_fg"]},
            "adv_panel":     {"bg": t["entry_bg"]},
            "adv_label":     {"bg": t["entry_bg"], "fg": t["fg"]},
            "btn":           {"bg": t["btn_bg"],   "fg": t["btn_fg"],
                              "activebackground": t["btn_act"],
                              "activeforeground": t["btn_fg"]},
            "check":         {"bg": t["bg"],       "fg": t["fg"],
                              "selectcolor":      t["entry_bg"],
                              "activebackground": t["bg"],
                              "activeforeground": t["fg"]},
            "run":           {"bg": t["run_bg"],   "fg": t["run_fg"],
                              "activebackground": t["run_act"],
                              "activeforeground": t["run_fg"]},
        }

        for widget, role in self._themed:
            cfg = ROLES.get(role)
            if cfg:
                try:
                    widget.config(**cfg)
                except tk.TclError:
                    pass

        self._log.config(bg=t["log_bg"], fg=t["log_fg"],
                         insertbackground=t["log_fg"])

        for om, _var, _choices in self._option_menus:
            try:
                om.config(bg=t["btn_bg"], fg=t["btn_fg"],
                          activebackground=t["entry_bg"],
                          activeforeground=t["fg"],
                          highlightthickness=0)
                om["menu"].config(bg=t["entry_bg"], fg=t["fg"],
                                  activebackground=t["btn_bg"],
                                  activeforeground=t["btn_fg"])
            except tk.TclError:
                pass

        self._toggle.config(bg=t["canvas_bg"])
        self._toggle.sync(self.mode)
        self.root.configure(bg=t["bg"])

    def _on_mode_change(self, new_mode: str) -> None:
        self.mode = new_mode
        self._apply_theme()
        # Frame loop keeps running uninterrupted — all caps advance in sync,
        # so _next_frame will simply start rendering the new mode's frames.
        # Only start the loop if it isn't already running (placeholder state).
        if self._anim_id is None:
            self._start_animation()

    # ── Widget factories ──────────────────────────────────────────────────────

    def _section(self, parent: tk.Widget, text: str) -> tk.Label:
        lbl = tk.Label(parent, text=text, font=("", 10, "bold"), anchor="w")
        lbl.pack(fill="x")
        self._reg(lbl, "section")
        return lbl

    def _hint(self, parent: tk.Widget, text: str) -> tk.Label:
        lbl = tk.Label(parent, text=text, font=("", 9), anchor="w")
        lbl.pack(anchor="w")
        self._reg(lbl, "hint")
        return lbl

    def _option_menu(self, parent: tk.Widget, var: tk.StringVar,
                     choices: list[str]) -> tk.OptionMenu:
        om = tk.OptionMenu(parent, var, *choices)
        om.config(relief="flat", font=("", 10), padx=6, pady=4,
                  indicatoron=True, bd=0)
        self._option_menus.append((om, var, choices))
        return om

    _PLATE_HINTS = {
        "standard":   "Dark agar — LB / LB-Amp plates",
        "light_agar": "Cream agar — use satellite exclusion",
    }
    _SIZE_HINTS = {
        "tiny":   "< 0.6 mm diameter",
        "medium": "0.1 – 2.0 mm diameter",
        "large":  "0.8 – 7.0 mm diameter",
        "mixed":  "any detectable size",
    }

    def _plate_hint_text(self) -> str:
        return self._PLATE_HINTS.get(self.v_plate.get(), "")

    def _size_hint_text(self) -> str:
        return self._SIZE_HINTS.get(self.v_size.get(), "")

    def _toggle_advanced(self) -> None:
        if self._adv_open:
            self._adv_frame.pack_forget()
            self._adv_btn.config(text="▶  Advanced settings")
            self._adv_open = False
        else:
            self._adv_frame.pack(fill="x", pady=(0, 6))
            self._adv_btn.config(text="▼  Advanced settings")
            self._adv_open = True

    def _browse(self) -> None:
        path = filedialog.askopenfilename(
            title="Select plate image",
            filetypes=[("Images", "*.jpeg *.jpg *.png *.tiff *.tif *.bmp"),
                       ("All files", "*.*")],
        )
        if path:
            self.v_image.set(path)

    # ── Animation ─────────────────────────────────────────────────────────────

    def _start_animation(self) -> None:
        """Open all three video captures and start a single shared frame loop.

        All caps advance together on every tick so they stay frame-perfect in
        sync.  Switching themes only changes which cap's frame is rendered —
        the loop itself never restarts, giving a seamless transition.
        """
        if self._anim_id is not None:
            return   # loop already running

        # Release any stale caps from a previous call
        for cap in self._caps.values():
            cap.release()
        self._caps = {}

        for mode, path in VIDEOS.items():
            if path.exists():
                cap = cv2.VideoCapture(str(path))
                if cap.isOpened():
                    self._caps[mode] = cap

        if self._caps:
            fps          = next(iter(self._caps.values())).get(cv2.CAP_PROP_FPS) or 25
            self._frm_ms = max(16, int(1000 / fps))
            self._next_frame()
        else:
            self._draw_placeholder()

    def _next_frame(self) -> None:
        if not self._caps:
            return

        # Advance every cap by one frame
        frames: dict[str, np.ndarray] = {}
        looped = False
        for mode, cap in self._caps.items():
            ret, frame = cap.read()
            if not ret:
                looped = True
                break
            frames[mode] = frame

        if looped:
            # Any cap hit end-of-file — reset ALL to keep them in sync
            for cap in self._caps.values():
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            frames.clear()
            for mode, cap in self._caps.items():
                ret, frame = cap.read()
                if ret:
                    frames[mode] = frame

        # Render only the active theme's frame
        frame = frames.get(self.mode)
        if frame is not None:
            rgb         = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            sq          = cv2.resize(rgb, (ANIM_SIZE, ANIM_SIZE),
                                     interpolation=cv2.INTER_LINEAR)
            self._photo = ImageTk.PhotoImage(Image.fromarray(sq))
            self._canvas.create_image(0, 0, anchor="nw", image=self._photo)

        self._anim_id = self.root.after(self._frm_ms, self._next_frame)

    def _draw_placeholder(self) -> None:
        t  = THEMES[self.mode]
        fg = t["hint_fg"]

        def h2rgb(h: str) -> tuple:
            h = h.lstrip("#")
            return tuple(int(h[i:i+2], 16) for i in (0, 2, 4))

        top = h2rgb(t["canvas_bg"])
        bot = tuple(max(0, c - 28) for c in top)
        arr = np.zeros((ANIM_SIZE, ANIM_SIZE, 3), dtype=np.uint8)
        for y in range(ANIM_SIZE):
            a      = y / ANIM_SIZE
            arr[y] = tuple(int(top[c] + a * (bot[c] - top[c])) for c in range(3))

        self._photo = ImageTk.PhotoImage(Image.fromarray(arr))
        self._canvas.create_image(0, 0, anchor="nw", image=self._photo)
        self._canvas.create_text(
            ANIM_SIZE // 2, ANIM_SIZE // 2,
            text="Place your MP4s in\nassets/anim_light.mp4\n"
                 "assets/anim_default.mp4\nassets/anim_dark.mp4",
            fill=fg, font=("", 10), justify="center",
        )

    # ── Run analysis ──────────────────────────────────────────────────────────

    def _run(self) -> None:
        path = self.v_image.get().strip()
        if not path:
            messagebox.showwarning("No image selected",
                                   "Please browse to a plate image first.")
            return

        mc_str       = self.v_manual.get().strip()
        manual_count = int(mc_str) if mc_str.lstrip("-").isdigit() else None

        try:
            exclude_mm       = float(self.v_sat.get())
            min_circularity  = float(self.v_circ.get())
            bg_sigma         = int(self.v_sigma.get())
            threshold_offset = int(self.v_offset.get())
        except ValueError as exc:
            messagebox.showerror("Invalid value",
                                 f"Check Advanced settings:\n{exc}")
            return

        self._log.config(state="normal")
        self._log.delete("1.0", "end")
        self._log.config(state="disabled")

        self._run_btn.config(state="disabled")
        self.v_status.set("Running…")
        self._result = None

        kwargs = dict(
            image_path                  = path,
            manual_count                = manual_count,
            plate_type                  = self.v_plate.get(),
            has_grid                    = self.v_grid.get(),
            colony_size                 = self.v_size.get(),
            exclude_satellites_below_mm = exclude_mm,
            save_output                 = self.v_save.get(),
            min_circularity             = min_circularity,
            bg_sigma                    = bg_sigma,
            threshold_offset            = threshold_offset,
        )
        threading.Thread(target=self._worker, kwargs=kwargs,
                         daemon=True).start()
        self._poll()

    def _worker(self, **kwargs) -> None:
        old = sys.stdout
        sys.stdout = self._live_log
        try:
            count        = cc.run(**kwargs)
            self._result = {"count": count,
                            "image_path": kwargs["image_path"],
                            "save": kwargs["save_output"],
                            "error": None}
        except Exception as exc:
            self._result = {"error": str(exc)}
        finally:
            sys.stdout = old

    def _poll(self) -> None:
        if self._result is None:
            self.root.after(100, self._poll)
        else:
            self._finish()

    def _finish(self) -> None:
        self._run_btn.config(state="normal")
        r = self._result

        if r.get("error"):
            self.v_status.set(f"Error — {r['error']}")
            messagebox.showerror("Analysis failed", r["error"])
            return

        self.v_status.set(f"Done — {r['count']} colonies detected.")

        if r.get("save"):
            img_p    = Path(r["image_path"])
            result_p = img_p.parent / (img_p.stem + "_result.png")
            if result_p.exists():
                subprocess.run(["open", str(result_p)], check=False)


# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    root = tk.Tk()
    ColonyCounterApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
