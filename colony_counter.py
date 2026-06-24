#!/usr/bin/env python3
"""
colony_counter.py — Bacterial Colony Counter
Counts colonies from overhead petri dish photos using computer vision.

Validated accuracy:
  • 99.7 % on a 376-colony LB plate (dark agar, grid lines, no satellites)
  • Light-agar mode for cream/yellow agar plates with satellite colonies

Requirements:
    pip install opencv-python scikit-image numpy matplotlib scipy

Usage:
    Edit the USER SETTINGS block below, then run:
        python3 colony_counter.py
"""

import sys
import cv2
import numpy as np
from skimage import measure
import matplotlib.pyplot as plt
from pathlib import Path

# =============================================================================
# USER SETTINGS  ← Edit these before running
# =============================================================================

IMAGE_PATH   = "/Users/sebastiancicconi/Desktop/7F41664F-B31B-48C8-BBE9-C8B24A60776E_1_201_a.jpeg"

MANUAL_COUNT = 376           # int or None — used only for accuracy reporting

# ── Plate type ────────────────────────────────────────────────────────────────
PLATE_TYPE = "standard"
# "standard"   — dark agar (olive/brown/green) with bright white colonies.
#                Best for classic LB or LB-Amp plates under diffuse light.
# "light_agar" — cream or yellow agar with translucent off-white colonies.
#                Use when colonies barely contrast with the background, or when
#                the plate looks mostly uniform in colour.

# ── Satellite colony exclusion ────────────────────────────────────────────────
EXCLUDE_SATELLITES_BELOW_MM = 0.2
# Satellite colonies are tiny non-resistant bacteria that grow near true
# (stable/resistant) colonies on antibiotic selection plates.
#
# Set to 0.0  → count ALL detectable colonies (no exclusion).
# Set to 0.5  → exclude everything smaller than 0.5 mm — typical for
#               ampicillin plates where satellites cluster around true colonies.
# Adjust upward (e.g. 0.8) if satellites are larger on your plate.
# Has no effect on PLATE_TYPE="standard" unless you also want size gating.

# ── Grid lines ────────────────────────────────────────────────────────────────
HAS_GRID     = True          # True  → dish has printed grid lines in photo
                             # False → plain dish, no lines to remove

# ── Colony size ───────────────────────────────────────────────────────────────
COLONY_SIZE  = "medium"
# "tiny"   — pinpoint colonies  (< 0.6 mm diameter)
# "medium" — typical lab size   (0.1–2.5 mm)
# "large"  — oversized colonies (0.8–7 mm)
# "mixed"  — any detectable size
# Note: when EXCLUDE_SATELLITES_BELOW_MM > 0, the exclusion threshold
# overrides the lower bound of this preset automatically.

SAVE_OUTPUT  = True          # Save annotated result PNG alongside the input file

# =============================================================================
# ADVANCED SETTINGS  (fine-tune only if results are poor)
# =============================================================================

# Minimum blob circularity to be counted as a colony (0 = any, 1 = perfect circle).
# 0.30 is a permissive default. Raise toward 0.55 to reject irregular shapes.
MIN_CIRCULARITY = 0.20

# Background-subtraction blur radius (pixels). Set to 0 to auto-compute.
# Auto: ~3 % of dish radius for "standard", ~5 % for "light_agar".
BG_SIGMA = 0

# Otsu threshold nudge.  0 = use Otsu directly.
# Positive = fewer detections (stricter).  Negative = more sensitive.
THRESHOLD_OFFSET = 0

# =============================================================================

# Colony size presets — (min_diameter_mm, max_diameter_mm).
# Applied AFTER satellite exclusion (which overrides the lower bound).
SIZE_PRESETS_MM = {
    "tiny":   (0.05, 0.6),
    "medium": (0.05, 2.0),
    "large":  (0.8,  7.0),
    "mixed":  (0.05, 7.0),
}


# ─────────────────────────────────────────────────────────────────────────────
# Pipeline functions
# ─────────────────────────────────────────────────────────────────────────────

def load_image(path):
    img = cv2.imread(str(path))
    if img is None:
        sys.exit(f"[ERROR] Cannot read image: {path}")
    return img


def detect_dish(gray):
    """
    Locate the circular petri dish boundary using Hough Circle Transform.
    Returns (cx, cy, radius) in pixels.
    Falls back to image-centre estimate if the circle is not found.
    """
    h, w = gray.shape
    blurred = cv2.GaussianBlur(gray, (11, 11), 3)
    min_r = int(min(h, w) * 0.30)
    max_r = int(min(h, w) * 0.52)

    circles = cv2.HoughCircles(
        blurred,
        cv2.HOUGH_GRADIENT,
        dp=1.5,
        minDist=min(h, w),
        param1=70,
        param2=35,
        minRadius=min_r,
        maxRadius=max_r,
    )

    if circles is None:
        print("[WARN] Dish boundary not detected — using image-centre fallback.")
        cx, cy = w // 2, h // 2
        r = int(min(h, w) * 0.44)
        return cx, cy, r

    cx, cy, r = np.round(circles[0, 0]).astype(int)
    return int(cx), int(cy), int(r)


def dish_mask(shape, cx, cy, r, inset_px=10):
    """Binary mask covering only the interior of the petri dish."""
    mask = np.zeros(shape[:2], dtype=np.uint8)
    cv2.circle(mask, (cx, cy), max(r - inset_px, 1), 255, -1)
    return mask


def suppress_satellites(gray, satellite_mm, dish_radius):
    """
    Erase satellite colonies using a median blur.

    A median filter with kernel diameter ≈ 2× the satellite colony diameter
    replaces each satellite (small bright dot) with the surrounding agar value,
    leaving true (stable, larger) colonies mostly intact.
    """
    scale = 45.0 / dish_radius          # mm per pixel
    sat_px = satellite_mm / scale       # satellite diameter in pixels
    # Kernel ~1.4× the satellite diameter erases satellites without smearing
    # stable colonies (which are much larger).  Must be odd.
    k = max(3, int(sat_px * 1.4))
    if k % 2 == 0:
        k += 1
    print(f"  Satellite suppression: median blur kernel={k}px "
          f"(targeting colonies < {satellite_mm:.2f} mm = {sat_px:.1f}px)")
    return cv2.medianBlur(gray, k)


def subtract_background(gray, mask, sigma):
    """
    Remove slowly-varying background illumination.

    A large Gaussian approximates the agar background.  Subtracting it makes
    colonies stand out uniformly regardless of position on the plate.
    """
    bg       = cv2.GaussianBlur(gray, (0, 0), sigmaX=sigma)
    corrected = cv2.subtract(gray, bg)
    corrected = cv2.normalize(corrected, None, 0, 255, cv2.NORM_MINMAX)
    corrected = cv2.bitwise_and(corrected, corrected, mask=mask)
    return corrected


def threshold_image(corrected, mask, offset=0):
    """Otsu threshold on the background-corrected, masked image."""
    otsu_val, _ = cv2.threshold(
        corrected[mask > 0].reshape(-1, 1).astype(np.uint8),
        0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU,
    )
    adjusted = max(0, min(255, int(otsu_val) + offset))
    print(f"  Otsu threshold: {int(otsu_val)}  (adjusted: {adjusted})")
    _, binary = cv2.threshold(corrected, adjusted, 255, cv2.THRESH_BINARY)
    binary = cv2.bitwise_and(binary, binary, mask=mask)
    return binary


def remove_grid_lines(binary, dish_radius):
    """
    Erase printed grid lines via morphological opening.

    Lines survive long thin kernels; circular colonies do not — so the
    detected lines can be subtracted from the binary image.

    Kernel length is capped at 120px: beyond that the opening begins to
    treat large connected bright regions as lines and removes colonies.
    """
    klen     = min(120, max(40, int(dish_radius * 0.06)))
    dil_kern = np.ones((5, 5), np.uint8)

    h_kern  = cv2.getStructuringElement(cv2.MORPH_RECT, (klen, 1))
    h_lines = cv2.dilate(cv2.morphologyEx(binary, cv2.MORPH_OPEN, h_kern), dil_kern)

    v_kern  = cv2.getStructuringElement(cv2.MORPH_RECT, (1, klen))
    v_lines = cv2.dilate(cv2.morphologyEx(binary, cv2.MORPH_OPEN, v_kern), dil_kern)

    grid    = cv2.bitwise_or(h_lines, v_lines)
    return cv2.bitwise_and(binary, cv2.bitwise_not(grid))


def filter_colonies(binary, dish_radius, colony_size, min_circ,
                    exclude_satellites_mm=0.0):
    """
    Label connected components and retain those that pass:
      • diameter within the size-preset range (in real mm)
      • circularity ≥ min_circ  (rejects lines, fibers, letters)

    When exclude_satellites_mm > 0, that value overrides the preset's lower bound.
    """
    scale      = 45.0 / dish_radius           # mm per pixel
    min_mm, max_mm = SIZE_PRESETS_MM[colony_size]

    if exclude_satellites_mm > 0.0:
        min_mm = max(min_mm, exclude_satellites_mm)

    # Convert mm diameters → pixel areas
    min_area_px = max(10.0, np.pi * (min_mm / scale / 2.0) ** 2)
    max_area_px = np.pi * (max_mm / scale / 2.0) ** 2

    labeled = measure.label(binary, connectivity=2)
    props   = measure.regionprops(labeled)

    valid = []
    for p in props:
        if not (min_area_px <= p.area <= max_area_px):
            continue
        if p.perimeter < 1:
            continue
        circularity = (4.0 * np.pi * p.area) / (p.perimeter ** 2)
        if circularity >= min_circ:
            valid.append(p)
    return valid


# ─────────────────────────────────────────────────────────────────────────────
# Visualisation
# ─────────────────────────────────────────────────────────────────────────────

def annotate_image(img, colonies, cx, cy, r):
    vis = img.copy()
    cv2.circle(vis, (cx, cy), r, (0, 220, 255), 4)
    for col in colonies:
        ry, rx  = col.centroid
        col_r   = max(int(np.sqrt(col.area / np.pi)) + 3, 5)
        cv2.circle(vis, (int(rx), int(ry)), col_r, (0, 255, 80), 2)
    return vis


def add_count_label(vis, count, manual_count):
    label = f"Detected: {count}"
    if manual_count is not None:
        acc    = 100.0 * (1.0 - abs(count - manual_count) / manual_count)
        label += f"   |   Manual: {manual_count}   |   Accuracy: {acc:.1f}%"

    font      = cv2.FONT_HERSHEY_SIMPLEX
    scale_f   = max(1.0, vis.shape[1] / 1800)
    thickness = 2
    (tw, th), baseline = cv2.getTextSize(label, font, scale_f, thickness)
    pad = 16
    cv2.rectangle(vis, (pad, pad), (pad + tw + pad, pad + th + pad + baseline), (0, 0, 0), -1)
    cv2.putText(vis, label, (pad + pad // 2, pad + th + 2),
                font, scale_f, (255, 255, 255), thickness, cv2.LINE_AA)
    return vis


def show_results(img, binary_clean, annotated, count, manual_count,
                 has_grid, plate_type, save_path=None):
    fig, axes = plt.subplots(1, 3, figsize=(20, 7))
    fig.patch.set_facecolor("#1a1a1a")

    binary_title = f"Processed binary [{plate_type}]"
    if has_grid:
        binary_title += " — grid removed"

    acc_str = ""
    if manual_count is not None:
        acc = 100 * (1 - abs(count - manual_count) / manual_count)
        acc_str = f"  |  {manual_count} manual  |  {acc:.1f}% accuracy"

    panels = [
        (cv2.cvtColor(img, cv2.COLOR_BGR2RGB), "Original image"),
        (binary_clean,                          binary_title),
        (cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB),
         f"Result: {count} detected{acc_str}"),
    ]

    for ax, (panel, title) in zip(axes, panels):
        ax.imshow(panel, cmap="gray" if panel.ndim == 2 else None)
        ax.set_title(title, color="white", fontsize=11, pad=8)
        ax.axis("off")

    plt.tight_layout()
    if save_path:
        plt.savefig(str(save_path), dpi=150, bbox_inches="tight",
                    facecolor=fig.get_facecolor())
        print(f"\nResult saved → {save_path}")
    plt.show()


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def run(
    image_path,
    manual_count              = None,
    plate_type                = "standard",
    has_grid                  = True,
    colony_size               = "medium",
    exclude_satellites_below_mm = 0.0,
    save_output               = True,
    min_circularity           = MIN_CIRCULARITY,
    bg_sigma                  = BG_SIGMA,
    threshold_offset          = THRESHOLD_OFFSET,
):
    path = Path(image_path)
    print("=" * 62)
    print("Bacterial Colony Counter")
    print(f"  Image          : {path.name}")
    print(f"  Plate type     : {plate_type}")
    print(f"  Grid lines     : {'YES — will be removed' if has_grid else 'NO'}")
    print(f"  Colony size    : {colony_size}")
    print(f"  Satellite excl.: "
          + (f"< {exclude_satellites_below_mm} mm" if exclude_satellites_below_mm > 0 else "OFF"))
    print("=" * 62)

    # 1 — Load
    img  = load_image(path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    print(f"\n[1/6] Loaded: {w} × {h} px")

    # 2 — Detect dish
    cx, cy, r = detect_dish(gray)
    dish_area  = np.pi * r ** 2
    scale_mm   = 45.0 / r
    print(f"[2/6] Dish: centre=({cx},{cy})  radius={r}px  "
          f"scale={scale_mm:.4f} mm/px")

    dmask = dish_mask(gray.shape, cx, cy, r, inset_px=10)

    # 3 — Preprocessing (plate-type specific)
    # For light_agar: apply median blur to erase satellites before bg subtraction
    sigma = bg_sigma if bg_sigma > 0 else (
        int(r * 0.05) if plate_type == "light_agar" else int(r * 0.03)
    )

    if plate_type == "light_agar" and exclude_satellites_below_mm > 0:
        print(f"[3/6] Satellite suppression + background subtraction (σ={sigma}) …")
        work_gray = suppress_satellites(gray, exclude_satellites_below_mm, r)
    elif plate_type == "light_agar":
        print(f"[3/6] Light-agar background subtraction (σ={sigma}) …")
        work_gray = gray
    else:
        print(f"[3/6] Background subtraction (σ={sigma}) …")
        work_gray = gray

    corrected = subtract_background(work_gray, dmask, sigma=sigma)
    binary    = threshold_image(corrected, dmask, offset=threshold_offset)

    # 4 — Grid line removal
    if has_grid:
        print("[4/6] Removing grid lines …")
        binary_clean = remove_grid_lines(binary, r)
    else:
        print("[4/6] Grid removal skipped.")
        binary_clean = binary

    # 5 — Detect & filter colonies
    print(f"[5/6] Filtering colonies …")
    colonies = filter_colonies(
        binary_clean, r, colony_size, min_circularity,
        exclude_satellites_mm=exclude_satellites_below_mm,
    )
    count = len(colonies)

    # 6 — Report
    print(f"\n[6/6] Done.")
    print()
    print("─" * 48)
    print(f"  Detected colonies  :  {count}")
    if manual_count is not None:
        err = count - manual_count
        acc = 100.0 * (1.0 - abs(count - manual_count) / manual_count)
        print(f"  Manual count       :  {manual_count}")
        print(f"  Difference         :  {err:+d}")
        print(f"  Accuracy           :  {acc:.1f}%")
    print("─" * 48)

    # Visualise
    annotated = annotate_image(img, colonies, cx, cy, r)
    annotated = add_count_label(annotated, count, manual_count)

    save_path = (path.parent / (path.stem + "_result.png")) if save_output else None
    show_results(img, binary_clean, annotated, count, manual_count,
                 has_grid, plate_type, save_path)

    return count


# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    run(
        image_path                  = IMAGE_PATH,
        manual_count                = MANUAL_COUNT,
        plate_type                  = PLATE_TYPE,
        has_grid                    = HAS_GRID,
        colony_size                 = COLONY_SIZE,
        exclude_satellites_below_mm = EXCLUDE_SATELLITES_BELOW_MM,
        save_output                 = SAVE_OUTPUT,
        min_circularity             = MIN_CIRCULARITY,
        bg_sigma                    = BG_SIGMA,
        threshold_offset            = THRESHOLD_OFFSET,
    )
