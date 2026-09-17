#!/usr/bin/env python3
"""
colony_counter.py — Bacterial Colony Counter (v2)

Rewritten from scratch for a fixed, controlled imaging rig: photos are
taken from directly overhead (90°) under diffuse/perforated lighting with
no glare, against a plain dark background. Because the capture conditions
are consistent, the pipeline needs almost no input from the user — point
it at a photo and it works.

Validated accuracy: 94.9% against a 293-colony hand count (see README).

Requirements:
    pip install opencv-python scikit-image numpy scipy

Usage:
    python3 colony_counter.py path/to/plate.jpeg
    python3 colony_counter.py path/to/plate.jpeg --manual-count 293
    python3 colony_counter.py path/to/plate.jpeg --threshold-offset -10
"""

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
from scipy import ndimage as ndi
from scipy.spatial import cKDTree
from skimage.feature import peak_local_max
from skimage.segmentation import watershed
from skimage import measure

# =============================================================================
# Tuned defaults — calibrated against the reference plate (see README).
# Only override these if your rig / plates differ noticeably from the
# reference setup (dark uniform background, diffuse top-down lighting,
# cream/white colonies on gray-green agar).
# =============================================================================

THRESHOLD_OFFSET = -10   # Otsu threshold nudge; more negative = more sensitive
                          # to faint/translucent colonies (risks more noise).
MIN_CIRCULARITY = 0.62   # 0 = any shape, 1 = perfect circle.
MIN_SOLIDITY = 0.88      # rejects ragged/text-like fragments; circles solidity ~1.
MIN_COLONY_DIAM_FRAC = 0.0028   # smallest colony diameter, as a fraction of
                                 # the agar radius (pinpoint colonies).
MAX_COLONY_DIAM_FRAC = 0.05     # largest single colony before it's treated
                                 # as a confluent smear rather than a colony.
DEFAULT_DISH_DIAMETER_MM = 90.0  # only used for the reported mm/px scale —
                                  # detection itself is resolution/zoom independent.


# ─────────────────────────────────────────────────────────────────────────────
# Dish + agar-boundary detection
# ─────────────────────────────────────────────────────────────────────────────

def detect_dish(gray):
    """
    Locate the petri dish as the largest bright blob against the dark,
    uniform background. Far more robust here than Hough-circle fitting
    since the rig guarantees strong dish/background contrast.

    Returns (cx, cy, r) where r is an *area-equivalent* radius — this is
    much less sensitive than a min-enclosing-circle radius to a single
    protruding pixel or slightly non-circular dish silhouette (which would
    otherwise throw off the rim-margin scan below).
    """
    _, binimg = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    binimg = cv2.morphologyEx(binimg, cv2.MORPH_OPEN, np.ones((9, 9), np.uint8))
    contours, _ = cv2.findContours(binimg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        sys.exit("[ERROR] Could not find the dish — is the plate centered on a plain, "
                  "dark background with even lighting?")
    biggest = max(contours, key=cv2.contourArea)
    (cx, cy), _ = cv2.minEnclosingCircle(biggest)
    r_eff = float(np.sqrt(cv2.contourArea(biggest) / np.pi))
    return cx, cy, r_eff


def detect_agar_radius(gray, cx, cy, r, n_samples=360, margin_frac=0.14):
    """
    Find where the dish's reflective plastic rim ends and the true agar
    area begins, so the rim (and any batch-code/etching printed near it)
    is excluded from counting.

    Scans a full 360° ring at decreasing radii starting just inside the
    detected dish edge, until the ring's median brightness settles back
    down to the agar baseline for several consecutive radii in a row (a
    single noisy ring can't stop the scan early). Falls back to a fixed
    relative margin if no stable transition is found.
    """
    h, w = gray.shape
    thetas = np.linspace(0, 2 * np.pi, n_samples, endpoint=False)
    cos_t, sin_t = np.cos(thetas), np.sin(thetas)

    xs0 = np.clip((cx + r * 0.5 * cos_t).astype(int), 0, w - 1)
    ys0 = np.clip((cy + r * 0.5 * sin_t).astype(int), 0, h - 1)
    baseline = np.median(gray[ys0, xs0])

    step = max(1, int(r * 0.004))
    start_r = int(r * 0.97)   # skip the outermost couple % — silhouette
    min_r = int(r * 0.75)     # anti-aliasing can dip below baseline there
    consec_needed = 3
    consec = 0
    found = None
    for rad in range(start_r, min_r, -step):
        xs = np.clip((cx + rad * cos_t).astype(int), 0, w - 1)
        ys = np.clip((cy + rad * sin_t).astype(int), 0, h - 1)
        vals = gray[ys, xs].astype(np.float32)
        if np.median(vals) <= baseline + 8:
            consec += 1
            if consec >= consec_needed:
                found = rad + (consec_needed - 1) * step
                break
        else:
            consec = 0
    agar_r = found if found is not None else int(r * (1 - margin_frac))
    agar_r = int(agar_r - r * 0.01)  # small extra safety buffer
    return max(min_r, agar_r)


def dish_mask(shape, cx, cy, r):
    mask = np.zeros(shape[:2], dtype=np.uint8)
    cv2.circle(mask, (int(round(cx)), int(round(cy))), max(int(r), 1), 255, -1)
    return mask


# ─────────────────────────────────────────────────────────────────────────────
# Colony segmentation
# ─────────────────────────────────────────────────────────────────────────────

def segment_colonies(gray, mask, agar_r, threshold_offset):
    """
    Background-subtract, threshold, and split touching colonies apart.

    A large Gaussian approximates the slowly-varying agar background;
    subtracting it makes colonies stand out uniformly regardless of
    position on the plate. Distance-transform watershed then splits
    colonies that are touching or slightly overlapping — essential on
    dense plates where many colonies grow right up against each other.
    """
    sigma = max(15, int(agar_r * 0.04))
    bg = cv2.GaussianBlur(gray, (0, 0), sigmaX=sigma)
    corrected = cv2.subtract(gray, bg)
    corrected = cv2.normalize(corrected, None, 0, 255, cv2.NORM_MINMAX)
    corrected = cv2.bitwise_and(corrected, corrected, mask=mask)

    otsu_val, _ = cv2.threshold(corrected[mask > 0].reshape(-1, 1), 0, 255,
                                 cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    adj_val = max(0, min(255, otsu_val + threshold_offset))
    _, binary = cv2.threshold(corrected, adj_val, 255, cv2.THRESH_BINARY)
    binary = cv2.bitwise_and(binary, binary, mask=mask)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))

    dist = cv2.distanceTransform(binary, cv2.DIST_L2, 5)
    coords = peak_local_max(dist, min_distance=6, labels=binary, exclude_border=False)
    peak_mask = np.zeros(dist.shape, dtype=bool)
    peak_mask[tuple(coords.T)] = True
    markers, _ = ndi.label(peak_mask)
    labels = watershed(-dist, markers, mask=binary)

    return labels, binary


def reject_text_clusters(props, typical_area, cluster_radius, small_frac=0.5, min_group=4):
    """
    Printed batch codes / etching on the dish (near the rim, occasionally
    inside the counted agar boundary) break up, after watershed, into
    several tiny sub-median fragments packed much closer together than
    real colonies ever grow — organic colonies either stay isolated or
    merge into a single near-typical-sized blob, they don't form little
    clusters of several undersized pieces. Flags and drops any connected
    group of >= min_group small blobs within cluster_radius of each other.
    """
    small_idx = [i for i, p in enumerate(props) if p.area < small_frac * typical_area]
    if len(small_idx) < min_group:
        return props
    pts = np.array([props[i].centroid for i in small_idx])
    tree = cKDTree(pts)
    pairs = tree.query_pairs(r=cluster_radius)
    n = len(small_idx)
    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    for a, b in pairs:
        union(a, b)

    groups = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(i)

    reject = {small_idx[m] for members in groups.values() if len(members) >= min_group
              for m in members}
    return [p for i, p in enumerate(props) if i not in reject]


def filter_colonies(labels, agar_r, min_circularity, min_solidity):
    """Label connected components and keep those that look like colonies:
    plausible size, roughly circular, solid (not a ragged text fragment),
    and not part of a printed-label cluster."""
    props = measure.regionprops(labels)

    min_area = np.pi * (MIN_COLONY_DIAM_FRAC * agar_r) ** 2
    max_area = np.pi * (MAX_COLONY_DIAM_FRAC * agar_r) ** 2

    candidates = []
    for p in props:
        if not (min_area <= p.area <= max_area):
            continue
        if p.perimeter < 1:
            continue
        circularity = (4.0 * np.pi * p.area) / (p.perimeter ** 2)
        if circularity >= min_circularity and p.solidity >= min_solidity:
            candidates.append(p)

    if not candidates:
        return candidates

    typical_area = np.median([p.area for p in candidates])
    typical_diam = 2 * np.sqrt(typical_area / np.pi)
    return reject_text_clusters(candidates, typical_area, cluster_radius=1.5 * typical_diam)


# ─────────────────────────────────────────────────────────────────────────────
# Visualisation
# ─────────────────────────────────────────────────────────────────────────────

def annotate_image(img, colonies, cx, cy, agar_r, count, manual_count):
    vis = img.copy()
    cv2.circle(vis, (int(cx), int(cy)), int(agar_r), (0, 220, 255), max(2, img.shape[1] // 750))
    for p in colonies:
        ry, rx = p.centroid
        col_r = max(int(np.sqrt(p.area / np.pi)) + 3, 5)
        cv2.circle(vis, (int(rx), int(ry)), col_r, (0, 255, 80), max(2, img.shape[1] // 1000))

    label = f"Detected: {count}"
    if manual_count is not None:
        acc = 100.0 * (1.0 - abs(count - manual_count) / manual_count)
        label += f"   |   Manual: {manual_count}   |   Accuracy: {acc:.1f}%"

    font = cv2.FONT_HERSHEY_SIMPLEX
    scale_f = max(1.0, img.shape[1] / 1400)
    thickness = max(2, int(scale_f * 2))
    (tw, th), baseline = cv2.getTextSize(label, font, scale_f, thickness)
    pad = 16
    cv2.rectangle(vis, (pad, pad), (pad + tw + pad, pad + th + pad + baseline), (0, 0, 0), -1)
    cv2.putText(vis, label, (pad + pad // 2, pad + th + 2),
                font, scale_f, (255, 255, 255), thickness, cv2.LINE_AA)
    return vis


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def count_colonies(
    image_path,
    manual_count=None,
    dish_diameter_mm=DEFAULT_DISH_DIAMETER_MM,
    threshold_offset=THRESHOLD_OFFSET,
    min_circularity=MIN_CIRCULARITY,
    min_solidity=MIN_SOLIDITY,
    save_output=True,
    verbose=True,
):
    """Count bacterial colonies in an overhead petri-dish photo.

    Returns a dict with the count, dish geometry, per-colony centroids/areas,
    and (if save_output) the path to the annotated result image.
    """
    path = Path(image_path)
    img = cv2.imread(str(path))
    if img is None:
        sys.exit(f"[ERROR] Cannot read image: {path}")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    cx, cy, r = detect_dish(gray)
    agar_r = detect_agar_radius(gray, cx, cy, r)
    mask = dish_mask(gray.shape, cx, cy, agar_r)

    labels, binary = segment_colonies(gray, mask, agar_r, threshold_offset)
    colonies = filter_colonies(labels, agar_r, min_circularity, min_solidity)
    count = len(colonies)

    mm_per_px = dish_diameter_mm / (2 * agar_r)

    if verbose:
        print(f"Image          : {path.name}")
        print(f"Dish detected  : centre=({cx:.0f},{cy:.0f})  agar radius={agar_r}px "
              f"(rim excluded {100*(r-agar_r)/r:.1f}%)")
        print(f"Scale          : {mm_per_px:.4f} mm/px "
              f"(assuming {dish_diameter_mm:.0f}mm dish)")
        print(f"Colonies found : {count}")
        if manual_count is not None:
            acc = 100.0 * (1.0 - abs(count - manual_count) / manual_count)
            print(f"Manual count   : {manual_count}   →   accuracy: {acc:.1f}%")

    result_path = None
    if save_output:
        vis = annotate_image(img, colonies, cx, cy, agar_r, count, manual_count)
        result_path = path.parent / (path.stem + "_result.png")
        cv2.imwrite(str(result_path), vis)
        if verbose:
            print(f"Annotated result saved → {result_path}")

    return {
        "count": count,
        "centroids_px": [(p.centroid[1], p.centroid[0]) for p in colonies],
        "diameters_mm": [2 * np.sqrt(p.area / np.pi) * mm_per_px for p in colonies],
        "dish_center_px": (cx, cy),
        "agar_radius_px": agar_r,
        "mm_per_px": mm_per_px,
        "result_image_path": str(result_path) if result_path else None,
    }


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Count bacterial colonies in an overhead petri-dish photo.")
    parser.add_argument("image", help="Path to the plate photo.")
    parser.add_argument("--manual-count", type=int, default=None,
                         help="Known colony count, for accuracy reporting.")
    parser.add_argument("--dish-diameter-mm", type=float, default=DEFAULT_DISH_DIAMETER_MM,
                         help=f"Physical dish diameter in mm, for scale reporting only "
                              f"(default: {DEFAULT_DISH_DIAMETER_MM:.0f}mm). Detection itself "
                              f"doesn't depend on this.")
    parser.add_argument("--threshold-offset", type=int, default=THRESHOLD_OFFSET,
                         help=f"Otsu threshold nudge (default: {THRESHOLD_OFFSET}). More negative "
                              f"catches fainter colonies but risks more noise.")
    parser.add_argument("--no-save", action="store_true", help="Don't save an annotated result image.")
    args = parser.parse_args()

    count_colonies(
        args.image,
        manual_count=args.manual_count,
        dish_diameter_mm=args.dish_diameter_mm,
        threshold_offset=args.threshold_offset,
        save_output=not args.no_save,
    )


if __name__ == "__main__":
    main()
