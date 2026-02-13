#!/usr/bin/env python3
"""Line detection visualizer for v4 dataset images.

Runs Canny + Hough line detection (without diagonal line elimination)
to test whether board edges can be found from arbitrary camera angles.

Uses the same algorithm as knightvision.corner_detection.detect_corners
but stripped down to just the line detection + clustering stage.

Usage:
    # Process a single image interactively
    python detect_lines.py images/ring1_az000_el30.png

    # Process all images in images/ directory
    python detect_lines.py

    # Adjust thresholds
    python detect_lines.py --canny-low 80 --canny-high 300 --hough-thresh 120

    # Save output instead of displaying
    python detect_lines.py --save
"""

import argparse
import glob
import os
import time

import cv2
import numpy as np
from sklearn.cluster import AgglomerativeClustering, DBSCAN
from sklearn.metrics.pairwise import pairwise_distances


# ── Resize ──────────────────────────────────────────────────────────────────

RESIZE_WIDTH = 1200


def resize_image(img, width=RESIZE_WIDTH):
    h, w = img.shape[:2]
    if w == width:
        return img, 1.0
    scale = width / w
    dims = (width, int(h * scale))
    return cv2.resize(img, dims), scale


# ── Edge detection ──────────────────────────────────────────────────────────

def detect_edges(gray, low_threshold=90, high_threshold=400, aperture=3):
    if gray.dtype != np.uint8:
        gray = (gray / gray.max() * 255).astype(np.uint8)
    return cv2.Canny(gray, low_threshold, high_threshold, aperture)


# ── Hough line detection ───────────────────────────────────────────────────

def fix_negative_rho(lines):
    lines = lines.copy()
    neg = lines[:, 0] < 0
    lines[neg, 0] *= -1
    lines[neg, 1] -= np.pi
    return lines


def detect_lines(edges, threshold=150):
    """Detect lines using HoughLines. No diagonal elimination."""
    raw = cv2.HoughLines(edges, 1, np.pi / 360, threshold)
    if raw is None:
        return np.empty((0, 2))
    lines = raw.squeeze(axis=-2)
    if lines.ndim == 1:
        lines = lines[np.newaxis, :]
    return fix_negative_rho(lines)


# ── Clustering ──────────────────────────────────────────────────────────────

def absolute_angle_difference(x, y):
    diff = np.mod(np.abs(x - y), 2 * np.pi)
    return np.min(np.stack([diff, np.pi - diff], axis=-1), axis=-1)


def cluster_lines(lines):
    """Split lines into two clusters by angle using AgglomerativeClustering."""
    if len(lines) < 2:
        return lines, np.empty((0, 2))

    thetas = lines[:, 1].reshape(-1, 1)
    dist = pairwise_distances(thetas, thetas, metric=absolute_angle_difference)
    agg = AgglomerativeClustering(
        n_clusters=2, metric="precomputed", linkage="average"
    )
    labels = agg.fit_predict(dist)

    # Label the cluster closer to vertical (theta~0) as "vertical"
    angle_with_y = absolute_angle_difference(thetas, 0.0)
    if angle_with_y[labels == 0].mean() > angle_with_y[labels == 1].mean():
        h_label, v_label = 0, 1
    else:
        h_label, v_label = 1, 0

    # Sort each cluster by rho
    h_lines = lines[labels == h_label]
    v_lines = lines[labels == v_label]
    h_lines = h_lines[np.argsort(h_lines[:, 0])]
    v_lines = v_lines[np.argsort(v_lines[:, 0])]
    return h_lines, v_lines


# ── Similar-line elimination ───────────────────────────────────────────────

def get_intersection_point(rho1, theta1, rho2, theta2):
    cos1, cos2 = np.cos(theta1), np.cos(theta2)
    sin1, sin2 = np.sin(theta1), np.sin(theta2)
    x = (sin1 * rho2 - sin2 * rho1) / (cos2 * sin1 - cos1 * sin2)
    y = (cos1 * rho2 - cos2 * rho1) / (sin2 * cos1 - sin1 * cos2)
    return x, y


def eliminate_similar_lines(lines, perpendicular_lines):
    """Remove near-duplicate lines using DBSCAN on intersection projections."""
    if len(lines) <= 1:
        return lines
    if len(perpendicular_lines) == 0:
        return lines

    perp_rho, perp_theta = perpendicular_lines.mean(axis=0)
    rho, theta = lines[:, 0], lines[:, 1]
    x, y = get_intersection_point(rho, theta, perp_rho, perp_theta)
    pts = np.stack([x, y], axis=-1)

    clustering = DBSCAN(eps=12, min_samples=1).fit(pts)

    filtered = []
    for c in range(clustering.labels_.max() + 1):
        cluster_lines = lines[clustering.labels_ == c]
        rhos = cluster_lines[:, 0]
        median_idx = np.argsort(rhos)[len(rhos) // 2]
        filtered.append(cluster_lines[median_idx])
    return np.stack(filtered)


# ── Intersection points ───────────────────────────────────────────────────

def get_all_intersections(h_lines, v_lines, img_w, img_h):
    """Compute all intersection points between two line clusters."""
    if len(h_lines) == 0 or len(v_lines) == 0:
        return np.empty((0, 2))

    rho1, theta1 = h_lines[:, 0], h_lines[:, 1]
    rho2, theta2 = v_lines[:, 0], v_lines[:, 1]
    rho1, rho2 = np.meshgrid(rho1, rho2, indexing="ij")
    theta1, theta2 = np.meshgrid(theta1, theta2, indexing="ij")
    x, y = get_intersection_point(rho1, theta1, rho2, theta2)
    pts = np.stack([x, y], axis=-1).reshape(-1, 2)

    # Filter to within image bounds (with 10% margin)
    margin = 0.1
    mask = (
        (pts[:, 0] >= -img_w * margin)
        & (pts[:, 0] <= img_w * (1 + margin))
        & (pts[:, 1] >= -img_h * margin)
        & (pts[:, 1] <= img_h * (1 + margin))
    )
    return pts[mask]


# ── Drawing helpers ────────────────────────────────────────────────────────

def draw_lines(img, lines, color=(0, 0, 255), thickness=2):
    """Draw Hough lines on an image (modifies in place)."""
    length = np.sqrt(img.shape[0] ** 2 + img.shape[1] ** 2)
    for rho, theta in lines:
        a, b = np.cos(theta), np.sin(theta)
        x0, y0 = a * rho, b * rho
        x1, y1 = int(x0 + length * (-b)), int(y0 + length * a)
        x2, y2 = int(x0 - length * (-b)), int(y0 - length * a)
        cv2.line(img, (x1, y1), (x2, y2), color, thickness)


def draw_intersections(img, pts, color=(0, 255, 255), radius=5, thickness=-1):
    for x, y in pts:
        cv2.circle(img, (int(x), int(y)), radius, color, thickness)
        cv2.circle(img, (int(x), int(y)), radius, (0, 0, 0), 1)


# ── Full pipeline ──────────────────────────────────────────────────────────

def run_pipeline(img_path, canny_low=90, canny_high=400, hough_thresh=150):
    """Run the full line detection pipeline on one image.

    Returns a dict with all intermediate results.
    """
    img_bgr = cv2.imread(img_path)
    if img_bgr is None:
        raise FileNotFoundError(f"Cannot read image: {img_path}")

    t0 = time.time()

    # Resize
    resized, scale = resize_image(img_bgr)
    h, w = resized.shape[:2]

    # Grayscale + Canny
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    edges = detect_edges(gray, canny_low, canny_high)

    # Hough lines (NO diagonal elimination)
    raw_lines = detect_lines(edges, hough_thresh)

    if len(raw_lines) == 0:
        elapsed = time.time() - t0
        return {
            "img": resized,
            "edges": edges,
            "raw_lines": raw_lines,
            "h_lines": np.empty((0, 2)),
            "v_lines": np.empty((0, 2)),
            "h_dedup": np.empty((0, 2)),
            "v_dedup": np.empty((0, 2)),
            "intersections": np.empty((0, 2)),
            "scale": scale,
            "elapsed": elapsed,
        }

    # Cluster into two groups
    h_lines, v_lines = cluster_lines(raw_lines)

    # Eliminate similar lines
    h_dedup = eliminate_similar_lines(h_lines, v_lines) if len(h_lines) > 0 else h_lines
    v_dedup = eliminate_similar_lines(v_lines, h_lines) if len(v_lines) > 0 else v_lines

    # Intersections
    intersections = get_all_intersections(h_dedup, v_dedup, w, h)

    elapsed = time.time() - t0

    return {
        "img": resized,
        "edges": edges,
        "raw_lines": raw_lines,
        "h_lines": h_lines,
        "v_lines": v_lines,
        "h_dedup": h_dedup,
        "v_dedup": v_dedup,
        "intersections": intersections,
        "scale": scale,
        "elapsed": elapsed,
    }


# ── Visualization ─────────────────────────────────────────────────────────

def visualize(result, title="", save_path=None):
    """Show a 2x2 visualization of the pipeline stages."""
    import matplotlib.pyplot as plt

    img_rgb = cv2.cvtColor(result["img"], cv2.COLOR_BGR2RGB)
    edges = result["edges"]
    h, w = img_rgb.shape[:2]

    fig, axes = plt.subplots(2, 2, figsize=(16, 9))
    fig.suptitle(title, fontsize=14, fontweight="bold")

    # 1) Original image
    axes[0, 0].imshow(img_rgb)
    axes[0, 0].set_title("Original (resized)")
    axes[0, 0].axis("off")

    # 2) Canny edges
    axes[0, 1].imshow(edges, cmap="gray")
    axes[0, 1].set_title(f"Canny edges ({np.count_nonzero(edges)} edge pixels)")
    axes[0, 1].axis("off")

    # 3) All raw Hough lines on image
    raw_overlay = img_rgb.copy()
    if len(result["raw_lines"]) > 0:
        # Draw raw lines in faint red
        raw_bgr = result["img"].copy()
        draw_lines(raw_bgr, result["raw_lines"], color=(0, 0, 180), thickness=1)
        raw_overlay = cv2.cvtColor(raw_bgr, cv2.COLOR_BGR2RGB)
    axes[1, 0].imshow(raw_overlay)
    axes[1, 0].set_title(f"Raw Hough lines ({len(result['raw_lines'])})")
    axes[1, 0].axis("off")

    # 4) Clustered + deduplicated lines + intersections
    final_overlay = result["img"].copy()
    if len(result["h_dedup"]) > 0:
        draw_lines(final_overlay, result["h_dedup"], color=(255, 255, 0), thickness=2)  # cyan in BGR
    if len(result["v_dedup"]) > 0:
        draw_lines(final_overlay, result["v_dedup"], color=(255, 0, 255), thickness=2)  # magenta in BGR
    if len(result["intersections"]) > 0:
        draw_intersections(final_overlay, result["intersections"], color=(0, 255, 255), radius=5)
    final_rgb = cv2.cvtColor(final_overlay, cv2.COLOR_BGR2RGB)
    axes[1, 1].imshow(final_rgb)
    stats = (
        f"Cluster 1 (H): {len(result['h_dedup'])} lines  |  "
        f"Cluster 2 (V): {len(result['v_dedup'])} lines  |  "
        f"Intersections: {len(result['intersections'])}  |  "
        f"Time: {result['elapsed']*1000:.0f}ms"
    )
    axes[1, 1].set_title(stats, fontsize=10)
    axes[1, 1].axis("off")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved: {save_path}")
    else:
        plt.show()


# ── Main ───────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Canny + Hough line detection on v4 dataset images "
                    "(no diagonal line elimination)."
    )
    parser.add_argument(
        "images", nargs="*", default=None,
        help="Image path(s). If omitted, processes all images in images/."
    )
    parser.add_argument("--canny-low", type=int, default=90)
    parser.add_argument("--canny-high", type=int, default=400)
    parser.add_argument("--hough-thresh", type=int, default=150)
    parser.add_argument(
        "--save", action="store_true",
        help="Save output PNGs to output/ instead of displaying."
    )
    args = parser.parse_args()

    # Resolve image list
    script_dir = os.path.dirname(os.path.abspath(__file__))
    if args.images:
        image_paths = args.images
    else:
        pattern = os.path.join(script_dir, "images", "*.png")
        image_paths = sorted(glob.glob(pattern))
        if not image_paths:
            print(f"No PNG images found in {os.path.join(script_dir, 'images')}/")
            return

    if args.save:
        out_dir = os.path.join(script_dir, "output")
        os.makedirs(out_dir, exist_ok=True)

    print(f"Processing {len(image_paths)} image(s)")
    print(f"  Canny: low={args.canny_low}, high={args.canny_high}")
    print(f"  Hough threshold: {args.hough_thresh}")
    print(f"  Diagonal line elimination: OFF")
    print()

    for path in image_paths:
        name = os.path.basename(path)
        print(f"[{name}]")

        try:
            result = run_pipeline(
                path,
                canny_low=args.canny_low,
                canny_high=args.canny_high,
                hough_thresh=args.hough_thresh,
            )
        except FileNotFoundError as e:
            print(f"  ERROR: {e}")
            continue

        print(f"  Raw lines: {len(result['raw_lines'])}")
        print(f"  Cluster 1 (H): {len(result['h_lines'])} → {len(result['h_dedup'])} after dedup")
        print(f"  Cluster 2 (V): {len(result['v_lines'])} → {len(result['v_dedup'])} after dedup")
        print(f"  Intersections: {len(result['intersections'])}")
        print(f"  Time: {result['elapsed']*1000:.0f}ms")

        save_path = None
        if args.save:
            save_path = os.path.join(out_dir, name.replace(".png", "_lines.png"))

        visualize(result, title=name, save_path=save_path)


if __name__ == "__main__":
    main()
