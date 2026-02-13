"""
utils.py -- Shared helper functions for the chess dataset pipeline.

Functions:
    load_scene_data()                                    -> dict
    load_camera_positions()                              -> list[dict]
    place_human(obj_name, slot_idx, pos_map, z_override) -> None
    seat_player(obj_name, side, scene_data)              -> None
    project_to_2d(point_3d, scene, camera)               -> (px, py, behind)
    pixel_to_ray(px, py, scene, camera)                  -> (origin, direction)
    get_convex_hull_2d(mesh_obj, scene, cam, dg)         -> [(x,y), ...]
    compute_occlusion(target, poly, scene, cam, dg, n)   -> (bool, float, [str])
    polygon_area(polygon)                                -> float
    polygon_bbox(polygon)                                -> [x, y, w, h]
    point_in_polygon(px, py, polygon)                    -> bool
    sample_points_in_polygon(polygon, n)                 -> [(x,y), ...]
    mmh3_hash_32(name)                                   -> int
    cryptomatte_name_to_float32(name)                    -> float
    extract_cryptomatte_masks(exr00, exr01, names, w, h) -> dict
    mask_to_polygons(mask, tolerance, min_area)           -> [[x,y,...], ...]
    trace_contours(binary)                               -> [[(x,y),...], ...]
    douglas_peucker(points, tolerance)                   -> [(x,y), ...]
"""

import bpy
import json
import math
import os
import random
import struct

import numpy as np

from mathutils import Vector
from bpy_extras.object_utils import world_to_camera_view

# ---------------------------------------------------------------------------
#  Paths
# ---------------------------------------------------------------------------

BASE_DIR = "<ABSOLUTE_PATH_TO_DATASET>"
SCRIPTS_DIR = os.path.join(BASE_DIR, "scripts")


def load_scene_data():
    """Load the pre-computed scene_data.json."""
    with open(os.path.join(BASE_DIR, "scene_data.json")) as f:
        return json.load(f)


def load_camera_positions():
    """Load camera_positions.json."""
    with open(os.path.join(BASE_DIR, "camera_positions.json")) as f:
        return json.load(f)


# ---------------------------------------------------------------------------
#  Human placement
# ---------------------------------------------------------------------------

def place_human(obj_name, slot_index, position_map, z_override=None):
    """
    Move a human to a pre-mapped spectator position and rotate to face the
    board.  *z_override* lets the caller restore the character's original Z
    so seated models keep their chair height and standing models keep theirs.
    """
    obj = bpy.data.objects.get(obj_name)
    if obj is None:
        print(f"[utils] WARNING: object '{obj_name}' not found")
        return
    slot = position_map[slot_index]
    obj.location.x = slot["location"][0]
    obj.location.y = slot["location"][1]
    if z_override is not None:
        obj.location.z = z_override
    obj.rotation_euler = (0.0, 0.0, slot["rotation_z"])


def seat_player(obj_name, side, scene_data):
    """
    Seat a player at the board.
    *side*: 'white' (Y ~ -0.7) or 'black' (Y ~ +0.7).
    """
    obj = bpy.data.objects.get(obj_name)
    if obj is None:
        print(f"[utils] WARNING: object '{obj_name}' not found")
        return
    if side == "white":
        seat = scene_data["white_side_seat"]
        rot_z = scene_data["white_side_rot_z"]
    else:
        seat = scene_data["black_side_seat"]
        rot_z = scene_data["black_side_rot_z"]
    obj.location = Vector(seat)
    obj.rotation_euler = (0.0, 0.0, rot_z)


# ---------------------------------------------------------------------------
#  Projection helpers
# ---------------------------------------------------------------------------

def project_to_2d(point_3d, scene, camera):
    """
    Project a 3D world point to 2D pixel coordinates.

    Returns (px, py, behind_camera).
    Pixel convention: origin at top-left, x right, y down.
    """
    co = world_to_camera_view(scene, camera, Vector(point_3d))
    res_x = scene.render.resolution_x
    res_y = scene.render.resolution_y
    px = co.x * res_x
    py = (1.0 - co.y) * res_y
    behind = co.z <= 0
    return (round(px, 2), round(py, 2), behind)


def pixel_to_ray(px, py, scene, camera):
    """Convert pixel (px, py) to a world-space ray (origin, direction)."""
    render = scene.render
    res_x = render.resolution_x
    res_y = render.resolution_y
    cam = camera.data

    origin = camera.matrix_world.translation.copy()

    # Sensor geometry
    sensor_width = cam.sensor_width
    aspect = res_x / res_y
    if cam.sensor_fit == "AUTO":
        sensor_fit = "HORIZONTAL" if res_x >= res_y else "VERTICAL"
    else:
        sensor_fit = cam.sensor_fit
    if sensor_fit == "HORIZONTAL":
        sensor_height = sensor_width / aspect
    else:
        sensor_height = sensor_width
        sensor_width = sensor_height * aspect

    # Normalised image coords (Blender: origin at bottom-left, 0-1)
    nx = px / res_x
    ny = 1.0 - (py / res_y)

    # Direction in camera local space (camera looks down -Z, X right, Y up)
    sensor_x = (nx - 0.5) * sensor_width
    sensor_y = (ny - 0.5) * sensor_height
    focal = cam.lens
    local_dir = Vector((sensor_x, sensor_y, -focal)).normalized()
    world_dir = (camera.matrix_world.to_3x3() @ local_dir).normalized()

    return origin, world_dir


# ---------------------------------------------------------------------------
#  Convex hull — Andrew's monotone chain
# ---------------------------------------------------------------------------

def _cross2d(o, a, b):
    return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])


def convex_hull(points):
    """Return the convex hull of 2D *points* as [(x, y), ...]."""
    pts = sorted(set(points))
    if len(pts) <= 2:
        return list(pts)

    lower = []
    for p in pts:
        while len(lower) >= 2 and _cross2d(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)

    upper = []
    for p in reversed(pts):
        while len(upper) >= 2 and _cross2d(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)

    return lower[:-1] + upper[:-1]


def get_convex_hull_2d(mesh_obj, scene, camera, depsgraph):
    """
    Project the evaluated mesh vertices of *mesh_obj* into 2D pixel space
    and return the convex hull polygon as [(x, y), ...].
    """
    eval_obj = mesh_obj.evaluated_get(depsgraph)
    mesh = eval_obj.to_mesh()

    pts = []
    for v in mesh.vertices:
        world_co = mesh_obj.matrix_world @ v.co
        px, py, behind = project_to_2d(world_co, scene, camera)
        if not behind:
            pts.append((px, py))

    eval_obj.to_mesh_clear()

    if len(pts) < 3:
        return []
    return convex_hull(pts)


# ---------------------------------------------------------------------------
#  Polygon geometry
# ---------------------------------------------------------------------------

def polygon_area(poly):
    """Shoelace formula for the area of a simple polygon."""
    n = len(poly)
    if n < 3:
        return 0.0
    area = 0.0
    for i in range(n):
        j = (i + 1) % n
        area += poly[i][0] * poly[j][1]
        area -= poly[j][0] * poly[i][1]
    return abs(area) / 2.0


def polygon_bbox(poly):
    """COCO-style bbox: [x_min, y_min, width, height]."""
    if not poly:
        return [0, 0, 0, 0]
    xs = [p[0] for p in poly]
    ys = [p[1] for p in poly]
    x0, y0 = min(xs), min(ys)
    return [round(x0, 2), round(y0, 2),
            round(max(xs) - x0, 2), round(max(ys) - y0, 2)]


def point_in_polygon(px, py, poly):
    """Ray-casting point-in-polygon test."""
    n = len(poly)
    inside = False
    j = n - 1
    for i in range(n):
        xi, yi = poly[i]
        xj, yj = poly[j]
        if ((yi > py) != (yj > py)) and \
           (px < (xj - xi) * (py - yi) / (yj - yi) + xi):
            inside = not inside
        j = i
    return inside


def sample_points_in_polygon(poly, n_samples):
    """Rejection-sample *n_samples* uniform points inside *poly*."""
    bbox = polygon_bbox(poly)
    x0, y0, w, h = bbox
    if w <= 0 or h <= 0:
        return []
    samples = []
    cap = n_samples * 30
    attempts = 0
    while len(samples) < n_samples and attempts < cap:
        x = x0 + random.random() * w
        y = y0 + random.random() * h
        if point_in_polygon(x, y, poly):
            samples.append((x, y))
        attempts += 1
    return samples


# ---------------------------------------------------------------------------
#  Occlusion computation
# ---------------------------------------------------------------------------

def compute_occlusion(target_obj, polygon_2d, scene, camera, depsgraph,
                      n_samples=80, exclude_occluders=None):
    """
    Estimate occlusion by raycasting from the camera through sampled points
    inside *polygon_2d*.

    *exclude_occluders* is an optional set of object names that should NOT
    count as occluders.  For chess-piece annotations pass
    ``{"Board", "WoodenTable_02"}`` so that rays that fall inside the convex
    hull but miss the piece geometry (hitting the board instead) are excluded
    from both numerator and denominator.

    Returns (occluded: bool, occlusion_rate: float, occluded_by: list[str]).
    """
    if exclude_occluders is None:
        exclude_occluders = set()

    if not polygon_2d or len(polygon_2d) < 3:
        return False, 0.0, []

    samples = sample_points_in_polygon(polygon_2d, n_samples)
    if not samples:
        return False, 0.0, []

    occluded_count = 0
    valid_count = 0          # rays that hit target OR a real occluder
    occluders = set()

    for sx, sy in samples:
        origin, direction = pixel_to_ray(sx, sy, scene, camera)
        ray_origin = origin + direction * 0.001

        hit, loc, norm, idx, hit_obj, mat = scene.ray_cast(
            depsgraph, ray_origin, direction
        )

        if not hit or hit_obj is None:
            continue                     # ray missed everything

        if hit_obj.name == target_obj.name:
            valid_count += 1             # visible sample
        elif hit_obj.name in exclude_occluders:
            pass                         # convex-hull overshoot — ignore
        else:
            occluded_count += 1
            valid_count += 1
            occluders.add(hit_obj.name)

    total = valid_count
    rate = occluded_count / total if total > 0 else 0.0
    return (rate > 0), round(rate, 4), sorted(occluders)


# ---------------------------------------------------------------------------
#  Cryptomatte: MurmurHash3 (32-bit, seed=0)
# ---------------------------------------------------------------------------

def mmh3_hash_32(name):
    """
    Pure-Python MurmurHash3 32-bit hash (seed=0).

    Matches the C implementation used by Blender / Cryptomatte for
    mapping object names to 32-bit hash values.
    """
    key = name.encode("utf-8")
    length = len(key)
    seed = 0
    c1 = 0xCC9E2D51
    c2 = 0x1B873593
    h1 = seed
    MASK32 = 0xFFFFFFFF

    # Process body (4-byte chunks)
    n_blocks = length // 4
    for i in range(n_blocks):
        k1 = struct.unpack_from("<I", key, i * 4)[0]
        k1 = (k1 * c1) & MASK32
        k1 = ((k1 << 15) | (k1 >> 17)) & MASK32
        k1 = (k1 * c2) & MASK32
        h1 ^= k1
        h1 = ((h1 << 13) | (h1 >> 19)) & MASK32
        h1 = (h1 * 5 + 0xE6546B64) & MASK32

    # Process tail
    tail_start = n_blocks * 4
    tail_len = length & 3
    k1 = 0
    if tail_len >= 3:
        k1 ^= key[tail_start + 2] << 16
    if tail_len >= 2:
        k1 ^= key[tail_start + 1] << 8
    if tail_len >= 1:
        k1 ^= key[tail_start]
        k1 = (k1 * c1) & MASK32
        k1 = ((k1 << 15) | (k1 >> 17)) & MASK32
        k1 = (k1 * c2) & MASK32
        h1 ^= k1

    # Finalization mix
    h1 ^= length
    h1 ^= (h1 >> 16)
    h1 = (h1 * 0x85EBCA6B) & MASK32
    h1 ^= (h1 >> 13)
    h1 = (h1 * 0xC2B2AE35) & MASK32
    h1 ^= (h1 >> 16)

    return h1


def cryptomatte_name_to_float32(name):
    """
    Convert an object name to its Cryptomatte float32 hash ID.

    Matches Blender's implementation: compute MurmurHash3, then clamp the
    IEEE 754 exponent to [1, 254] to avoid denormals, infinity, and NaN.
    The sign bit and mantissa are preserved.
    """
    h = mmh3_hash_32(name)
    if h == 0:
        h = 1
    mantissa = h & ((1 << 23) - 1)
    exponent = (h >> 23) & 0xFF
    exponent = max(exponent, 1)
    exponent = min(exponent, 254)
    sign = h & 0x80000000
    h = sign | (exponent << 23) | mantissa
    return struct.unpack("f", struct.pack("I", h))[0]


# ---------------------------------------------------------------------------
#  Cryptomatte: EXR mask extraction
# ---------------------------------------------------------------------------

def extract_cryptomatte_masks(exr_00_path, exr_01_path, obj_names, width, height):
    """
    Load two Cryptomatte EXR files and extract per-object binary masks.

    Parameters
    ----------
    exr_00_path : str   Path to CryptoObject00.exr (RGBA = hash1, cov1, hash2, cov2)
    exr_01_path : str   Path to CryptoObject01.exr (RGBA = hash3, cov3, hash4, cov4)
    obj_names   : list  Object names to extract masks for
    width, height : int Render resolution

    Returns
    -------
    dict : {obj_name: [[x1,y1,x2,y2,...], ...]}  COCO polygon segmentations
    """
    def load_exr_pixels(path):
        img = bpy.data.images.load(path, check_existing=False)
        pixels = np.array(img.pixels[:], dtype=np.float32)
        bpy.data.images.remove(img)
        # Blender stores pixels bottom-to-top; reshape and flip
        pixels = pixels.reshape((height, width, 4))
        pixels = np.flipud(pixels)
        return pixels

    exr0 = load_exr_pixels(exr_00_path)
    exr1 = load_exr_pixels(exr_01_path)

    # Channels: R=hash_slot_0, G=coverage_slot_0, B=hash_slot_1, A=coverage_slot_1
    hash_channels = [
        exr0[:, :, 0], exr0[:, :, 2],  # CryptoObject00 slots 0,1
        exr1[:, :, 0], exr1[:, :, 2],  # CryptoObject01 slots 0,1
    ]
    cov_channels = [
        exr0[:, :, 1], exr0[:, :, 3],
        exr1[:, :, 1], exr1[:, :, 3],
    ]

    results = {}
    for name in obj_names:
        target_hash = np.float32(cryptomatte_name_to_float32(name))

        # Sum coverage across all 4 hash slots where hash matches
        coverage = np.zeros((height, width), dtype=np.float32)
        for h_ch, c_ch in zip(hash_channels, cov_channels):
            match = h_ch == target_hash
            coverage += np.where(match, c_ch, 0.0)

        mask = coverage > 0.5
        if not np.any(mask):
            continue

        polygons = mask_to_polygons(mask)
        if polygons:
            results[name] = polygons

    return results


# ---------------------------------------------------------------------------
#  Cryptomatte: mask → COCO polygons
# ---------------------------------------------------------------------------

def mask_to_polygons(mask, tolerance=2.0, min_area=100):
    """
    Convert a binary mask to a list of COCO-format polygon coordinate lists.

    Each polygon is [x1, y1, x2, y2, ...] in pixel coordinates.
    Polygons with area < min_area are filtered out.
    """
    # Pad with 1px zero border to ensure contours are closed
    padded = np.pad(mask.astype(np.uint8), 1, mode="constant", constant_values=0)
    contours = trace_contours(padded)

    polygons = []
    for contour in contours:
        # Adjust for padding offset
        pts = [(x - 1, y - 1) for x, y in contour]

        if len(pts) < 3:
            continue

        # Simplify
        pts = douglas_peucker(pts, tolerance)
        if len(pts) < 3:
            continue

        # Check area
        area = polygon_area(pts)
        if area < min_area:
            continue

        # Flatten to COCO format
        flat = []
        for x, y in pts:
            flat.extend([round(float(x), 2), round(float(y), 2)])
        polygons.append(flat)

    return polygons


# ---------------------------------------------------------------------------
#  Contour tracing (Moore neighborhood)
# ---------------------------------------------------------------------------

def trace_contours(binary):
    """
    Trace outer boundary contours in a binary image using Moore
    neighborhood tracing.

    Returns a list of contours, each a list of (x, y) tuples.
    Handles multiple disconnected components.
    """
    h, w = binary.shape
    visited = np.zeros_like(binary, dtype=bool)
    contours = []

    # 8-connected neighbor offsets (clockwise from right)
    #               E     SE    S     SW    W     NW    N     NE
    dx = np.array([ 1,  1,  0, -1, -1, -1,  0,  1], dtype=int)
    dy = np.array([ 0,  1,  1,  1,  0, -1, -1, -1], dtype=int)

    for sy in range(h):
        for sx in range(w):
            if binary[sy, sx] == 0 or visited[sy, sx]:
                continue
            # Check that this is a border pixel (has at least one 0-neighbor)
            is_border = False
            for d in range(8):
                ny, nx = sy + dy[d], sx + dx[d]
                if ny < 0 or ny >= h or nx < 0 or nx >= w or binary[ny, nx] == 0:
                    is_border = True
                    break
            if not is_border:
                visited[sy, sx] = True
                continue

            # Start tracing from this border pixel
            contour = []
            cx, cy = sx, sy

            # Find the direction of the first background neighbor (entry dir)
            start_dir = 0
            for d in range(8):
                ny, nx = cy + dy[d], cx + dx[d]
                if ny < 0 or ny >= h or nx < 0 or nx >= w or binary[ny, nx] == 0:
                    start_dir = (d + 1) % 8  # start scanning from next dir
                    break

            first_x, first_y = cx, cy
            direction = start_dir
            max_iter = h * w * 2

            for _ in range(max_iter):
                contour.append((cx, cy))
                visited[cy, cx] = True

                # Scan neighbors starting from (direction + 5) % 8
                # (backtrack: turn around ~135 degrees)
                scan_start = (direction + 5) % 8
                found = False
                for i in range(8):
                    d = (scan_start + i) % 8
                    ny, nx = cy + dy[d], cx + dx[d]
                    if 0 <= ny < h and 0 <= nx < w and binary[ny, nx] != 0:
                        cx, cy = nx, ny
                        direction = d
                        found = True
                        break

                if not found:
                    break  # isolated pixel

                if cx == first_x and cy == first_y:
                    break

            if len(contour) >= 3:
                contours.append(contour)

            # Mark interior pixels connected to this contour as visited
            # (flood fill would be overkill; the outer scan will skip them)

    return contours


# ---------------------------------------------------------------------------
#  Douglas-Peucker polygon simplification
# ---------------------------------------------------------------------------

def douglas_peucker(points, tolerance):
    """
    Simplify a polygon using the Douglas-Peucker algorithm.

    *points* is a list of (x, y) tuples forming a closed polygon.
    Returns a simplified list of (x, y) tuples.
    """
    if len(points) <= 3:
        return points

    def _perp_dist(p, a, b):
        """Perpendicular distance from point *p* to line segment *a*-*b*."""
        dx = b[0] - a[0]
        dy = b[1] - a[1]
        length_sq = dx * dx + dy * dy
        if length_sq == 0:
            return math.hypot(p[0] - a[0], p[1] - a[1])
        t = max(0, min(1, ((p[0] - a[0]) * dx + (p[1] - a[1]) * dy) / length_sq))
        proj_x = a[0] + t * dx
        proj_y = a[1] + t * dy
        return math.hypot(p[0] - proj_x, p[1] - proj_y)

    def _simplify(pts, start, end, tol):
        max_dist = 0.0
        max_idx = start
        for i in range(start + 1, end):
            d = _perp_dist(pts[i], pts[start], pts[end])
            if d > max_dist:
                max_dist = d
                max_idx = i

        if max_dist > tol:
            left = _simplify(pts, start, max_idx, tol)
            right = _simplify(pts, max_idx, end, tol)
            return left[:-1] + right
        else:
            return [pts[start], pts[end]]

    # Close the polygon for processing, then remove duplicate end
    closed = list(points) + [points[0]]
    simplified = _simplify(closed, 0, len(closed) - 1, tolerance)
    if (simplified[-1][0] == simplified[0][0] and
            simplified[-1][1] == simplified[0][1]):
        simplified = simplified[:-1]

    return simplified
