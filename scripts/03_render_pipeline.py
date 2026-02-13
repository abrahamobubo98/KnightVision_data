"""
03_render_pipeline.py -- Full render + annotation pipeline (v4, Cryptomatte).

Uses 36 hardcoded configurations from 04_configurations.py for scene variety.
Each of the 25 camera positions gets a different configuration:

  Ring 1 (el=30deg, 12 cams): Round 1 configs 0-11  (all crowded)
  Ring 2 (el=60deg, 12 cams): Round 2 configs 12-23 (mixed styles)
  Top    (el=83deg, 1 cam):   Config 33             (clear)

v4 changes: Cryptomatte pixel-level segmentation replaces convex hull.
Cryptomatte passes are saved as EXR via File Output nodes, then parsed
with numpy to produce pixel-accurate COCO polygon masks.

Output:
    chess_dataset/v4/images/*.png
    chess_dataset/v4/annotations.json
    chess_dataset/v4/progress.json

Run from Blender:
    blender --background scene.blend --python chess_dataset/scripts/03_render_pipeline.py
"""

import bpy
import glob
import importlib.util
import json
import math
import os
import shutil
import sys
import time

from mathutils import Vector

# Ensure utils is importable
SCRIPTS_DIR = "<ABSOLUTE_PATH_TO_DATASET>/scripts"
sys.path.insert(0, SCRIPTS_DIR)

from utils import (
    load_scene_data,
    load_camera_positions,
    project_to_2d,
    get_convex_hull_2d,
    compute_occlusion,
    polygon_area,
    polygon_bbox,
    extract_cryptomatte_masks,
)

# Load 04_configurations.py (filename starts with digit, can't import directly)
_spec = importlib.util.spec_from_file_location(
    "configurations",
    os.path.join(SCRIPTS_DIR, "04_configurations.py"),
)
_configurations_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_configurations_mod)

apply_configuration = _configurations_mod.apply_configuration
CONFIGURATIONS = _configurations_mod.CONFIGURATIONS

# ---------------------------------------------------------------------------
#  Config
# ---------------------------------------------------------------------------

BASE_DIR = "<ABSOLUTE_PATH_TO_DATASET>"
V4_DIR = os.path.join(BASE_DIR, "v4")
IMAGES_DIR = os.path.join(V4_DIR, "images")
ANNOTATIONS_PATH = os.path.join(V4_DIR, "annotations.json")
PROGRESS_PATH = os.path.join(V4_DIR, "progress.json")
CRYPTO_TMP_DIR = os.path.join(V4_DIR, "_crypto_tmp")

# Set to a number for test run; set to None for full 25-image run
TEST_MAX_IMAGES = 5

RENDER_WIDTH = 4096
RENDER_HEIGHT = 2304

BOARD_CENTER = Vector((0.0, 0.0, 0.0033))

# Occlusion sampling density
PIECE_OCCLUSION_SAMPLES = 80
BOARD_OCCLUSION_SAMPLES = 200


# ---------------------------------------------------------------------------
#  COCO category definitions
# ---------------------------------------------------------------------------

CATEGORY_MAP = {
    "chessboard":    1,
    "white_king":    2,
    "white_queen":   3,
    "white_rook":    4,
    "white_bishop":  5,
    "white_knight":  6,
    "white_pawn":    7,
    "black_king":    8,
    "black_queen":   9,
    "black_rook":   10,
    "black_bishop": 11,
    "black_knight": 12,
    "black_pawn":   13,
}


def build_coco_categories():
    """Build the ``categories`` list for the COCO JSON."""
    cats = []
    for name, cid in sorted(CATEGORY_MAP.items(), key=lambda x: x[1]):
        cat = {"id": cid, "name": name, "supercategory": "chess"}
        if name == "chessboard":
            cat["keypoints"] = [
                "corner_a1", "corner_a8", "corner_h1", "corner_h8"
            ]
            cat["skeleton"] = [[0, 1], [1, 3], [3, 2], [2, 0]]
        cats.append(cat)
    return cats


# ---------------------------------------------------------------------------
#  Camera helpers
# ---------------------------------------------------------------------------

def set_camera(camera_obj, position, target):
    """Move *camera_obj* to *position* and aim it at *target*."""
    camera_obj.location = Vector(position)
    direction = target - camera_obj.location
    rot_quat = direction.to_track_quat('-Z', 'Y')
    camera_obj.rotation_euler = rot_quat.to_euler()


# ---------------------------------------------------------------------------
#  Scene safety helpers
# ---------------------------------------------------------------------------

def ensure_scene_objects_visible():
    """Make sure key scene objects (board, table, lights, etc.) are visible."""
    essential = ["Board", "WoodenTable_02"]
    for name in essential:
        obj = bpy.data.objects.get(name)
        if obj:
            obj.hide_viewport = False
            obj.hide_render = False


def restore_piece_visibility(scene_data):
    """
    Restore chess piece visibility to match scene_data.json.

    Pieces marked ``visible: true`` are shown; captured pieces
    (``visible: false``) are hidden from render.
    """
    for piece in scene_data["pieces"]:
        obj = bpy.data.objects.get(piece["object_name"])
        if obj is None:
            continue
        if piece["visible"]:
            obj.hide_viewport = False
            obj.hide_render = False
        else:
            obj.hide_viewport = True
            obj.hide_render = True


# ---------------------------------------------------------------------------
#  Render plan: 25 camera positions -> 25 config indices
# ---------------------------------------------------------------------------

def build_render_plan(cam_positions):
    """
    Map each camera position to a configuration index.

    Ring 1 (indices 0-11, el=30):  configs  0-11  (Round 1, all crowded)
    Ring 2 (indices 12-23, el=60): configs 12-23  (Round 2, mixed)
    Top    (index 24, el=83):      config  33     (r3_az270_clr, clear)
    """
    plan = []
    for cam in cam_positions:
        idx = cam["index"]
        if idx < 12:
            config_idx = idx
        elif idx < 24:
            config_idx = idx
        else:
            config_idx = 33
        plan.append((cam, config_idx))
    return plan


# ---------------------------------------------------------------------------
#  Progress / crash recovery
# ---------------------------------------------------------------------------

def load_progress():
    """Load progress.json if it exists, returning set of completed filenames."""
    if os.path.exists(PROGRESS_PATH):
        with open(PROGRESS_PATH) as f:
            data = json.load(f)
        return set(data.get("completed", []))
    return set()


def save_progress(completed_set, coco):
    """Persist progress so we can resume after a crash."""
    data = {
        "completed": sorted(completed_set),
        "images_done": len(completed_set),
        "images_total": 25,
        "annotations_so_far": len(coco["annotations"]),
    }
    with open(PROGRESS_PATH, "w") as f:
        json.dump(data, f, indent=2)


# ---------------------------------------------------------------------------
#  Cryptomatte compositor setup
# ---------------------------------------------------------------------------

def setup_compositor_cryptomatte(scene):
    """
    Enable the Cryptomatte Object pass and create a File Output node
    to save CryptoObject00 and CryptoObject01 as single-layer EXR files.

    Returns the File Output node so the caller can set base_path per shot.
    """
    view_layer = scene.view_layers[0]
    view_layer.use_pass_cryptomatte_object = True
    # depth = number of ID-coverage pairs; each render output holds 2 pairs,
    # so depth=4 → CryptoObject00 (2 pairs) + CryptoObject01 (2 pairs).
    view_layer.pass_cryptomatte_depth = 4

    scene.use_nodes = True
    tree = scene.node_tree

    # Force node tree to refresh so new Cryptomatte outputs appear
    tree.update_tag()
    bpy.context.view_layer.update()

    # Find the existing Render Layers node
    rl_node = None
    for node in tree.nodes:
        if node.type == "R_LAYERS":
            rl_node = node
            break
    if rl_node is None:
        raise RuntimeError("No Render Layers node found in compositor")

    # Debug: print available outputs so we can diagnose issues
    print("[crypto] Render Layers outputs:",
          [o.name for o in rl_node.outputs])

    # Create File Output node
    fo_node = tree.nodes.new(type="CompositorNodeOutputFile")
    fo_node.name = "CryptoFileOutput"
    fo_node.label = "Cryptomatte EXR Output"
    fo_node.location = (rl_node.location.x + 400, rl_node.location.y - 300)

    # Configure for single-layer EXR (float32, ZIP compression)
    fo_node.format.file_format = "OPEN_EXR"
    fo_node.format.color_mode = "RGBA"
    fo_node.format.color_depth = "32"
    fo_node.format.exr_codec = "ZIP"

    # Remove default input slot, add our two slots
    fo_node.file_slots.clear()
    fo_node.file_slots.new("CryptoObject00")
    fo_node.file_slots.new("CryptoObject01")

    # Connect Render Layers → File Output
    tree.links.new(rl_node.outputs["CryptoObject00"], fo_node.inputs["CryptoObject00"])
    tree.links.new(rl_node.outputs["CryptoObject01"], fo_node.inputs["CryptoObject01"])

    return fo_node


def teardown_compositor_cryptomatte(scene, fo_node):
    """Remove the File Output node and disable the Cryptomatte pass."""
    if fo_node is not None:
        tree = scene.node_tree
        tree.nodes.remove(fo_node)
    view_layer = scene.view_layers[0]
    view_layer.use_pass_cryptomatte_object = False


def find_crypto_exr_paths(shot_dir):
    """
    Find Cryptomatte EXR files in *shot_dir* by globbing.

    Blender's File Output node naming varies (frame padding, separators),
    so we search for files matching CryptoObject00*.exr and CryptoObject01*.exr.

    Returns (exr_00_path, exr_01_path) or (None, None) if not found.
    """
    # List everything in the directory for diagnostics
    all_files = os.listdir(shot_dir) if os.path.isdir(shot_dir) else []
    print(f"\n[crypto] Files in {shot_dir}: {all_files}", flush=True)

    exr_00_matches = sorted(glob.glob(os.path.join(shot_dir, "CryptoObject00*.exr")))
    exr_01_matches = sorted(glob.glob(os.path.join(shot_dir, "CryptoObject01*.exr")))

    exr_00 = exr_00_matches[0] if exr_00_matches else None
    exr_01 = exr_01_matches[0] if exr_01_matches else None

    if not exr_00 or not exr_01:
        # Broader search: any .exr file
        all_exr = sorted(glob.glob(os.path.join(shot_dir, "*.exr")))
        print(f"[crypto] All EXR files: {all_exr}", flush=True)
        if not all_exr:
            # Check parent directory too (Blender may ignore base_path)
            parent_exr = sorted(glob.glob(os.path.join(os.path.dirname(shot_dir), "*.exr")))
            if parent_exr:
                print(f"[crypto] EXR in parent dir: {parent_exr}", flush=True)

    return exr_00, exr_01


# ---------------------------------------------------------------------------
#  Annotation: board
# ---------------------------------------------------------------------------

def annotate_board(scene, camera, depsgraph, image_id, ann_id,
                   scene_data):
    """Return a list containing the chessboard COCO annotation."""
    corners = scene_data["board_corners"]
    res_x = scene.render.resolution_x
    res_y = scene.render.resolution_y

    # Project the four corners
    projected = {}
    for cname in ["GridCorner_a1", "GridCorner_a8",
                   "GridCorner_h1", "GridCorner_h8"]:
        coord = corners[cname]
        px, py, behind = project_to_2d(coord, scene, camera)
        vis = 0 if behind else 2
        if not behind and (px < 0 or px > res_x or py < 0 or py > res_y):
            vis = 0          # in frame but out of pixel bounds
        projected[cname] = (px, py, vis)

    # Segmentation quadrilateral  (a1 -> h1 -> h8 -> a8)
    seg_order = ["GridCorner_a1", "GridCorner_h1",
                 "GridCorner_h8", "GridCorner_a8"]
    seg_poly = [(projected[c][0], projected[c][1]) for c in seg_order]
    seg_flat = []
    for x, y in seg_poly:
        seg_flat.extend([round(x, 2), round(y, 2)])

    # Keypoints  (a1, a8, h1, h8)
    kp_order = ["GridCorner_a1", "GridCorner_a8",
                "GridCorner_h1", "GridCorner_h8"]
    keypoints = []
    num_kp = 0
    for cname in kp_order:
        px, py, vis = projected[cname]
        keypoints.extend([round(px, 2), round(py, 2), vis])
        if vis > 0:
            num_kp += 1

    area = polygon_area(seg_poly)
    bbox = polygon_bbox(seg_poly)

    # Occlusion via raycasting
    board_obj = bpy.data.objects.get("Board")
    occ, occ_rate, occ_by = compute_occlusion(
        board_obj, seg_poly, scene, camera, depsgraph,
        n_samples=BOARD_OCCLUSION_SAMPLES,
    )

    return [{
        "id":              ann_id,
        "image_id":        image_id,
        "category_id":     CATEGORY_MAP["chessboard"],
        "bbox":            bbox,
        "area":            round(area, 2),
        "segmentation":    [seg_flat],
        "iscrowd":         0,
        "occluded":        occ,
        "occlusion_rate":  occ_rate,
        "occluded_by":     occ_by,
        "keypoints":       keypoints,
        "num_keypoints":   num_kp,
    }]


# ---------------------------------------------------------------------------
#  Annotation: pieces
# ---------------------------------------------------------------------------

def annotate_pieces(scene, camera, depsgraph, image_id, ann_id_start,
                    pieces_data, cryptomatte_masks=None):
    """Return COCO annotations for every visible chess piece.

    If *cryptomatte_masks* is provided (dict of obj_name → list of COCO
    polygon lists), use pixel-accurate masks. Otherwise fall back to
    convex-hull projection.
    """
    annotations = []
    ann_id = ann_id_start

    for piece in pieces_data:
        obj_name = piece["object_name"]
        obj = bpy.data.objects.get(obj_name)
        if obj is None:
            continue
        if not obj.visible_get() or obj.hide_render:
            continue

        # --- Segmentation: prefer Cryptomatte, fall back to convex hull ---
        if cryptomatte_masks and obj_name in cryptomatte_masks:
            seg_polygons = cryptomatte_masks[obj_name]  # list of flat polys
        else:
            hull = get_convex_hull_2d(obj, scene, camera, depsgraph)
            if len(hull) < 3:
                continue
            seg_flat = []
            for x, y in hull:
                seg_flat.extend([round(x, 2), round(y, 2)])
            seg_polygons = [seg_flat]

        if not seg_polygons:
            continue

        # Multi-polygon area and bbox: union across all polygons
        total_area = 0.0
        all_xs = []
        all_ys = []
        for flat_poly in seg_polygons:
            # Convert flat [x1,y1,x2,y2,...] to [(x,y),...]
            pts = [(flat_poly[i], flat_poly[i + 1])
                   for i in range(0, len(flat_poly), 2)]
            total_area += polygon_area(pts)
            all_xs.extend(flat_poly[0::2])
            all_ys.extend(flat_poly[1::2])

        if total_area < 1.0:
            continue

        x0, y0 = min(all_xs), min(all_ys)
        bbox = [round(x0, 2), round(y0, 2),
                round(max(all_xs) - x0, 2), round(max(all_ys) - y0, 2)]

        # Occlusion — sample from the first (largest) polygon
        first_pts = [(seg_polygons[0][i], seg_polygons[0][i + 1])
                     for i in range(0, len(seg_polygons[0]), 2)]
        occ, occ_rate, occ_by = compute_occlusion(
            obj, first_pts, scene, camera, depsgraph,
            n_samples=PIECE_OCCLUSION_SAMPLES,
            exclude_occluders={"Board", "WoodenTable_02"},
        )

        cat_name = piece["category_name"]
        cat_id = CATEGORY_MAP.get(cat_name)
        if cat_id is None:
            continue

        annotations.append({
            "id":              ann_id,
            "image_id":        image_id,
            "category_id":     cat_id,
            "bbox":            bbox,
            "area":            round(total_area, 2),
            "segmentation":    seg_polygons,
            "iscrowd":         0,
            "occluded":        occ,
            "occlusion_rate":  occ_rate,
            "occluded_by":     occ_by,
        })
        ann_id += 1

    return annotations


# ---------------------------------------------------------------------------
#  Main pipeline
# ---------------------------------------------------------------------------

def main():
    os.makedirs(IMAGES_DIR, exist_ok=True)
    os.makedirs(CRYPTO_TMP_DIR, exist_ok=True)

    scene_data = load_scene_data()
    cam_positions = load_camera_positions()
    render_plan = build_render_plan(cam_positions)

    # Limit images for test run
    if TEST_MAX_IMAGES is not None:
        render_plan = render_plan[:TEST_MAX_IMAGES]

    scene = bpy.context.scene
    camera = bpy.data.objects["Camera"]

    # ---- Render settings ----
    scene.render.resolution_x = RENDER_WIDTH
    scene.render.resolution_y = RENDER_HEIGHT
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = "PNG"

    # ---- Cryptomatte compositor setup ----
    crypto_fo_node = setup_compositor_cryptomatte(scene)

    # ---- Crash recovery ----
    completed = load_progress()

    # ---- COCO scaffold ----
    coco = {
        "images": [],
        "categories": build_coco_categories(),
        "annotations": [],
    }
    ann_id = 1

    # If resuming, fast-forward ann_id past already-completed images
    # (estimate ~33 annotations per image as upper bound)
    if completed:
        ann_id = len(completed) * 40 + 1

    total = len(render_plan)
    limit_str = f" (TEST: {TEST_MAX_IMAGES})" if TEST_MAX_IMAGES else ""

    print(f"\n{'=' * 60}")
    print(f"  CHESS DATASET RENDER PIPELINE  v4 (Cryptomatte)")
    print(f"  Resolution : {RENDER_WIDTH} x {RENDER_HEIGHT}")
    print(f"  Cameras    : {total}{limit_str}")
    print(f"  Configs    : {len(CONFIGURATIONS)} available")
    print(f"  Output     : {V4_DIR}")
    if completed:
        print(f"  Resuming   : {len(completed)} already done")
    print(f"{'=' * 60}\n")

    # ---- Safety: ensure non-human scene objects are visible ----
    ensure_scene_objects_visible()
    restore_piece_visibility(scene_data)

    pipeline_start = time.time()

    try:
        for step, (cam_pos, config_idx) in enumerate(render_plan):
            shot_idx = cam_pos["index"]
            filename = cam_pos["filename"]
            filepath = os.path.join(IMAGES_DIR, filename)

            # ---- Skip if already completed (crash recovery) ----
            if filename in completed:
                print(f"[{step + 1:2d}/{total}] {filename} -- SKIPPED (already done)")
                continue

            t0 = time.time()
            config = CONFIGURATIONS[config_idx]
            print(f"[{step + 1:2d}/{total}] {filename}  "
                  f"config={config['name']} ({config['style']}) ... ",
                  end="", flush=True)

            # --- 1. Apply configuration (seats players, places spectators) ---
            apply_configuration(config_idx)

            # --- 2. Restore piece visibility (config hides all humans, not pieces,
            #         but ensure captured pieces stay hidden) ---
            restore_piece_visibility(scene_data)

            # --- 3. Ensure board/table visible ---
            ensure_scene_objects_visible()

            # --- 4. Camera setup ---
            set_camera(camera, cam_pos["location"], BOARD_CENTER)
            bpy.context.view_layer.update()

            # --- 5. Set per-shot Cryptomatte EXR output directory ---
            shot_crypto_dir = os.path.join(CRYPTO_TMP_DIR, f"shot_{shot_idx:03d}")
            os.makedirs(shot_crypto_dir, exist_ok=True)
            crypto_fo_node.base_path = shot_crypto_dir

            # --- 6. Render (produces PNG + 2 EXR files) ---
            scene.render.filepath = filepath
            bpy.ops.render.render(write_still=True)
            render_time = time.time() - t0
            print(f"render {render_time:.1f}s ... ", end="", flush=True)

            # --- 7. Extract Cryptomatte masks ---
            exr_00, exr_01 = find_crypto_exr_paths(shot_crypto_dir)
            cryptomatte_masks = None

            if exr_00 and exr_01:
                visible_pieces = [p for p in scene_data["pieces"] if p["visible"]]
                obj_names = [p["object_name"] for p in visible_pieces]
                cryptomatte_masks = extract_cryptomatte_masks(
                    exr_00, exr_01, obj_names, RENDER_WIDTH, RENDER_HEIGHT
                )
                n_masks = len(cryptomatte_masks)
                print(f"crypto {n_masks} masks ... ", end="", flush=True)
            else:
                print(f"crypto MISSING exr_00={exr_00} exr_01={exr_01} (fallback) ... ",
                      end="", flush=True)

            # --- 8. Annotations ---
            depsgraph = bpy.context.evaluated_depsgraph_get()
            image_id = shot_idx + 1

            # Image entry (with config metadata)
            coco["images"].append({
                "id":           image_id,
                "file_name":    f"images/{filename}",
                "width":        RENDER_WIDTH,
                "height":       RENDER_HEIGHT,
                "config_name":  config["name"],
                "config_style": config["style"],
            })

            # Board (unchanged — uses projected corners)
            board_anns = annotate_board(
                scene, camera, depsgraph, image_id, ann_id, scene_data
            )
            coco["annotations"].extend(board_anns)
            ann_id += len(board_anns)

            # Pieces (with Cryptomatte masks)
            piece_anns = annotate_pieces(
                scene, camera, depsgraph, image_id, ann_id,
                scene_data["pieces"],
                cryptomatte_masks=cryptomatte_masks,
            )
            coco["annotations"].extend(piece_anns)
            ann_id += len(piece_anns)

            n_ann = len(board_anns) + len(piece_anns)
            total_time = time.time() - t0
            print(f"annotated {n_ann} objs  ({total_time:.1f}s total)")

            # --- 9. Cleanup temp EXR files for this shot ---
            shutil.rmtree(shot_crypto_dir, ignore_errors=True)

            # --- 10. Save progress (crash recovery) ---
            completed.add(filename)
            save_progress(completed, coco)

    finally:
        # --- Restore compositor: remove Cryptomatte nodes ---
        teardown_compositor_cryptomatte(scene, crypto_fo_node)

        # --- Cleanup temp directory ---
        shutil.rmtree(CRYPTO_TMP_DIR, ignore_errors=True)

    # --- 11. Save COCO JSON ---
    with open(ANNOTATIONS_PATH, "w") as f:
        json.dump(coco, f, indent=2)

    elapsed = time.time() - pipeline_start
    print(f"\n{'=' * 60}")
    print(f"  PIPELINE COMPLETE  ({elapsed / 60:.1f} min)")
    print(f"  Images      : {len(coco['images'])}")
    print(f"  Annotations : {len(coco['annotations'])}")
    print(f"  Saved to    : {ANNOTATIONS_PATH}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
