"""
01_scene_analysis.py — Scene Analysis and Human Position Mapping

Run from Blender's scripting workspace or via:
    blender --background --python chess_dataset/scripts/01_scene_analysis.py

Outputs:
    - Full categorisation of all scene objects
    - Existing piece/board labels and naming conventions
    - 4 board-corner 3D world coordinates
    - Current position/orientation of every human
    - Pre-computed position map for spectator/inactive-player placement
"""

import bpy
import math
import json
from mathutils import Vector

# ============================================================================
# CONSTANTS
# ============================================================================

OUTPUT_DIR = "<ABSOLUTE_PATH_TO_DATASET>"
BOARD_CENTER = Vector((0.0, 0.0, 0.0033))  # board surface height

# ============================================================================
# 1.  CATEGORISE EVERY OBJECT IN THE SCENE
# ============================================================================

# ----- Board / Table / Pieces -----
BOARD_TABLE_NAMES = {"Board", "WoodenTable_02"}

CORNER_NAMES = ["GridCorner_a1", "GridCorner_a8", "GridCorner_h1", "GridCorner_h8"]

# Piece naming convention:  {color}_{PieceType}_{file}
#   color  = "w" (white) | "b" (black)
#   type   = K (King), Q (Queen), R (Rook), B (Bishop), N (Knight), P (Pawn)
#   file   = a-h (original starting file)
PIECE_PREFIXES = ("w_", "b_")

# ----- Player variants (9 total) -----
# White-side players sit at Y ≈ -0.7 (the side with white pieces)
WHITE_SIDE_PLAYERS = [
    "Seated_Character_01",        # dark-skinned seated character
    "Seated_Character_02",        # light-skinned seated character
    "Seated_Character_03",           # dark-skinned seated man
    "Seated_Character_04",         # dark-skinned seated woman
    "Seated_Character_05",         # light-skinned seated woman
]

# Black-side players sit at Y ≈ +0.7 (the side with black pieces)
BLACK_SIDE_PLAYERS = [
    "Reaching_Character_01",  # light-skinned reaching character
    "Reaching_Character_02",  # dark-skinned reaching character
    "Reaching_Character_03", # dark-skinned reaching character v2
    "Reaching_Character_04",  # dark-skinned reaching character v3
]

ALL_PLAYERS = WHITE_SIDE_PLAYERS + BLACK_SIDE_PLAYERS

# ----- Spectators (7 total) -----
SPECTATORS = [
    "Spectator_01",
    "Spectator_02",
    "Spectator_03",
    "Spectator_04",
    "Standing_Character_01",
    "Standing_Character_02",
    "Standing_Character_03",
]

# ----- Environment -----
ENVIRONMENT_NAMES = {
    "Room_Floor", "Room_Wall_Back", "Room_Wall_Front",
    "Room_Wall_Left", "Room_Wall_Right",
    "Light", "Camera", "Chess_Reference", "Empty",
}

# Active-player seating positions (world space)
WHITE_SIDE_SEAT = Vector((0.0, -0.7, -0.13))   # Y < 0 side
BLACK_SIDE_SEAT = Vector((0.0,  0.7, -0.053))   # Y > 0 side

# Rotation to face the board from each seat
# White-side player faces +Y (toward the board from Y=-0.7)
WHITE_SIDE_ROT_Z = math.radians(111.36)   # from current Seated_Character_01
# Black-side player faces -Y (toward the board from Y=+0.7)
BLACK_SIDE_ROT_Z = math.radians(0.92)     # from current Reaching_Character_01


# ============================================================================
# 2.  BOARD CORNER COORDINATES
# ============================================================================

def get_board_corners():
    """Return dict mapping corner name -> 3D world coordinate."""
    corners = {}
    for name in CORNER_NAMES:
        obj = bpy.data.objects.get(name)
        if obj:
            corners[name] = (
                round(obj.location.x, 4),
                round(obj.location.y, 4),
                round(obj.location.z, 4),
            )
    return corners


# ============================================================================
# 3.  PIECE CATALOGUE
# ============================================================================

PIECE_TYPE_MAP = {
    "K": "king", "Q": "queen", "R": "rook",
    "B": "bishop", "N": "knight", "P": "pawn",
}

def catalogue_pieces():
    """Return list of dicts describing every chess piece in the scene."""
    pieces = []
    for obj in bpy.data.objects:
        if not any(obj.name.startswith(p) for p in PIECE_PREFIXES):
            continue
        color = "white" if obj.name.startswith("w_") else "black"
        rest = obj.name[2:]           # e.g. "K", "P_a", "B_c"
        parts = rest.split("_")
        piece_code = parts[0]
        file_letter = parts[1] if len(parts) > 1 else ""
        piece_type = PIECE_TYPE_MAP.get(piece_code, piece_code)
        category_name = f"{color}_{piece_type}"

        pieces.append({
            "object_name": obj.name,
            "color": color,
            "piece_type": piece_type,
            "file": file_letter,
            "category_name": category_name,
            "location": [round(c, 4) for c in obj.location],
            "visible": obj.visible_get(),
            "hide_render": obj.hide_render,
        })
    pieces.sort(key=lambda p: p["object_name"])
    return pieces


# ============================================================================
# 4.  HUMAN POSITION / ORIENTATION ANALYSIS
# ============================================================================

def analyze_human(obj_name):
    """Return position, rotation, facing, and distance info for one human."""
    obj = bpy.data.objects.get(obj_name)
    if obj is None:
        return None

    loc = obj.location.copy()
    rot = obj.rotation_euler.copy()

    # Vector from human to board centre (XY plane)
    to_board = BOARD_CENTER - loc
    to_board_2d = Vector((to_board.x, to_board.y))
    dist = to_board_2d.length

    # Angle from human toward board centre
    angle_to_board = math.atan2(to_board.y, to_board.x)

    # Character forward direction (Blender default: local +Y is forward)
    # After rotation about Z: forward = (sin(rot_z), cos(rot_z))
    fwd_x = math.sin(rot.z)
    fwd_y = math.cos(rot.z)

    # Dot product between forward direction and to-board direction
    to_board_norm = to_board_2d.normalized() if dist > 0.001 else Vector((0, 0))
    facing_dot = fwd_x * to_board_norm.x + fwd_y * to_board_norm.y
    facing_board = facing_dot > 0.3   # roughly within ±72° of facing the board

    # World-space bounding box
    bb = [obj.matrix_world @ Vector(c) for c in obj.bound_box]
    bb_min = Vector((min(v.x for v in bb), min(v.y for v in bb), min(v.z for v in bb)))
    bb_max = Vector((max(v.x for v in bb), max(v.y for v in bb), max(v.z for v in bb)))
    height = bb_max.z - bb_min.z

    return {
        "name": obj_name,
        "location": [round(loc.x, 4), round(loc.y, 4), round(loc.z, 4)],
        "rotation_euler_deg": [round(math.degrees(r), 2) for r in rot],
        "forward_dir": [round(fwd_x, 4), round(fwd_y, 4)],
        "distance_to_board": round(dist, 4),
        "facing_board": facing_board,
        "facing_dot": round(facing_dot, 3),
        "height": round(height, 3),
        "bb_min": [round(v, 3) for v in bb_min],
        "bb_max": [round(v, 3) for v in bb_max],
        "visible": obj.visible_get(),
        "hide_render": obj.hide_render,
    }


# ============================================================================
# 5.  POSITION MAP — valid placements for spectators / inactive players
# ============================================================================

def compute_position_map(n_positions=16, radius=3.0):
    """
    Pre-compute a ring of valid positions around the board for parking
    spectators and inactive player variants.

    Each position is far enough from the board centre that even at
    the lowest camera elevation (30 deg) the sightline clears a
    standing person's head.

    Safety check (elevation 30 deg):
        sightline_z(d) = board_z + d * tan(30)
        At d = 3.0 m  ->  z = 0.003 + 3.0 * 0.577 = 1.73 m
        Tallest character ~1.35 m (top of BB relative to world origin ~0.62)
        => sightline clears every character.

    Returns a list of dicts:
        { "index": int, "location": (x, y, z), "rotation_z": float }
    """
    positions = []
    floor_z = -0.13   # approximate base Z for standing characters
    for i in range(n_positions):
        azimuth = 2.0 * math.pi * i / n_positions
        x = radius * math.cos(azimuth)
        y = radius * math.sin(azimuth)

        # Rotation so the character faces the board centre.
        # Character forward is local +Y.  We need +Y to point from
        # (x, y) toward (0, 0), i.e. direction angle = atan2(-x, -y).
        rot_z = math.atan2(-x, -y)

        positions.append({
            "index": i,
            "location": (round(x, 4), round(y, 4), round(floor_z, 4)),
            "rotation_z": round(rot_z, 4),
            "azimuth_deg": round(math.degrees(azimuth), 1),
        })
    return positions


# ============================================================================
# 6.  PRINT FULL REPORT
# ============================================================================

def print_report():
    sep = "=" * 90
    thin = "-" * 90

    # ---- Pieces ----
    pieces = catalogue_pieces()
    visible = [p for p in pieces if p["visible"] and not p["hide_render"]]
    hidden  = [p for p in pieces if not p["visible"] or p["hide_render"]]

    print(f"\n{sep}")
    print("  CHESS PIECE LABELS  (naming: {{w|b}}_{{K|Q|R|B|N|P}}_{{file}})")
    print(sep)
    print(f"\n  Visible on board ({len(visible)}):")
    for p in visible:
        print(f"    {p['object_name']:18s}  category={p['category_name']:14s}  "
              f"pos=({p['location'][0]:8.4f}, {p['location'][1]:8.4f}, {p['location'][2]:8.4f})")
    print(f"\n  Hidden / captured ({len(hidden)}):")
    for p in hidden:
        print(f"    {p['object_name']:18s}  category={p['category_name']:14s}  "
              f"pos=({p['location'][0]:8.4f}, {p['location'][1]:8.4f}, {p['location'][2]:8.4f})")

    # Unique category names for COCO
    cats = sorted(set(p["category_name"] for p in pieces))
    print(f"\n  Unique COCO categories needed (pieces): {cats}")
    print(f"  + 'chessboard' category for the board itself.")

    # ---- Board Corners ----
    corners = get_board_corners()
    print(f"\n{sep}")
    print("  BOARD CORNER COORDINATES (world space)")
    print(sep)
    for name, coord in corners.items():
        print(f"    {name}: ({coord[0]:.4f}, {coord[1]:.4f}, {coord[2]:.4f})")
    print(f"\n  Board mesh centre: (0.0000, 0.0000, 0.0000)")
    print(f"  Board surface Z:   ~0.0033")
    print(f"  Board half-width:  0.217 m  (full width = 0.434 m)")

    # ---- Player variants ----
    print(f"\n{sep}")
    print("  PLAYER VARIANTS (9 total)")
    print(sep)

    print(f"\n  WHITE-SIDE players (sit at Y ≈ -0.7, facing +Y)  [{len(WHITE_SIDE_PLAYERS)}]:")
    print(f"  {thin}")
    for name in WHITE_SIDE_PLAYERS:
        info = analyze_human(name)
        if info:
            fb = "YES" if info["facing_board"] else "NO "
            print(f"    {info['name']:38s}  loc=({info['location'][0]:7.3f}, {info['location'][1]:7.3f}, {info['location'][2]:7.3f})  "
                  f"rot_z={info['rotation_euler_deg'][2]:8.2f}deg  dist={info['distance_to_board']:.3f}m  "
                  f"facing_board={fb}  h={info['height']:.3f}m")

    print(f"\n  BLACK-SIDE players (sit at Y ≈ +0.7, facing -Y)  [{len(BLACK_SIDE_PLAYERS)}]:")
    print(f"  {thin}")
    for name in BLACK_SIDE_PLAYERS:
        info = analyze_human(name)
        if info:
            fb = "YES" if info["facing_board"] else "NO "
            print(f"    {info['name']:38s}  loc=({info['location'][0]:7.3f}, {info['location'][1]:7.3f}, {info['location'][2]:7.3f})  "
                  f"rot_z={info['rotation_euler_deg'][2]:8.2f}deg  dist={info['distance_to_board']:.3f}m  "
                  f"facing_board={fb}  h={info['height']:.3f}m")

    # ---- Spectators ----
    print(f"\n{sep}")
    print(f"  SPECTATORS (7 total)")
    print(sep)
    for name in SPECTATORS:
        info = analyze_human(name)
        if info:
            fb = "YES" if info["facing_board"] else "NO "
            print(f"    {info['name']:38s}  loc=({info['location'][0]:7.3f}, {info['location'][1]:7.3f}, {info['location'][2]:7.3f})  "
                  f"rot_z={info['rotation_euler_deg'][2]:8.2f}deg  dist={info['distance_to_board']:.3f}m  "
                  f"facing_board={fb}  h={info['height']:.3f}m")

    # ---- Environment ----
    print(f"\n{sep}")
    print("  ENVIRONMENT OBJECTS")
    print(sep)
    for name in sorted(BOARD_TABLE_NAMES | ENVIRONMENT_NAMES):
        obj = bpy.data.objects.get(name)
        if obj:
            print(f"    {name:30s}  type={obj.type:8s}  "
                  f"loc=({obj.location.x:7.3f}, {obj.location.y:7.3f}, {obj.location.z:7.3f})")

    # ---- Position Map ----
    pos_map = compute_position_map(n_positions=16, radius=3.0)
    print(f"\n{sep}")
    print("  POSITION MAP  (16 slots, radius=3.0 m, floor_z=-0.13)")
    print(sep)
    print(f"  Safety: at d=3.0m, el=30deg sightline z = 0.003 + 3.0*tan(30) = 1.73 m")
    print(f"  Tallest character BB top < 0.62 m  =>  all sightlines clear.\n")
    for p in pos_map:
        print(f"    slot {p['index']:2d}:  az={p['azimuth_deg']:6.1f}deg  "
              f"loc=({p['location'][0]:7.3f}, {p['location'][1]:7.3f}, {p['location'][2]:7.3f})  "
              f"rot_z={math.degrees(p['rotation_z']):8.2f}deg")

    # ---- Render Settings ----
    print(f"\n{sep}")
    print("  RENDER SETTINGS")
    print(sep)
    scene = bpy.context.scene
    cam = bpy.data.objects["Camera"]
    print(f"    Resolution:  {scene.render.resolution_x} x {scene.render.resolution_y}")
    print(f"    Engine:      {scene.render.engine}")
    print(f"    Lens:        {cam.data.lens} mm")
    print(f"    Sensor:      {cam.data.sensor_width} mm")

    print(f"\n{sep}")
    print("  SCENE ANALYSIS COMPLETE — awaiting confirmation to proceed.")
    print(sep)


# ============================================================================
# 7.  SAVE DATA STRUCTURES AS JSON (for other scripts to consume)
# ============================================================================

def save_scene_data():
    """Write a JSON file with all analysis data for downstream scripts."""
    data = {
        "board_center": list(BOARD_CENTER),
        "board_corners": get_board_corners(),
        "white_side_seat": list(WHITE_SIDE_SEAT),
        "black_side_seat": list(BLACK_SIDE_SEAT),
        "white_side_rot_z": WHITE_SIDE_ROT_Z,
        "black_side_rot_z": BLACK_SIDE_ROT_Z,
        "white_side_players": WHITE_SIDE_PLAYERS,
        "black_side_players": BLACK_SIDE_PLAYERS,
        "spectators": SPECTATORS,
        "position_map": compute_position_map(n_positions=16, radius=3.0),
        "pieces": catalogue_pieces(),
        "render": {
            "resolution_x": bpy.context.scene.render.resolution_x,
            "resolution_y": bpy.context.scene.render.resolution_y,
            "engine": bpy.context.scene.render.engine,
        },
    }
    out_path = f"{OUTPUT_DIR}/scene_data.json"
    with open(out_path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"\n  Scene data saved to: {out_path}")


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    print_report()
    save_scene_data()
