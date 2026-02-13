"""
04_configurations.py -- Hardcoded scene configurations for the chess dataset.

36 configurations across 12 camera azimuths (0, 30, 60, ..., 330 degrees).
Three rounds of 12, each with different variety:

  Round 1 (0-11):  Standard crowded scenes, one per azimuth
  Round 2 (12-23): Mixed -- occlusion, clear, sparse, varied crowds
  Round 3 (24-35): More variety -- different combos, compositions

Configuration styles:
  crowded   -- 8-10+ spectators packed around the board
  sparse    -- 2-4 spectators
  clear     -- no spectators (unoccluded board view)
  occlusion -- reaching player seated close to board, hand over pieces

All models face -Y at rot_z=0 (EULER XYZ mode).
Rotation to face the board from (x, y):  rot_z = atan2(-x, y)
"""

import bpy
import math
import random
import sys

# ---------------------------------------------------------------------------
#  Constants
# ---------------------------------------------------------------------------

BOARD_CENTER = (0.0, 0.0, 0.0033)

WHITE_SEAT = {"x": 0.0, "y": -0.7}
BLACK_SEAT_NORMAL = {"x": 0.0, "y":  0.7}
BLACK_SEAT_CLOSE  = {"x": 0.0, "y":  0.5}   # occlusion configs

# Original Z per character (preserves correct ground/chair height)
ORIGINAL_Z = {
    "Seated_Character_01":          -0.1298,
    "Seated_Character_02":          -0.0994,
    "Seated_Character_03":             -0.1298,
    "Seated_Character_04":           -0.1298,
    "Seated_Character_05":           -0.13,
    "Reaching_Character_01":    -0.0531,
    "Reaching_Character_02":    -0.1315,
    "Reaching_Character_03":   -0.13,
    "Reaching_Character_04": -0.1063,
    "Reaching_Character_05": -0.1315,
    "Spectator_01":                   0.095,
    "Standing_Character_04":     0.0,
    "Standing_Character_05":     0.0,
    "Standing_Character_06":      0.0,
    "Standing_Character_01":            0.1202,
    "Standing_Character_02":          0.0707,
    "Standing_Character_03":          0.0443,
}

ALL_HUMANS = list(ORIGINAL_Z.keys())

WHITE_SIDE_PLAYERS = [
    "Seated_Character_01", "Seated_Character_02", "Seated_Character_03",
    "Seated_Character_04", "Seated_Character_05",
]
BLACK_SIDE_PLAYERS = [
    "Reaching_Character_01", "Reaching_Character_02",
    "Reaching_Character_03", "Reaching_Character_04",
    "Reaching_Character_05",
]
SPECTATORS = [
    "Spectator_01", "Standing_Character_04", "Standing_Character_05",
    "Standing_Character_06", "Standing_Character_01", "Standing_Character_02",
    "Standing_Character_03",
]


def _face_board(x, y):
    """Compute rot_z for a -Y-forward model at (x, y) to face the board."""
    return math.atan2(-x, y)


# ---------------------------------------------------------------------------
#  Position generation
# ---------------------------------------------------------------------------

def _check_clearance(x, y, occupied, min_dist=0.5):
    """Return True if (x, y) is at least *min_dist* from all *occupied*."""
    for ox, oy in occupied:
        if math.sqrt((x - ox) ** 2 + (y - oy) ** 2) < min_dist:
            return False
    return True


def _generate_positions(cam_az_deg, black_seat_y=0.7):
    """
    Generate spectator (x, y) positions for a camera at *cam_az_deg*.

    Places characters in two rings on the visible side of the board
    (the semicircle opposite the camera):
        Close ring:  r = 0.9 m   (right at the table edge)
        Far ring:    r = 1.3 m   (standing behind the close ring)
    """
    cam_az = math.radians(cam_az_deg)

    seats = [(WHITE_SEAT["x"], WHITE_SEAT["y"]),
             (0.0, black_seat_y)]

    positions = []

    # Close ring -- 5 candidates at 35 deg spacing across the visible arc
    close_r = 0.9
    for offset_deg in [105, 140, 175, 210, 245]:
        angle = cam_az + math.radians(offset_deg)
        x = round(close_r * math.cos(angle), 3)
        y = round(close_r * math.sin(angle), 3)
        if _check_clearance(x, y, seats + positions, min_dist=0.45):
            positions.append((x, y))

    # Far ring -- 8 candidates at 27 deg spacing, wider arc
    far_r = 1.3
    for offset_deg in [80, 107, 134, 161, 199, 226, 253, 280]:
        angle = cam_az + math.radians(offset_deg)
        x = round(far_r * math.cos(angle), 3)
        y = round(far_r * math.sin(angle), 3)
        if _check_clearance(x, y, seats + positions, min_dist=0.50):
            positions.append((x, y))

    return positions


# ---------------------------------------------------------------------------
#  Configuration specs
# ---------------------------------------------------------------------------
#
# Each spec: (name, az_deg, white_idx, black_idx, style, max_crowd, seed)
#
#   white_idx  -> index into WHITE_SIDE_PLAYERS
#   black_idx  -> index into BLACK_SIDE_PLAYERS
#                 (1 = Reaching_Character_02, stretched right hand)
#                 (4 = Reaching_Character_05, stretched left hand)
#   style      -> "crowded" | "sparse" | "clear" | "occlusion"
#   max_crowd  -> max spectators to place (0 for clear)
#   seed       -> deterministic shuffle seed for crowd composition

_CONFIG_SPECS = [
    # ==== Round 1: Standard crowded (one per azimuth) ====
    ("r1_az000",     0,   0, 0, "crowded",   15, 100),
    ("r1_az030",     30,  1, 1, "crowded",   15, 101),
    ("r1_az060",     60,  2, 2, "crowded",   15, 102),
    ("r1_az090",     90,  3, 3, "crowded",   15, 103),
    ("r1_az120",     120, 4, 4, "crowded",   15, 104),
    ("r1_az150",     150, 0, 1, "crowded",   15, 105),
    ("r1_az180",     180, 1, 2, "crowded",   15, 106),
    ("r1_az210",     210, 2, 3, "crowded",   15, 107),
    ("r1_az240",     240, 3, 4, "crowded",   15, 108),
    ("r1_az270",     270, 4, 0, "crowded",   15, 109),
    ("r1_az300",     300, 0, 2, "crowded",   15, 110),
    ("r1_az330",     330, 2, 4, "crowded",   15, 111),

    # ==== Round 2: Mixed variety ====
    ("r2_az000_occ", 0,   2, 1, "occlusion", 10, 200),
    ("r2_az030_mix", 30,  4, 3, "crowded",   15, 201),
    ("r2_az060_spr", 60,  0, 0, "sparse",     3, 202),
    ("r2_az090_clr", 90,  1, 4, "clear",      0, 203),
    ("r2_az120_occ", 120, 3, 4, "occlusion",  8, 204),
    ("r2_az150_mix", 150, 2, 0, "crowded",   15, 205),
    ("r2_az180_clr", 180, 4, 2, "clear",      0, 206),
    ("r2_az210_spr", 210, 1, 1, "sparse",     4, 207),
    ("r2_az240_occ", 240, 0, 1, "occlusion", 10, 208),
    ("r2_az270_mix", 270, 3, 3, "crowded",   15, 209),
    ("r2_az300_spr", 300, 4, 0, "sparse",     3, 210),
    ("r2_az330_occ", 330, 1, 4, "occlusion",  8, 211),

    # ==== Round 3: More variety ====
    ("r3_az000_spr", 0,   4, 3, "sparse",     4, 300),
    ("r3_az030_occ", 30,  0, 1, "occlusion", 12, 301),
    ("r3_az060_mix", 60,  3, 4, "crowded",   15, 302),
    ("r3_az090_spr", 90,  2, 0, "sparse",     2, 303),
    ("r3_az120_mix", 120, 0, 2, "crowded",   15, 304),
    ("r3_az150_occ", 150, 3, 4, "occlusion", 10, 305),
    ("r3_az180_spr", 180, 0, 3, "sparse",     2, 306),
    ("r3_az210_occ", 210, 4, 1, "occlusion",  8, 307),
    ("r3_az240_mix", 240, 1, 0, "crowded",   15, 308),
    ("r3_az270_clr", 270, 2, 2, "clear",      0, 309),
    ("r3_az300_occ", 300, 3, 4, "occlusion", 10, 310),
    ("r3_az330_mix", 330, 0, 1, "crowded",   15, 311),
]


# ---------------------------------------------------------------------------
#  Build all 36 configurations (precomputed at import time)
# ---------------------------------------------------------------------------

CONFIGURATIONS = []

for _spec in _CONFIG_SPECS:
    _name, _az_deg, _wi, _bi, _style, _max_crowd, _seed = _spec

    _wp = WHITE_SIDE_PLAYERS[_wi]
    _bp = BLACK_SIDE_PLAYERS[_bi]

    # Black seat position: closer for occlusion configs
    _black_seat_y = BLACK_SEAT_CLOSE["y"] if _style == "occlusion" \
                    else BLACK_SEAT_NORMAL["y"]

    # Build available character pool (everyone not playing)
    # Mix of standing spectators and inactive seated/reaching players
    _available = list(SPECTATORS)
    for _p in WHITE_SIDE_PLAYERS:
        if _p != _wp:
            _available.append(_p)
    for _p in BLACK_SIDE_PLAYERS:
        if _p != _bp:
            _available.append(_p)

    # Shuffle deterministically so each config has a unique crowd mix
    _rng = random.Random(_seed)
    _rng.shuffle(_available)

    # Generate positions and assign characters
    if _style == "clear":
        _spectators = []
    else:
        _slots = _generate_positions(_az_deg, black_seat_y=_black_seat_y)
        _spectators = []
        for _j, (_sx, _sy) in enumerate(_slots):
            if _j >= _max_crowd or _j >= len(_available):
                break
            _spectators.append({"name": _available[_j], "x": _sx, "y": _sy})

    CONFIGURATIONS.append({
        "name": _name,
        "azimuth_deg": _az_deg,
        "style": _style,
        "white_player": _wp,
        "black_player": _bp,
        "black_seat_y": _black_seat_y,
        "spectators": _spectators,
    })


# ---------------------------------------------------------------------------
#  Apply a configuration
# ---------------------------------------------------------------------------

def apply_configuration(config_index):
    """
    Apply configuration *config_index* to the Blender scene.

    1. Hide all human characters and move them off-stage.
    2. Place the white-side player at the white seat, facing the board.
    3. Place the black-side player at the black seat, facing the board.
       (occlusion configs use a closer seat so the hand reaches over pieces)
    4. Place each listed spectator at their position, facing the board.
    5. Update the dependency graph.

    Returns the config dict for reference.
    """
    config = CONFIGURATIONS[config_index % len(CONFIGURATIONS)]

    # 1. Hide everyone and park off-stage
    for name in ALL_HUMANS:
        obj = bpy.data.objects.get(name)
        if obj is None:
            continue
        obj.rotation_mode = 'XYZ'
        obj.hide_viewport = True
        obj.hide_render = True
        obj.location = (50.0, 50.0, 0.0)

    # 2. White-side player
    wp_name = config["white_player"]
    wp = bpy.data.objects.get(wp_name)
    if wp:
        wp.hide_viewport = False
        wp.hide_render = False
        wp.location = (WHITE_SEAT["x"], WHITE_SEAT["y"],
                       ORIGINAL_Z.get(wp_name, -0.13))
        wp.rotation_euler = (0.0, 0.0, _face_board(WHITE_SEAT["x"],
                                                     WHITE_SEAT["y"]))

    # 3. Black-side player (may be closer for occlusion configs)
    bp_name = config["black_player"]
    bp = bpy.data.objects.get(bp_name)
    black_y = config["black_seat_y"]
    if bp:
        bp.hide_viewport = False
        bp.hide_render = False
        bp.location = (0.0, black_y,
                       ORIGINAL_Z.get(bp_name, -0.053))
        bp.rotation_euler = (0.0, 0.0, _face_board(0.0, black_y))

    # 4. Spectators
    for spec in config["spectators"]:
        obj = bpy.data.objects.get(spec["name"])
        if obj is None:
            continue
        obj.hide_viewport = False
        obj.hide_render = False
        sx, sy = spec["x"], spec["y"]
        obj.location = (sx, sy, ORIGINAL_Z.get(spec["name"], 0.0))
        obj.rotation_euler = (0.0, 0.0, _face_board(sx, sy))

    # 5. Refresh
    bpy.context.view_layer.update()

    n_spec = len(config["spectators"])
    visible = 2 + n_spec
    print(f"[config] Applied '{config['name']}' ({config['style']}) -- "
          f"{config['white_player']} vs {config['black_player']}, "
          f"{visible} characters visible")
    return config


# ---------------------------------------------------------------------------
#  CLI: apply a config by index
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    idx = 0
    if "--" in sys.argv:
        args = sys.argv[sys.argv.index("--") + 1:]
        if args:
            idx = int(args[0])
    apply_configuration(idx)
