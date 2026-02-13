"""
02_camera_rig.py -- Compute and store 25 camera positions.

Defines a spherical camera rig around the chessboard centre:
    Ring 1:  12 positions at elevation 30 deg, azimuth 0/30/60/.../330
    Ring 2:  12 positions at elevation 60 deg, same azimuths
    Top:      1 position  at elevation 83 deg, azimuth 0

Saves to:  chess_dataset/camera_positions.json

Run from Blender:
    blender --background scene.blend --python chess_dataset/scripts/02_camera_rig.py
"""

import json
import math
import os

# ---------------------------------------------------------------------------
#  Config
# ---------------------------------------------------------------------------

BASE_DIR = "<ABSOLUTE_PATH_TO_DATASET>"
BOARD_HEIGHT = 0.0033          # Z of the board surface
CAMERA_RADIUS = 3.0            # metres from board centre


# ---------------------------------------------------------------------------
#  Camera position computation
# ---------------------------------------------------------------------------

def compute_camera_positions():
    """Return a list of 25 camera-position dicts."""
    positions = []

    # Ring 1 -- elevation 30 deg, 12 azimuths
    for i in range(12):
        az_deg = i * 30
        el_deg = 30
        az = math.radians(az_deg)
        el = math.radians(el_deg)
        x = CAMERA_RADIUS * math.cos(el) * math.cos(az)
        y = CAMERA_RADIUS * math.cos(el) * math.sin(az)
        z = CAMERA_RADIUS * math.sin(el) + BOARD_HEIGHT
        positions.append({
            "index": len(positions),
            "ring": 1,
            "azimuth_deg": az_deg,
            "elevation_deg": el_deg,
            "location": [round(x, 4), round(y, 4), round(z, 4)],
            "filename": f"ring1_az{az_deg:03d}_el30.png",
        })

    # Ring 2 -- elevation 60 deg, 12 azimuths
    for i in range(12):
        az_deg = i * 30
        el_deg = 60
        az = math.radians(az_deg)
        el = math.radians(el_deg)
        x = CAMERA_RADIUS * math.cos(el) * math.cos(az)
        y = CAMERA_RADIUS * math.cos(el) * math.sin(az)
        z = CAMERA_RADIUS * math.sin(el) + BOARD_HEIGHT
        positions.append({
            "index": len(positions),
            "ring": 2,
            "azimuth_deg": az_deg,
            "elevation_deg": el_deg,
            "location": [round(x, 4), round(y, 4), round(z, 4)],
            "filename": f"ring2_az{az_deg:03d}_el60.png",
        })

    # Top -- elevation 83 deg
    az_deg = 0
    el_deg = 83
    az = math.radians(az_deg)
    el = math.radians(el_deg)
    x = CAMERA_RADIUS * math.cos(el) * math.cos(az)
    y = CAMERA_RADIUS * math.cos(el) * math.sin(az)
    z = CAMERA_RADIUS * math.sin(el) + BOARD_HEIGHT
    positions.append({
        "index": len(positions),
        "ring": 0,
        "azimuth_deg": az_deg,
        "elevation_deg": el_deg,
        "location": [round(x, 4), round(y, 4), round(z, 4)],
        "filename": "top_az000_el83.png",
    })

    return positions


# ---------------------------------------------------------------------------
#  Main
# ---------------------------------------------------------------------------

def main():
    positions = compute_camera_positions()

    out_path = os.path.join(BASE_DIR, "camera_positions.json")
    with open(out_path, "w") as f:
        json.dump(positions, f, indent=2)

    print(f"Saved {len(positions)} camera positions to {out_path}")
    for p in positions:
        print(f"  [{p['index']:2d}] ring={p['ring']}  az={p['azimuth_deg']:3d} deg  "
              f"el={p['elevation_deg']} deg  "
              f"loc=({p['location'][0]:7.3f}, {p['location'][1]:7.3f}, "
              f"{p['location'][2]:7.3f})  {p['filename']}")

    # ----- Optional: update Blender camera for quick preview -----
    try:
        import bpy
        from mathutils import Vector

        target = Vector((0.0, 0.0, BOARD_HEIGHT))
        cam = bpy.data.objects.get("Camera")
        if cam:
            pos = positions[0]
            cam.location = Vector(pos["location"])
            direction = target - cam.location
            rot_quat = direction.to_track_quat('-Z', 'Y')
            cam.rotation_euler = rot_quat.to_euler()
            print(f"\n  Camera moved to position 0 for preview.")
    except ImportError:
        pass


if __name__ == "__main__":
    main()
