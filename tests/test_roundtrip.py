import sys
import os
import numpy as np
import shutil

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))

from vmd2png.vmd import write_vmd, parse_vmd
from vmd2png.converter import export_vmd_to_files, convert_motion_to_vmd, load_motion_dict

def test_pipeline():
    print("Testing Pipeline...")
    output_dir = "test_output"
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir)
    
    # 1. Create Dummy Animation
    print("1. Creating dummy VMD data...")
    bone_frames = []
    for i in range(10):
        frame = {
            "name": "Center",
            "frame_num": i,
            "position": (0, i * 0.1, 0), # Rising up
            "rotation": (0, 0, 0, 1),
            "bezier": bytearray([20]*64)
        }
        bone_frames.append(frame)
        
    camera_frames = []
    for i in range(10):
        frame = {
            "frame_num": i,
            "dist": 10.0,
            "position": (0, i * 0.1, 0),
            "rotation": (0, 0, 0, 1),
            "fov": 30,
            "bezier": bytearray([20]*24)
        }
        camera_frames.append(frame)
        
    anim = {
        "bone_frames": bone_frames,
        "camera_frames": camera_frames,
        "morph_frames": [],
        "unit": 0.085
    }
    
    vmd_path = os.path.join(output_dir, "test.vmd")
    success = write_vmd(vmd_path, anim)
    assert success, "Failed to write VMD"
    print("   VMD writen successfully.")
    
    # 2. Parse VMD
    print("2. Parsing VMD...")
    success, parsed_anim = parse_vmd(vmd_path)
    assert success, "Failed to parse VMD"
    assert len(parsed_anim["bone_frames"]) == 10
    assert len(parsed_anim["camera_frames"]) == 10
    print("   VMD parsed successfully.")
    
    # 3. Export to Files
    print("3. Exporting to PNG/NPY...")
    files_dir = os.path.join(output_dir, "exported")
    success = export_vmd_to_files(vmd_path, files_dir)
    assert success, "Failed to export files"
    
    exp_char_png = os.path.join(files_dir, "test_character.png")
    exp_cam_png = os.path.join(files_dir, "test_camera.png")
    assert os.path.exists(exp_char_png), "Character PNG missing"
    assert os.path.exists(exp_cam_png), "Camera PNG missing"
    print("   Export successful.")
    
    # 4. Load from PNG and Convert back
    print("4. Importing from PNG and converting back to VMD...")
    back_vmd_char = os.path.join(output_dir, "back_char.vmd")
    back_vmd_cam = os.path.join(output_dir, "back_cam.vmd")
    
    success = convert_motion_to_vmd(exp_char_png, back_vmd_char, mode='character')
    assert success, "Failed to convert character PNG to VMD"
    
    success = convert_motion_to_vmd(exp_cam_png, back_vmd_cam, mode='camera')
    assert success, "Failed to convert camera PNG to VMD"
    print("   Conversion successful.")
    
    # 5. Verify Content of converted VMD
    print("5. Verifying converted VMD...")
    success, char_anim = parse_vmd(back_vmd_char)
    # The converted VMD will have frames for ALL bones (baked), not just the original ones.
    # So count will be frames * num_bones.
    assert len(char_anim["bone_frames"]) >= 10
    print(f"   Converted VMD has {len(char_anim['bone_frames'])} frames (expected >= 10).")
    
    # Check "Center" position in last frame (approx)
    # Original: (0, 0.9, 0) for frame 9
    
    # Find frame 9 for Center
    f9 = next((f for f in char_anim["bone_frames"] if f["frame_num"] == 9 and f["name"] == "Center"), None)
    assert f9 is not None, "Center frame 9 not found"
    # Note: vmd.py parse_vmd applies 'unit' scaling when reading?
    # parse_vmd: position = (px * unit, ...)
    # write_vmd: px / unit
    # So the values in 'f9' are scaled by 0.085 if we use default unit in parse_vmd.
    # Wait, 'write_vmd' divides by 'unit'. 'parse_vmd' multiplies by unit.
    # So roundtrip should preserve the value passed to write.
    # But converter uses 'vmd_to_motion_data' which deals in "Meters" (scaled).
    # 'vmd_to_motion_data' calls 'parse_vmd(unit=0.085)'.
    # So 'character_data' NPY contains meters.
    # Then we save to PNG (mapped -20 to 20 meters).
    # Then we load from PNG (meters).
    # Then 'convert_motion_to_vmd' calls 'write_vmd'.
    # 'write_vmd' divides by 'unit' (0.085) to store MMD units.
    # 'parse_vmd' reads MMD units and multiplies by 0.085 to get meters.
    # So `f9['position']` should be in Meters.
    # Original pos (0, 0.9, 0).
    print(f"   Original Y: 0.9. Recovered Y: {f9['position'][1]}")
    assert abs(f9['position'][1] - 0.9) < 0.01, f"Position mismatch: {f9['position'][1]}"
    
    print("All tests passed!")

if __name__ == "__main__":
    test_pipeline()
