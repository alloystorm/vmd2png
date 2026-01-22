import numpy as np
import os
from PIL import Image
from .vmd import vmd_to_motion_data, write_vmd
from .skeleton import build_standard_skeleton

def float_to_uint16(data, min_val, max_val):
    if min_val == max_val:
        return np.zeros_like(data, dtype=np.uint16)
    norm = (data - min_val) / (max_val - min_val)
    norm = np.clip(norm, 0, 1)
    return (norm * 65535).astype(np.uint16)

def uint16_to_float(data, min_val, max_val):
    norm = data.astype(np.float32) / 65535.0
    return norm * (max_val - min_val) + min_val

def save_as_png_16bit(data, output_path, min_val=-20.0, max_val=20.0):
    if data is None or data.size == 0:
        return
    uint16_data = float_to_uint16(data, min_val, max_val)
    img = Image.fromarray(uint16_data, mode='I;16')
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    img.save(output_path)

def load_from_png_16bit(file_path, min_val=-20.0, max_val=20.0):
    img = Image.open(file_path)
    if img.mode != 'I;16':
        raise ValueError(f"Expected I;16 mode PNG, got {img.mode}")
    arr = np.array(img)
    return uint16_to_float(arr, min_val, max_val)

def export_vmd_to_files(vmd_path, output_dir, png_scale_pos=20.0):
    results = vmd_to_motion_data(vmd_path, verbose=False)
    if not results:
        print(f"Failed to extract info from {vmd_path}")
        return False
        
    name = os.path.basename(vmd_path).replace('.vmd', '')
    os.makedirs(output_dir, exist_ok=True)
    
    if results['character'] is not None:
        c_data = results['character']
        np.save(os.path.join(output_dir, f"{name}_character.npy"), c_data)
        save_as_png_16bit(c_data, os.path.join(output_dir, f"{name}_character.png"), -png_scale_pos, png_scale_pos)
        
    if results['camera'] is not None:
        cam_data = results['camera']
        np.save(os.path.join(output_dir, f"{name}_camera.npy"), cam_data)
        save_as_png_16bit(cam_data, os.path.join(output_dir, f"{name}_camera.png"), -50.0, 50.0)
        
    return True

def load_motion_dict(input_path, mode='character'):
    ext = os.path.splitext(input_path)[1].lower()
    if ext == '.vmd':
        success, anim = vmd_to_motion_data(input_path, verbose=False) # Wait, vmd_to_motion_data returns data arrays, not anim dict.
        # We need parse_vmd for anim dict.
        from .vmd import parse_vmd
        success, anim = parse_vmd(input_path) 
        if not success: return None
        return anim
        
    if ext == '.npy':
        data = np.load(input_path)
    elif ext == '.png':
        scale = 50.0 if mode == 'camera' else 20.0
        data = load_from_png_16bit(input_path, -scale, scale)
    else:
        print("Unsupported format")
        return None
        
    frames = []
    
    if mode == 'camera':
        for i in range(len(data)):
            row = data[i]
            pos = row[0:3]
            fov = row[3]
            rot = row[4:8]
            
            frame = {
                "frame_num": i,
                "position": pos,
                "rotation": rot,
                "dist": 0.0, 
                "fov": int(fov),
                "bezier": bytearray([20]*24)
            }
            frames.append(frame)
        anim = {"camera_frames": frames, "bone_frames": [], "morph_frames": [], "unit": 0.085, "duration": len(data)/30.0}
        
    else:
        root, _ = build_standard_skeleton()
        bones = root.export_bones()
        
        bone_frames = []
        for i in range(len(data)):
            row = data[i]
            curr = 4
            for bone in bones:
                if curr + 4 <= len(row):
                    quat = row[curr:curr+4]
                    curr += 4
                    pos = (0,0,0)
                    if bone.name == "Center":
                        pos = tuple(row[0:3])
                    
                    frame = {
                        "name": bone.name,
                        "frame_num": i,
                        "position": pos,
                        "rotation": tuple(quat),
                        "bezier": bytearray([20]*64)
                    }
                    bone_frames.append(frame)
        
        # Convert list of dicts to bones dict of lists
        bones_dict = {}
        for f in bone_frames:
            if f["name"] not in bones_dict:
                bones_dict[f["name"]] = []
            bones_dict[f["name"]].append(f)
            
        anim = {
            "bone_frames": bone_frames, 
            "bones": bones_dict,
            "morph_frames": [], 
            "camera_frames": [], 
            "unit": 0.085,
            "duration": len(data)/30.0
        }

    return anim

def convert_motion_to_vmd(input_path, output_vmd_path, mode='character'):
    anim = load_motion_dict(input_path, mode)
    if not anim:
        return False
    return write_vmd(output_vmd_path, anim)
