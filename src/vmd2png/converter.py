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
    frames, dimension = data.shape
    print(f'{data.shape} {frames} {dimension}')
    
    # Pad dimension to multiple of 4 (RGBA)
    padded_dim = (dimension + 3) // 4 * 4
    dim_pad = padded_dim - dimension
    if dim_pad > 0:
        data = np.pad(data, ((0, 0), (0, dim_pad)), mode='constant')
        
    # Frames will be columns, Dimension will be rows (4 vals per pixel)
    rows_per_frame = padded_dim // 4
    
    # Reshape data to (frames, rows_per_frame, 4)
    data_reshaped = data.reshape(frames, rows_per_frame, 4)
    
    max_width = 1024
    
    if frames <= max_width:
        # Layout: Width=frames, Height=rows_per_frame
        # Transpose (frames, rows, 4) -> (rows, frames, 4)
        # Image Height = rows_per_frame, Image Width = frames
        img_data_float = data_reshaped.transpose(1, 0, 2)
    else:
        # Layout: Width=max_width, Height= Multiple blocks of rows_per_frame
        # Pad frames to multiple of max_width
        frame_pad = (max_width - (frames % max_width)) % max_width
        if frame_pad > 0:
            data_reshaped = np.pad(data_reshaped, ((0, frame_pad), (0, 0), (0, 0)), mode='constant')
            
        total_frames = data_reshaped.shape[0]
        num_blocks = total_frames // max_width
        
        # Reshape to (num_blocks, max_width, rows, 4)
        blocks = data_reshaped.reshape(num_blocks, max_width, rows_per_frame, 4)
        
        # Transpose to (num_blocks, rows, max_width, 4) to stack vertically
        blocks = blocks.transpose(0, 2, 1, 3)
        
        # Combine blocks: flatten first two dims
        img_data_float = blocks.reshape(num_blocks * rows_per_frame, max_width, 4)
    
    # Convert to uint16
    img_data_uint16 = float_to_uint16(img_data_float, min_val, max_val)
    
    # OpenCV expects BGRA for 4-channel images, but we want to map logically
    # R=0, G=1, B=2, A=3. If we want raw data preservation order:
    # If we write [0, 1, 2, 3] via imwrite, it writes B=0, G=1, R=2, A=3?
    # No, imwrite expects [B, G, R, A].
    # So if our data is [v1, v2, v3, v4], and we want v1 in R, v2 in G, v3 in B, v4 in A:
    # We should pass [v3, v2, v1, v4] to imwrite.
    # HOWEVER, if we simple want to store data and retrieve it SAME ORDER:
    # We can just ignore the color channel meaning. Write [v1, v2, v3, v4] as BGRA, read as BGRA -> [v1, v2, v3, v4].
    # So we don't need to permute.
    
    import cv2
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    success = cv2.imwrite(output_path, img_data_uint16)
    if not success:
        print(f"Failed to write image to {output_path}")

def load_from_png_16bit(file_path, min_val, max_val, stride=None):
    import cv2
    img = cv2.imread(file_path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise ValueError(f"Failed to load image: {file_path}")
        
    # Expect 4 channels, uint16
    if img.dtype != np.uint16:
        # If it loaded as uint8 (e.g. min/max val small), convert? 
        # But imread(..., UNCHANGED) should preserve depth if PNG is 16bit.
        # If the file wasn't 16bit, this might be an issue.
        # But we assume we wrote it as 16bit.
        if img.dtype == np.uint8:
             img = (img.astype(np.float32) * 257).astype(np.uint16) # Scale up? Or just cast? 
             # Usually 8bit PNGs scale 0-255. 16bit is 0-65535. 
             # 255 * 257 = 65535.
             pass
    
    # If standard BGR loaded (3 channels) or Gray (1 channel), handle?
    # We expect 4 channels for our data format.
    if len(img.shape) == 2: # Grayscale
        # Treat as 1 channel stream?
        img = img.reshape(img.shape[0], img.shape[1], 1)
        
    if stride is None:
        flat_data = img.flatten()
    else:
        H, W = img.shape[:2]
        C = img.shape[2] if len(img.shape) > 2 else 1
        rows_per_frame = (stride + 3) // 4
        num_blocks = H // rows_per_frame
        
        # Reshape to (Blocks, Rows, Width, C)
        # Handle cases where image height might not be perfect multiple if cropped somehow (robustness)
        valid_H = num_blocks * rows_per_frame
        img = img[:valid_H, :, :]
        
        blocks = img.reshape(num_blocks, rows_per_frame, W, C)
        
        # Transpose to (Blocks, Width, Rows, C) -> (TotalFrames, Rows, C)
        blocks = blocks.transpose(0, 2, 1, 3)
        
        # Flatten frames content to 1D stream per frame (TotalFrames, Rows*C)
        frames_data = blocks.reshape(-1, rows_per_frame * C)
        
        # Trim padding in dimension
        if frames_data.shape[1] > stride:
            frames_data = frames_data[:, :stride]
            
        flat_data = frames_data.flatten()
    
    return uint16_to_float(flat_data, min_val, max_val)

def export_vmd_to_files(vmd_path, output_dir):
    results = vmd_to_motion_data(vmd_path, verbose=False)
    if not results:
        print(f"Failed to extract info from {vmd_path}")
        return False
        
    name = os.path.basename(vmd_path).replace('.vmd', '')
    os.makedirs(output_dir, exist_ok=True)
    
    if results['character'] is not None:
        c_data = results['character']
        np.save(os.path.join(output_dir, f"{name}_character.npy"), c_data)
        save_as_png_16bit(c_data, os.path.join(output_dir, f"{name}_character.png"), -1, 1)
        
    if results['camera'] is not None:
        cam_data = results['camera']
        np.save(os.path.join(output_dir, f"{name}_camera.npy"), cam_data)
        save_as_png_16bit(cam_data, os.path.join(output_dir, f"{name}_camera.png"), -1, 1)
        
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
        # Determine stride first to decode PNG layout
        if mode == 'camera':
            stride = 8 # Pos(3) + FOV(1) + Rot(4)
        else:
            root, _ = build_standard_skeleton()
            bones = root.export_bones()
            # Stride = Center(3) + Scale(1) + Bones(N)*4
            stride = 4 + len(bones) * 4
            
        scale = 1.0 #50.0 if mode == 'camera' else 20.0
        flat_data = load_from_png_16bit(input_path, -scale, scale, stride=stride)
        
        # Calculate number of frames
        # Use integer division, ignore trailing padding zeros if any
        num_frames = len(flat_data) // stride
        if num_frames == 0:
            return None
            
        # Truncate to exact multiple
        data = flat_data[:num_frames * stride]
        data = data.reshape(num_frames, stride)
    else:
        print("Unsupported format")
        return None
        
    frames = []
    
    if mode == 'camera':
        for i in range(len(data)):
            row = data[i]
            pos = row[0:3] * 32768 / 1000
            fov = row[3] * 6
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
                        pos = tuple(row[0:3] * 32768 / 1000)
                    
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
        return False, anim
    return write_vmd(output_vmd_path, anim), anim
