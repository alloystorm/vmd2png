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

def save_as_png_16bit(data, output_path, min_val=-1, max_val=1):
    if data is None or data.size == 0:
        return
    frames, dimension = data.shape
    
    # Pad dimension to multiple of 4 (RGBA)
    padded_dim = (dimension + 3) // 4 * 4
    dim_pad = padded_dim - dimension
    if dim_pad > 0:
        data = np.pad(data, ((0, 0), (0, dim_pad)), mode='constant')
        
    # Frames will be columns, Dimension will be rows (4 vals per pixel)
    rows_per_frame = padded_dim // 4
    
    # Reshape data to (frames, rows_per_frame, 4)
    data_reshaped = data.reshape(frames, rows_per_frame, 4)
    
    max_width = 1800 
    
    if frames <= max_width:
        img_data_float = data_reshaped.transpose(1, 0, 2)
    else:
        frame_pad = (max_width - (frames % max_width)) % max_width
        if frame_pad > 0:
            data_reshaped = np.pad(data_reshaped, ((0, frame_pad), (0, 0), (0, 0)), mode='constant')
            
        total_frames = data_reshaped.shape[0]
        num_blocks = total_frames // max_width
        
        blocks = data_reshaped.reshape(num_blocks, max_width, rows_per_frame, 4)
        blocks = blocks.transpose(0, 2, 1, 3)
        img_data_float = blocks.reshape(num_blocks * rows_per_frame, max_width, 4)
    
    img_data_uint16 = float_to_uint16(img_data_float, min_val, max_val)
    
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
        
    if img.dtype != np.uint16:
        if img.dtype == np.uint8:
             img = (img.astype(np.float32) * 257).astype(np.uint16)
    
    if len(img.shape) == 2: 
        img = img.reshape(img.shape[0], img.shape[1], 1)
        
    if stride is None:
        flat_data = img.flatten()
    else:
        H, W = img.shape[:2]
        C = img.shape[2] if len(img.shape) > 2 else 1
        rows_per_frame = (stride + 3) // 4
        num_blocks = H // rows_per_frame
        
        valid_H = num_blocks * rows_per_frame
        img = img[:valid_H, :, :]
        
        blocks = img.reshape(num_blocks, rows_per_frame, W, C)
        blocks = blocks.transpose(0, 2, 1, 3)
        frames_data = blocks.reshape(-1, rows_per_frame * C)
        
        if frames_data.shape[1] > stride:
            frames_data = frames_data[:, :stride]
            
        flat_data = frames_data.flatten()
    
    return uint16_to_float(flat_data, min_val, max_val)

def export_vmd_to_files(vmd_path, output_path, out_type='png', leg_ik=False, camera_vmd_path=None):
    if out_type == 'vmd':
        anim = load_motion_dict(vmd_path, leg_ik=leg_ik, camera_vmd_path=camera_vmd_path)
        if not anim: return False
        return write_vmd(output_path, anim)

    results = vmd_to_motion_data(vmd_path, camera_vmd_path=camera_vmd_path, verbose=False, leg_ik=leg_ik)
    if results is None:
        print(f"Failed to extract info from {vmd_path}")
        return False
        
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    
    if out_type == 'npy':
        np.save(output_path, results)
    else:
        save_as_png_16bit(results, output_path, -1, 1)
    
    return True

def load_motion_dict(input_path, leg_ik=False, camera_vmd_path=None):
    ext = os.path.splitext(input_path)[1].lower()
    
    if ext == '.vmd':
        from .vmd import parse_vmd, merge_camera_motion
        success, anim = parse_vmd(input_path, unit=0.085) 
        if not success: return None
        
        if camera_vmd_path:
            anim = merge_camera_motion(anim, camera_vmd_path)
            
        return anim

    from .skeleton import build_standard_skeleton
    root_skel, _ = build_standard_skeleton()
    actor_bones_list = root_skel.export_bones()
    stride_actor = 4 + len(actor_bones_list) * 4
    stride_cam = 8
    
    stride = stride_actor + stride_cam

    data = None
    if ext == '.npy':
        data = np.load(input_path)
    elif ext == '.png':
        scale = 1.0
        flat_data = load_from_png_16bit(input_path, -scale, scale, stride=stride)
        
        num_frames = len(flat_data) // stride
        if num_frames == 0:
            return None
            
        data = flat_data[:num_frames * stride]
        data = data.reshape(num_frames, stride)
    else:
        print("Unsupported format")
        return None
        
    bone_frames = []
    camera_frames = []
    
    
    cam_data = data[:, stride_actor:]
    for i in range(len(cam_data)):
        row = cam_data[i]
        
        pos = row[0:3] * 32768 / 1000
        fov = row[3] * 180
        rot = row[4:8]
        
        if fov < 0.01: break
        
        camera_frames.append({
            "frame_num": i,
            "position": tuple(pos),
            "rotation": rot, 
            "dist": 0.0, 
            "fov": int(fov),
            "bezier": bytearray([20]*24),
            "is_perspective": 0
        })
            
    actor_data = data[:, :stride_actor]
    
    for i in range(len(actor_data)):
        row = actor_data[i]
        if row[3] < 0.01: break
        
        center_pos = row[0:3]
        
        for j, bone in enumerate(actor_bones_list):
            idx = 4 + j * 4
            if idx+4 > len(row): break
            quat = row[idx:idx+4]
            
            if np.all(quat == 0): quat = np.array([0,0,0,1])

            pos = (0,0,0)
            if bone.name == "Center":
                pos = tuple(center_pos * 32768 / 1000)
            
            frame = {
                "name": bone.name,
                "frame_num": i,
                "position": pos,
                "rotation": tuple(quat),
                "bezier": bytearray([20]*64)
            }
            bone_frames.append(frame)
        
    bones_dict = {}
    for f in bone_frames:
        if f["name"] not in bones_dict:
            bones_dict[f["name"]] = []
        bones_dict[f["name"]].append(f)
            
    anim = {
        "bone_frames": bone_frames, 
        "bones": bones_dict,
        "morph_frames": [], 
        "camera_frames": camera_frames, 
        "unit": 0.085,
        "duration": len(data)/30.0
    }

    return anim

def convert_motion_to_vmd(input_path, output_vmd_path):
    anim = load_motion_dict(input_path)
    if not anim:
        return False, anim
    return write_vmd(output_vmd_path, anim), anim