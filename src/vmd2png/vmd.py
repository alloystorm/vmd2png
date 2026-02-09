import math
import os
import struct
import numpy as np
from collections import defaultdict
from scipy.spatial.transform import Rotation as R, Slerp
import time
from .ik import solve_two_bone_ik
from .skeleton import build_standard_skeleton, verify_global_positions

# Japanese to English translations
bone_name_translation = {
    "全ての親": "Master",
    "センター": "Center",
    "グルーブ": "Groove",
    "腰": "Waist",
    "下半身": "HipMaster",
    "上半身": "Torso",
    "上半身2": "Torso2",
    "胸": "Chest",
    "首": "Neck",
    "頭": "Head",
    "左肩": "LeftShoulder",
    "左腕": "LeftArm",
    "左ひじ": "LeftElbow",
    "左手首": "LeftWrist",
    "右肩": "RightShoulder",
    "右腕": "RightArm",
    "右ひじ": "RightElbow",
    "右手首": "RightWrist",
    "左足": "LeftLeg",
    "左ひざ": "LeftKnee",
    "左足首": "LeftAnkle",
    "右足": "RightLeg",
    "右ひざ": "RightKnee",
    "右足首": "RightAnkle",
    "左つま先": "LeftToe",
    "右つま先": "RightToe",
    "左親指１": "LeftThumb0",
    "左親指２": "LeftThumb1",
    "左親指３": "LeftThumb2",
    "右親指１": "RightThumb0",
    "右親指２": "RightThumb1",
    "右親指３": "RightThumb2",
    "左人指１": "LeftIndexFinger1",
    "左人指２": "LeftIndexFinger2",
    "左人指３": "LeftIndexFinger3",
    "右人指１": "RightIndexFinger1",
    "右人指２": "RightIndexFinger2",
    "右人指３": "RightIndexFinger3",
    "左中指１": "LeftMiddleFinger1",
    "左中指２": "LeftMiddleFinger2",
    "左中指３": "LeftMiddleFinger3",
    "右中指１": "RightMiddleFinger1",
    "右中指２": "RightMiddleFinger2",
    "右中指３": "RightMiddleFinger3",
    "左薬指１": "LeftRingFinger1",
    "左薬指２": "LeftRingFinger2",
    "左薬指３": "LeftRingFinger3",
    "右薬指１": "RightRingFinger1",
    "右薬指２": "RightRingFinger2",
    "右薬指３": "RightRingFinger3",
    "左小指１": "LeftPinky1",
    "左小指２": "LeftPinky2",
    "左小指３": "LeftPinky3",
    "右小指１": "RightPinky1",
    "右小指２": "RightPinky2",
    "右小指３": "RightPinky3",
}

# Reverse translation (English to Japanese) for writing VMD
bone_name_reverse_translation = {v: k for k, v in bone_name_translation.items()}

def parse_vmd(file_path, unit=0.085, fps=30.0):
    """
    Parse a VMD file and return an animation dictionary.
    """
    try:
        with open(file_path, "rb") as f:
            f.read(30)
            f.read(20)
            num_frames = struct.unpack("<I", f.read(4))[0]
            max_frame = 0
            bone_frames = []
            bones = defaultdict(list)
            bone_frame_size = 111
            
            last_quats = {}
            
            for _ in range(num_frames):
                data = f.read(bone_frame_size)
                unpacked = struct.unpack("<15s I 3f 4f 64s", data)
                name_bytes, frame_num, px, py, pz, rx, ry, rz, rw, bezier = unpacked
                if frame_num > max_frame:
                    max_frame = frame_num
                
                if rx == 0 and ry == 0 and rz == 0 and rw == 0:
                    rw = 1.0
                
                name = name_bytes.decode('shift_jis', errors='ignore').rstrip('\x00')
                name = bone_name_translation.get(name, name)
                
                current_quat = np.array([rx, ry, rz, rw])
                quat_norm = np.linalg.norm(current_quat)
                if quat_norm > 1e-10:
                    current_quat = current_quat / quat_norm
                
                if name in last_quats:
                    last_quat = last_quats[name]
                    dot_product = np.dot(last_quat, current_quat)
                    if dot_product < 0:
                        current_quat = -current_quat
                
                last_quats[name] = current_quat
                rx, ry, rz, rw = current_quat
                
                frame_data = {
                    "frame_num": frame_num,
                    "position": (px * unit, py * unit, pz * unit),
                    "rotation": (rx, ry, rz, rw),
                    "bezier": bezier,
                }
                
                bones[name].append(frame_data)
                bone_frames.append({
                    "name": name,
                    "frame_num": frame_num,
                    "position": (px * unit, py * unit, pz * unit),
                    "rotation": (rx, ry, rz, rw),
                    "bezier": bezier,
                })

            num_morph_frames = struct.unpack("<I", f.read(4))[0]
            morph_frames = []
            morphs = defaultdict(list)
            morph_frame_size = 23
            for _ in range(num_morph_frames):
                data = f.read(morph_frame_size)
                unpacked = struct.unpack("<15s I f", data)
                name_bytes, frame_num, value = unpacked
                if frame_num > max_frame:
                    max_frame = frame_num
                name = name_bytes.decode('shift_jis', errors='ignore').rstrip('\x00')
                frame_data = {"frame_num": frame_num, "value": value}
                morphs[name].append(frame_data)
                morph_frames.append({
                    "name": name,
                    "frame_num": frame_num,
                    "value": value,
                })

            num_camera_frames = struct.unpack("<I", f.read(4))[0]
            camera_frames = []
            cam_frame_size = 61
            for _ in range(num_camera_frames):
                data = f.read(cam_frame_size)
                unpacked = struct.unpack("<I f 3f 3f 24s I B", data)
                frame_num, dist, px, py, pz, rx, ry, rz, bezier, fov, is_perspective = unpacked
                if frame_num > max_frame:
                    max_frame = frame_num
                rx_deg = -math.degrees(rx)
                ry_deg = -math.degrees(ry)
                rz_deg = math.degrees(rz)
                rot = R.from_euler('xyz', [rx_deg, ry_deg, rz_deg], degrees=True)
                camera_frames.append({
                    "frame_num": frame_num,
                    "position": (px * unit, py * unit, pz * unit),
                    "rotation": rot.as_quat(),
                    "dist": dist,
                    "fov": fov,
                    "bezier": bezier,
                    "is_perspective": bool(is_perspective),
                })

            duration = max_frame / fps
            animation = {
                "unit": unit,
                "duration": duration,
                "bone_frames": bone_frames,
                "morph_frames": morph_frames,
                "bones": dict(bones),
                "morphs": dict(morphs),
                "camera_frames": camera_frames,
            }
            return True, animation
    except Exception as e:
        print(f"Error parsing {file_path}: {e}")
        return False, {"unit": unit, "duration": 0}

def write_vmd(file_path, animation_dict, model_name="MotionOutput"):
    """
    Write animation data to a VMD file.
    
    Args:
        file_path: Output file path
        animation_dict: Dictionary containing 'bone_frames', 'morph_frames', 'camera_frames'
        model_name: Name of the model
    """
    try:
        with open(file_path, "wb") as f:
            # Header
            header = b"Vocaloid Motion Data 0002\x00\x00\x00\x00\x00"
            f.write(header)
            
            # Model Name (20 bytes, Shift-JIS)
            try:
                model_name_bytes = model_name.encode('shift_jis')
            except UnicodeEncodeError:
                model_name_bytes = model_name.encode('ascii', 'ignore')
            f.write(model_name_bytes[:20].ljust(20, b'\x00'))
            
            # Bone Frames
            bone_frames = animation_dict.get('bone_frames', [])
            f.write(struct.pack("<I", len(bone_frames)))
            
            unit = animation_dict.get('unit', 0.085)
            
            # Write keyframes
            for frame in bone_frames:
                name = frame['name']
                # Translate back to Japanese if possible
                jp_name = bone_name_reverse_translation.get(name, name)
                
                try:
                    name_bytes = jp_name.encode('shift_jis')
                except UnicodeEncodeError:
                    name_bytes = jp_name.encode('ascii', 'ignore')
                
                frame_num = int(frame['frame_num'])
                px, py, pz = frame['position']
                rx, ry, rz, rw = frame['rotation']
                
                # Default bezier interpolation if not present
                bezier = frame.get('bezier', b'\x00' * 64)
                if len(bezier) != 64:
                     # Standard VMD interpolation (S-curve)
                    bezier = bytearray([20, 20, 20, 20] * 16)
                    
                f.write(struct.pack("<15s I 3f 4f 64s", 
                                  name_bytes[:15].ljust(15, b'\x00'),
                                  frame_num,
                                  px / unit, py / unit, pz / unit,
                                  rx, ry, rz, rw,
                                  bytes(bezier)))
                                  
            # Morph Frames
            morph_frames = animation_dict.get('morph_frames', [])
            f.write(struct.pack("<I", len(morph_frames)))
            for frame in morph_frames:
                name = frame['name']
                try:
                    name_bytes = name.encode('shift_jis')
                except UnicodeEncodeError:
                    name_bytes = name.encode('ascii', 'ignore')
                    
                f.write(struct.pack("<15s I f",
                                  name_bytes[:15].ljust(15, b'\x00'),
                                  int(frame['frame_num']),
                                  frame['value']))
                                  
            # Camera Frames
            camera_frames = animation_dict.get('camera_frames', [])
            f.write(struct.pack("<I", len(camera_frames)))
            for frame in camera_frames:
                frame_num = int(frame['frame_num'])
                dist = frame['dist']
                px, py, pz = frame['position']
                
                # Convert quaternion back to X-Y-Z euler (approximate)
                rot_quat = frame['rotation'] # xyzw
                r = R.from_quat(rot_quat)
                euler = r.as_euler('xyz', degrees=True)
                rx_deg, ry_deg, rz_deg = euler
                # VMD stores neg radians/degrees logic? 
                # Parse: rx_deg = -math.degrees(rx) => rx = -rx_deg * pi/180
                rx = -math.radians(rx_deg)
                ry = -math.radians(ry_deg)
                rz = math.radians(rz_deg)
                
                fov = int(frame.get('fov', 30))
                bezier = frame.get('bezier', b'\x00' * 24)
                if len(bezier) != 24:
                    bezier = bytearray([20] * 24)
                is_perspective = 0 # Default OFF
                
                f.write(struct.pack("<I f 3f 3f 24s I B",
                                  frame_num,
                                  dist,
                                  px / unit, py / unit, pz / unit,
                                  rx, ry, rz,
                                  bytes(bezier),
                                  fov,
                                  is_perspective))
                                  
            # Light frames (0)
            f.write(struct.pack("<I", 0))
            # Shadow frames (0)
            f.write(struct.pack("<I", 0))
            # IK frames (0 for now)
            f.write(struct.pack("<I", 0))
            
            return True
    except Exception as e:
        print(f"Error writing VMD: {e}")
        return False

def vmd_to_motion_data(file_path, unit=0.085, fps=30.0, mode='local', verbose=True, leg_ik=False):
    """
    Process VMD and return separated character and camera data.
    """
    root, all_bones = build_standard_skeleton()
    center = root.find("Center")
    
    success, anim = parse_vmd(file_path, unit=unit, fps=fps)
    if not success:
        return None
    
    load_vmd_to_skeleton(anim, all_bones)
    verify_global_positions(root)
    
    totalFrames = int(anim["duration"] * fps) + 1
    if totalFrames <= 0:
        if verbose: print(f"No frames found in animation: {file_path}")
        return None
    
    character_data = []
    camera_data = []
    cam_frames_dict = {f["frame_num"]: f for f in anim["camera_frames"]}
    
    if verbose: print(f"Processing {totalFrames} frames...")

    for frame in range(totalFrames):
        animate_skeleton(root, frame, leg_ik)
        center.update_world_pos()

        # Character
        centerPos = center.globalPos.copy() - center.zeroPos
        
        frame_char = []
        # Format: Center Offset(3) + Scale(1)? + Bones...
        frame_char.extend(centerPos * 1000 / 32768)
        frame_char.append(1.0)
        frame_char.extend(root.export_data(mode=mode))
        character_data.append(frame_char)
        
        # Camera
        frame_cam = []
        if frame in cam_frames_dict:
            cf = cam_frames_dict[frame]
            # Convert to LookAt + Dist + Rot representation
            # We construct camera_pos from look_at (stored in position)
            # But the VMD stores TargetPos, Dist, Rot.
            # Our exported NPY format was: CameraPos(3), FOV(1), Rot(4).
            
            look_at = np.array(cf["position"])
            dist = cf["dist"]
            rot = cf["rotation"]
            
            # Calc camera pos
            rot_mat = R.from_quat(rot).as_matrix()
            forward = rot_mat[:, 2]
            cam_pos = look_at + forward * dist
            
            frame_cam.extend(cam_pos * 1000 / 32768)
            frame_cam.append(float(cf["fov"]) / 180)
            frame_cam.extend(rot)
        else:
            frame_cam.extend([0,0.03,0.1])
            frame_cam.append(30.0 / 180)
            frame_cam.extend([0,0,0,1])
        camera_data.append(frame_cam)
        
    return {
        'actor': np.array(character_data, dtype=np.float32),
        'camera': np.array(camera_data, dtype=np.float32) if anim["camera_frames"] else None
    }
    
def load_vmd_to_skeleton(animation, skeleton_bones):
    """
    Load VMD animation data into the skeleton bones.
    """
    for bone_name, frames in animation["bones"].items():
        translated_name = bone_name_translation.get(bone_name, bone_name)
        if translated_name in skeleton_bones:
            bone = skeleton_bones[translated_name]
            bone.frames = frames

def update_bone_for_frame(bone, frame_num):
    bone.update_for_frame(frame_num)

def animate_skeleton(root_bone, frame_num, leg_ik):
    """Update the skeleton to a specific frame number."""
    update_bone_for_frame(root_bone, frame_num)
    
    # Calculate global positions before IK so solver uses current frame's hip/target positions
    root_pos = np.zeros(3)
    root_rot = np.identity(3)
    root_bone.calc_world_pos(root_pos, root_rot)

    if leg_ik:
        apply_leg_ik(root_bone)        
        # Recalculate global positions after IK modified leg rotations
        root_pos = np.zeros(3)
        root_rot = np.identity(3)
        root_bone.calc_world_pos(root_pos, root_rot)

def apply_leg_ik(root_bone):
    from .ik import solve_two_bone_ik
    left_leg_ik = root_bone.find("LeftLegIK")
    right_leg_ik = root_bone.find("RightLegIK")
    if left_leg_ik:
        apply_single_leg_ik(root_bone, "Left", left_leg_ik.globalPos)
    if right_leg_ik:
        apply_single_leg_ik(root_bone, "Right", right_leg_ik.globalPos)

def has_ik_movement(ik_bone):
    if not ik_bone:
        return False
    return ik_bone.frames and len(ik_bone.frames) > 1

def apply_single_leg_ik(root_bone, side, target_pos):
    hip_bone = root_bone.find(f"{side}Leg")
    knee_bone = root_bone.find(f"{side}Knee") 
    ankle_bone = root_bone.find(f"{side}Ankle")
    if not all([hip_bone, knee_bone, ankle_bone]):
        return
    try:
        from .ik import solve_ik_geometry, calculate_rotation_between_vectors
        
        hip_pos = hip_bone.globalPos
        knee_pos = knee_bone.globalPos
        ankle_pos = ankle_bone.globalPos
        
        # Calculate bone lengths
        upper_leg_length = np.linalg.norm(knee_pos - hip_pos)
        lower_leg_length = np.linalg.norm(ankle_pos - knee_pos)
        
        # Calculate target knee position using geometric IK
        # Pass current knee_pos as hint to keep bending consistent
        target_knee_pos_global = solve_ik_geometry(
            hip_pos, target_pos, upper_leg_length, lower_leg_length, knee_pos
        )
        
        # --- Calculate Hip Rotation ---
        # 1. Determine local bone rest direction
        v_upper_rest_local = knee_bone.zeroPos - hip_bone.zeroPos
        norm_upper = np.linalg.norm(v_upper_rest_local)
        if norm_upper > 1e-6:
            v_upper_rest_local /= norm_upper
        else:
            v_upper_rest_local = np.array([0, -1, 0])
            
        # 2. Convert target direction to parent space
        v_upper_target_global = target_knee_pos_global - hip_pos
        parent_rot = R.from_quat(hip_bone.parent.globalQuat)
        v_upper_target_local = parent_rot.inv().apply(v_upper_target_global)
        
        # 3. Calculate hip local rotation
        hip_rotation = calculate_rotation_between_vectors(v_upper_rest_local, v_upper_target_local)
        hip_bone.quat = hip_rotation.as_quat()
        
        # Update hip bone global transform immediately so child (knee) uses correct parent transform
        hip_bone.calc_world_pos(hip_bone.parent.globalPos, hip_bone.parent.globalQuat)
        
        # --- Calculate Knee Rotation ---
        # 1. Determine local bone rest direction
        v_lower_rest_local = ankle_bone.zeroPos - knee_bone.zeroPos
        norm_lower = np.linalg.norm(v_lower_rest_local)
        if norm_lower > 1e-6:
            v_lower_rest_local /= norm_lower
        else:
            v_lower_rest_local = np.array([0, -1, 0])
            
        # 2. Convert target to parent (Hip Global) space
        # Note: Knee's parent is Hip
        v_lower_target_global = target_pos - target_knee_pos_global
        
        hip_global_rot = R.from_quat(hip_bone.globalQuat)
        v_lower_target_local = hip_global_rot.inv().apply(v_lower_target_global)
        
        # 3. Calculate knee local rotation
        knee_rotation = calculate_rotation_between_vectors(v_lower_rest_local, v_lower_target_local)
        knee_bone.quat = knee_rotation.as_quat()
        
    except Exception as e:
        print(f"Warning: IK failed for {side.lower()} leg: {e}")
