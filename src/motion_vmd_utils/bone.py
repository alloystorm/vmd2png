import numpy as np
from scipy.spatial.transform import Rotation as R

def rot_lerp(prev_quat, next_quat, t):
    """
    Linear interpolation between quaternions.
    
    Args:
        prev_quat: Starting quaternion as [x, y, z, w]
        next_quat: Ending quaternion as [x, y, z, w]
        t: Interpolation factor (0.0 to 1.0)
    
    Returns:
        Interpolated quaternion
    """
    # Check if we need to flip the quaternion to ensure shortest path
    if np.dot(prev_quat, next_quat) < 0:
        next_quat = -next_quat
        
    lerp_quat = (1 - t) * prev_quat + t * next_quat
    norm = np.linalg.norm(lerp_quat)
    if norm > 1e-10:  # Avoid division by near-zero
        lerp_quat /= norm
        
    return lerp_quat

class Bone:
    def __init__(self, name, zeroPos, optional=False):
        self.name = name
        self.isOptional = optional
        self.isTip = name in ("LeftToe", "RightToe", "LeftEye", "RightEye")
        self.zeroPos = np.array(zeroPos, dtype=float)
        self.offset = np.zeros(3, dtype=float)
        self.children = []
        self.pos = np.zeros(3, dtype=float)
        self.quat = np.array([0.0, 0.0, 0.0, 1.0])  # Store rotation as quaternion [x, y, z, w]
        self.frames = []
        self.frames_sorted = False
        self.globalPos = np.zeros(3, dtype=float)
        self.globalQuat = np.array([0.0, 0.0, 0.0, 1.0])  # Global rotation as quaternion
        self.parent = None

    def set_parent(self, parent):
        self.parent = parent
        self.offset = self.zeroPos - parent.zeroPos
        #print(f"Setting parent of {self.name} to {parent.name}. Offset: {self.offset}")

    def calc_world_pos(self, parent_pos, parent_rot):
        # Convert parent rotation matrix to quaternion if needed
        if isinstance(parent_rot, np.ndarray) and parent_rot.shape == (3, 3):
            parent_quat = R.from_matrix(parent_rot).as_quat()
        else:
            parent_quat = parent_rot
            
        # Calculate global position
        parent_rot_matrix = R.from_quat(parent_quat)
        self.globalPos = parent_pos + parent_rot_matrix.apply(self.offset + self.pos)
        
        # Calculate global rotation (quaternion multiplication)
        parent_r = R.from_quat(parent_quat)
        local_r = R.from_quat(self.quat)
        global_r = parent_r * local_r
        self.globalQuat = global_r.as_quat()
        
        # Update children
        for child in self.children:
            child.calc_world_pos(self.globalPos, self.globalQuat)

    def update_world_pos(self):
        if self.parent:
            self.parent.update_world_pos()
            parent_pos = self.parent.globalPos
            parent_rot = R.from_quat(self.parent.globalQuat)
            self.globalPos = parent_pos + parent_rot.apply(self.offset + self.pos)
            self.globalQuat = parent_rot * R.from_quat(self.quat)
        else:
            self.globalPos = self.zeroPos + self.pos
            self.globalQuat = self.quat

    def export_header(self, withOptional=True):
        header = ""
        if not self.isOptional or withOptional:
            header += f"{self.name} "
        for child in self.children:
            header += child.export_header()
        return header

    # def record_last_frame(self):
    #     self.last_pos = self.globalPos
    #     self.last_quat = self.globalQuat.copy()
    #     for child in self.children:
    #         child.record_last_frame()

    def groundPos(self, pos):
        return np.array([pos[0], 0, pos[2]])

    def update_for_frame(self, frame_num):
        """
        Update this bone's local transform based on animation frames.
        
        Args:
            frame_num: The frame number to update to
        """
        # Update all children first
        for child in self.children:
            child.update_for_frame(frame_num)
            
        # Skip if no animation data for this bone
        if not self.frames:
            return
        
        # Sort frames if needed
        frames = self.frames
        if not self.frames_sorted:
            frames.sort(key=lambda x: x["frame_num"])
            self.frames_sorted = True
        
        # Binary search to find surrounding frames
        low, high = 0, len(frames) - 1
        prev_idx = 0
        
        while low <= high:
            mid = (low + high) // 2
            if frames[mid]["frame_num"] <= frame_num:
                prev_idx = mid
                low = mid + 1
            else:
                high = mid - 1
        
        prev_frame = frames[prev_idx]
        next_idx = prev_idx + 1
        next_frame = frames[next_idx] if next_idx < len(frames) else None
        
        # Apply rotation and position
        if frame_num == prev_frame["frame_num"] or next_frame is None:
            # Direct use, no interpolation needed
            self.set_quat(prev_frame["rotation"])
            if self.name in ("Master", "センター", "Center"):
                self.pos = np.array(prev_frame["position"])
        else:
            t = (frame_num - prev_frame["frame_num"]) / (next_frame["frame_num"] - prev_frame["frame_num"])
            
            # Interpolate rotation
            prev_quat = np.array(prev_frame["rotation"])
            next_quat = np.array(next_frame["rotation"])
            self.set_quat(rot_lerp(prev_quat, next_quat, t))
            
            # Interpolate position for center bones
            if self.name in ("Master", "センター", "Center"):
                prev_pos = np.array(prev_frame["position"])
                next_pos = np.array(next_frame["position"])
                self.pos = prev_pos + t * (next_pos - prev_pos)

    def set_quat(self, quat):
        """
        Set the bone's quaternion, flipping sign if necessary for compatibility.
        """
        quat = np.array(quat)
        if np.dot(self.quat, quat) < 0:
            quat = -quat
        self.quat = quat

    def export_data(self, centerPos, mode='pos', withOptional=True):
        data = []
        if not self.isOptional or withOptional:
            # if self.name == "Center":
            #     data.extend(self.groundPos(self.globalPos) - self.groundPos(self.last_pos))
            if mode == 'pos':
                data.extend(self.globalPos - centerPos)
            if mode == 'global':
                data.extend(self.globalQuat)
            elif mode == 'local':
                data.extend(self.quat)
        for child in self.children:
            data.extend(child.export_data(centerPos, mode))
        return data

    def export_bones(self, withOptional=True):
        bones = []
        if not self.isOptional or withOptional:
            bones.append(self)
        for child in self.children:
            bones.extend(child.export_bones())
        return bones

    def find(self, name):
        if self.name == name:
            return self
        for child in self.children:
            found = child.find(name)
            if found:
                return found
        return None
