import numpy as np
from scipy.spatial.transform import Rotation as R

def solve_two_bone_ik(start_pos, end_pos, upper_length, lower_length, knee_hint_pos=None):
    """
    Solve two-bone inverse kinematics (like for a leg: hip-knee-ankle).
    Returns global rotations assuming default bone direction is (0,0,-1).
    kept for backward compatibility with initial implementation logic, 
    but likely you want to use solve_ik_geometry and calculate rotations yourself.
    """
    knee_pos = solve_ik_geometry(start_pos, end_pos, upper_length, lower_length, knee_hint_pos)
    
    # Hip rotation: rotate from initial direction to actual upper bone direction
    initial_bone_dir = np.array([0, 0, -1])  # Assume bones point down initially
    actual_upper_bone_dir = (knee_pos - start_pos) / upper_length
    hip_rotation = calculate_rotation_between_vectors(initial_bone_dir, actual_upper_bone_dir)
    
    # Knee rotation: rotate from upper bone direction to lower bone direction
    actual_lower_bone_dir = (end_pos - knee_pos) / lower_length
    knee_rotation = calculate_rotation_between_vectors(actual_upper_bone_dir, actual_lower_bone_dir)
    
    return hip_rotation.as_quat(), knee_rotation.as_quat()

def solve_ik_geometry(start_pos, end_pos, upper_length, lower_length, knee_hint_pos=None):
    """
    Calculate the joint position (knee) for a two-bone IK chain.
    """
    # Vector from start to end
    target_vector = end_pos - start_pos
    target_distance = np.linalg.norm(target_vector)
    
    # Handle zero distance case
    if target_distance < 1e-6:
        return start_pos + np.array([0, -upper_length, 0]) # Arbitrary default
    
    # Check if target is reachable
    max_reach = upper_length + lower_length
    min_reach = abs(upper_length - lower_length)
    
    if target_distance > max_reach:
        target_distance = max_reach * 0.999
        target_vector = (target_vector / np.linalg.norm(target_vector)) * target_distance
    elif target_distance < min_reach:
        target_distance = min_reach * 1.001
        target_vector = (target_vector / np.linalg.norm(target_vector)) * target_distance
    
    # Use law of cosines to find angles
    cos_hip_angle = (upper_length**2 + target_distance**2 - lower_length**2) / (2 * upper_length * target_distance)
    cos_hip_angle = np.clip(cos_hip_angle, -1.0, 1.0)
    hip_angle = np.arccos(cos_hip_angle)
    
    # Normalize target direction
    target_dir = target_vector / np.linalg.norm(target_vector)
    
    # Calculate the knee position using the triangle geometry
    hip_to_knee_proj = upper_length * np.cos(hip_angle)
    knee_height = upper_length * np.sin(hip_angle)
    
    # Determine knee bend direction
    if knee_hint_pos is not None:
        hint_vector = knee_hint_pos - start_pos
        hint_proj = hint_vector - np.dot(hint_vector, target_dir) * target_dir
        if np.linalg.norm(hint_proj) > 1e-6:
            bend_dir = hint_proj / np.linalg.norm(hint_proj)
        else:
            bend_dir = get_default_bend_direction(target_dir)
    else:
        bend_dir = get_default_bend_direction(target_dir)
    
    return start_pos + target_dir * hip_to_knee_proj + bend_dir * knee_height

def get_default_bend_direction(target_dir):
    """Get default bend direction perpendicular to target direction."""
    up_vector = np.array([0, 1, 0])
    bend_dir = np.cross(target_dir, up_vector)
    if np.linalg.norm(bend_dir) < 1e-6:
        # Target is parallel to up vector, use different reference
        bend_dir = np.array([1, 0, 0])
    else:
        bend_dir = bend_dir / np.linalg.norm(bend_dir)
    return bend_dir

def calculate_rotation_between_vectors(from_vec, to_vec):
    """Calculate rotation from one vector to another."""
    from_vec = from_vec / np.linalg.norm(from_vec)
    to_vec = to_vec / np.linalg.norm(to_vec)
    
    # Check if vectors are the same
    if np.allclose(from_vec, to_vec):
        return R.identity()
    
    # Check if vectors are opposite
    if np.allclose(from_vec, -to_vec):
        # Find a perpendicular vector
        if abs(from_vec[0]) < 0.9:
            perp = np.array([1, 0, 0])
        else:
            perp = np.array([0, 1, 0])
        axis = np.cross(from_vec, perp)
        axis = axis / np.linalg.norm(axis)
        return R.from_rotvec(np.pi * axis)
    
    # Normal case
    axis = np.cross(from_vec, to_vec)
    axis = axis / np.linalg.norm(axis)
    angle = np.arccos(np.clip(np.dot(from_vec, to_vec), -1.0, 1.0))
    return R.from_rotvec(angle * axis)

def clamp_to_reachable_distance(start_pos, target_pos, max_reach, min_reach=0):
    """
    Clamp a target position to be within reachable distance.
    
    Args:
        start_pos: Starting position
        target_pos: Desired target position
        max_reach: Maximum reachable distance
        min_reach: Minimum reachable distance (default 0)
        
    Returns:
        Clamped target position
    """
    vector = target_pos - start_pos
    distance = np.linalg.norm(vector)
    
    if distance > max_reach:
        return start_pos + (vector / distance) * max_reach
    elif distance < min_reach:
        if distance > 1e-6:
            return start_pos + (vector / distance) * min_reach
        else:
            return start_pos + np.array([0, 0, min_reach])  # Arbitrary direction
    
    return target_pos
