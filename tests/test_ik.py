import unittest
import numpy as np
import sys
import os

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from scipy.spatial.transform import Rotation as R
from vmd2png.ik import solve_two_bone_ik, calculate_rotation_between_vectors, clamp_to_reachable_distance

class TestIK(unittest.TestCase):
    def test_calculate_rotation_between_vectors(self):
        v1 = np.array([1.0, 0.0, 0.0])
        v2 = np.array([0.0, 1.0, 0.0])
        rot = calculate_rotation_between_vectors(v1, v2)
        rotated_v1 = rot.apply(v1)
        np.testing.assert_allclose(rotated_v1, v2, atol=1e-6)

        # Identity
        rot = calculate_rotation_between_vectors(v1, v1)
        # Identity can be [0,0,0,1] or [0,0,0,-1] or similar
        self.assertTrue(np.allclose(rot.as_quat(), [0.0, 0.0, 0.0, 1.0]) or 
                        np.allclose(np.abs(rot.as_quat()), [0.0, 0.0, 0.0, 1.0]))

        # Opposite
        v3 = np.array([-1.0, 0.0, 0.0])
        rot = calculate_rotation_between_vectors(v1, v3)
        rotated_v1 = rot.apply(v1)
        np.testing.assert_allclose(rotated_v1, v3, atol=1e-6)

    def test_solve_two_bone_ik_reachable(self):
        start_pos = np.array([0.0, 0.0, 0.0])
        upper_len = 1.0
        lower_len = 1.0
        
        # Case 1: Straight leg along -Z axis
        # initial_bone_dir = np.array([0, 0, -1]) in ik.py
        end_pos = np.array([0.0, 0.0, -2.0])
        
        hip_quat, knee_quat = solve_two_bone_ik(start_pos, end_pos, upper_len, lower_len)
        
        hip_rot = R.from_quat(hip_quat)
        knee_rot = R.from_quat(knee_quat)
        
        v_init = np.array([0.0, 0.0, -1.0])
        
        # Apply hip rotation to initial upper bone 
        v_upper = hip_rot.apply(v_init) * upper_len
        
        # Knee rotation rotates from ACTUAL upper bone direction to LOWER bone direction
        v_upper_dir = v_upper / np.linalg.norm(v_upper)
        v_lower = knee_rot.apply(v_upper_dir) * lower_len
        
        end_effector = start_pos + v_upper + v_lower
        
        np.testing.assert_allclose(end_effector, end_pos, atol=1e-5)

    def test_solve_two_bone_ik_bent(self):
        start_pos = np.array([0.0, 0.0, 0.0])
        upper_len = 1.0
        lower_len = 1.0
        # Target somewhat close so knee must bend
        end_pos = np.array([0.0, 1.0, -1.0]) 
        
        hip_quat, knee_quat = solve_two_bone_ik(start_pos, end_pos, upper_len, lower_len)
        
        hip_rot = R.from_quat(hip_quat)
        knee_rot = R.from_quat(knee_quat)
        
        v_init = np.array([0.0, 0.0, -1.0])
        v_upper = hip_rot.apply(v_init) * upper_len
        v_upper_dir = v_upper / np.linalg.norm(v_upper)
        v_lower = knee_rot.apply(v_upper_dir) * lower_len
        
        end_effector = start_pos + v_upper + v_lower
        
        np.testing.assert_allclose(end_effector, end_pos, atol=1e-5)

    def test_solve_two_bone_ik_with_hint(self):
        start_pos = np.array([0.0, 0.0, 0.0])
        upper_len = 1.0
        lower_len = 1.0
        end_pos = np.array([0.0, 0.0, -1.0]) # Target at -1 Z. 
        
        # Hint to bend towards +X
        knee_hint = np.array([1.0, 0.0, -0.5])
        
        hip_quat, knee_quat = solve_two_bone_ik(start_pos, end_pos, upper_len, lower_len, knee_hint_pos=knee_hint)
        
        hip_rot = R.from_quat(hip_quat)
        
        v_init = np.array([0.0, 0.0, -1.0])
        v_upper = hip_rot.apply(v_init) * upper_len
        
        # The knee position (end of upper bone) should have positive X component
        knee_pos = start_pos + v_upper
        self.assertGreater(knee_pos[0], 0.1)

    def test_solve_two_bone_ik_unreachable(self):
        start_pos = np.array([0.0, 0.0, 0.0])
        upper_len = 1.0
        lower_len = 1.0
        end_pos = np.array([0.0, 0.0, -5.0]) # Too far
        
        hip_quat, knee_quat = solve_two_bone_ik(start_pos, end_pos, upper_len, lower_len)
        
        hip_rot = R.from_quat(hip_quat)
        knee_rot = R.from_quat(knee_quat)
        
        v_init = np.array([0.0, 0.0, -1.0])
        v_upper = hip_rot.apply(v_init) * upper_len
        v_upper_dir = v_upper / np.linalg.norm(v_upper)
        v_lower = knee_rot.apply(v_upper_dir) * lower_len
        
        end_effector = start_pos + v_upper + v_lower
        
        # Should be clamped to max reach (approx 2.0)
        dist = np.linalg.norm(end_effector)
        self.assertTrue(dist < 2.01)
        self.assertTrue(dist > 1.95)
        
    def test_clamp_to_reachable_distance(self):
        start = np.array([0.0, 0.0, 0.0])
        target = np.array([3.0, 0.0, 0.0])
        max_reach = 2.0
        
        clamped = clamp_to_reachable_distance(start, target, max_reach)
        np.testing.assert_allclose(clamped, np.array([2.0, 0.0, 0.0]))

if __name__ == '__main__':
    unittest.main()
