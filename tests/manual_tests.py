import sys
import os
import numpy as np
import shutil

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))

from vmd2png.vmd import write_vmd, parse_vmd
from vmd2png.converter import export_vmd_to_files, convert_motion_to_vmd, load_motion_dict
from vmd2png.preview import preview_motion

#export_vmd_to_files("data/conqueror.vmd", "test_output", leg_ik=False, camera_vmd_path="data/conqueror_cam.vmd")
#convert_motion_to_vmd("test_output/conqueror.png", "test_output/reconstructed.vmd")
#preview_motion("test_output/reconstructed.vmd", leg_ik=False)
# preview_motion("test_output/conqueror/conqueror_with_cam_motion.png", leg_ik=False)
# preview_motion("data/conqueror.vmd", camera_vmd_path="data/conqueror_cam.vmd", leg_ik=False)
# preview_motion("data/kimagure.vmd", leg_ik=True)
preview_motion("data/conqueror.vmd", camera_vmd_path="data/conqueror_cam.vmd")