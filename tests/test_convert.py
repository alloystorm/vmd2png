import sys
import os
import numpy as np
import shutil

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))

from vmd2png.vmd import write_vmd, parse_vmd
from vmd2png.converter import export_vmd_to_files, convert_motion_to_vmd, load_motion_dict
from vmd2png.preview import preview_motion

#export_vmd_to_files("data/conqueror.vmd", "test_output/conqueror", leg_ik=False, camera_vmd_path="data/conqueror_cam.vmd")
preview_motion("test_output/conqueror/conqueror_with_cam_motion.png", mode='combined', leg_ik=False)
#preview_motion("data/conqueror.vmd", leg_ik=False)