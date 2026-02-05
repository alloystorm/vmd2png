import sys
import os
import numpy as np
import shutil

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))

from vmd2png.vmd import write_vmd, parse_vmd
from vmd2png.converter import export_vmd_to_files, convert_motion_to_vmd, load_motion_dict
from vmd2png.preview import preview_motion

# export_vmd_to_files("../data/conqueror_cam.vmd", "test_output/conqueror")
preview_motion("../data/conqueror.vmd")