import numpy as np
from .bone import Bone
from scipy.spatial.transform import Rotation as R

# Skeleton definition from original script
skeleton_def = """
[Master] (0.00, 0.00, 0.00)
	Center (0.00, 1.14, 0.00)
		[Groove] (0.00, 1.14, 0.00)
			[Waist] (0.00, 1.14, 0.00)
				HipMaster (0.00, 1.14, 0.00)
					LeftLeg (0.07, 0.96, 0.05)
						[LeftLegTwist] (0.07, 0.75, 0.05)
						LeftKnee (0.07, 0.55, 0.04)
							LeftAnkle (0.08, 0.10, 0.06)
								[LeftToe] (0.08, 0.00, -0.11)
					RightLeg (-0.07, 0.96, 0.05)
						[RightLegTwist] (-0.07, 0.75, 0.05)
						RightKnee (-0.07, 0.55, 0.04)
							RightAnkle (-0.08, 0.10, 0.06)
								[RightToe] (-0.08, 0.00, -0.11)
				Torso (0.00, 1.14, 0.00)
					[Torso1] (0.00, 1.20, 0.00)
						Torso2 (0.00, 1.24, 0.00)
							Neck (0.00, 1.42, 0.05)
								Head (0.00, 1.48, 0.04)
									[Eyes] (0.00, 1.64, 0.00)
									[RightEye] (-0.03, 1.54, 0.00)
										[RightEyeTip] (-0.03, 1.54, 0.02)
									[LeftEye] (0.03, 1.54, 0.00)
										[LeftEyeTip] (0.03, 1.54, 0.02)
							[LeftBreast] (0.02, 1.29, 0.12)
							[RightBreast] (-0.02, 1.29, 0.12)
							[LeftShoulder] (0.02, 1.37, 0.06)
								LeftArm (0.11, 1.36, 0.06)
									[LeftArmTwist] (0.21, 1.31, 0.06)
										LeftElbow (0.31, 1.26, 0.06)
											[LeftHandTwist] (0.40, 1.21, 0.06)
												LeftWrist (0.50, 1.16, 0.06)
													[LeftIndexFinger1] (0.57, 1.14, 0.05)
														[LeftIndexFinger2] (0.60, 1.12, 0.04)
															[LeftIndexFinger3] (0.61, 1.11, 0.04)
													[LeftPinky1] (0.56, 1.14, 0.09)
														[LeftPinky2] (0.58, 1.13, 0.10)
															[LeftPinky3] (0.60, 1.12, 0.10)
													[LeftRingFinger1] (0.57, 1.14, 0.08)
														[LeftRingFinger2] (0.60, 1.12, 0.08)
															[LeftRingFinger3] (0.62, 1.11, 0.09)
													[LeftMiddleFinger1] (0.57, 1.14, 0.06)
														[LeftMiddleFinger2] (0.60, 1.12, 0.06)
															[LeftMiddleFinger3] (0.62, 1.11, 0.06)
													[LeftThumb0] (0.51, 1.15, 0.05)
														[LeftThumb1] (0.54, 1.13, 0.03)
															[LeftThumb2] (0.55, 1.11, 0.02)
							[RightShoulder] (-0.02, 1.37, 0.06)
								RightArm (-0.11, 1.36, 0.06)
									[RightArmTwist] (-0.21, 1.31, 0.06)
										RightElbow (-0.31, 1.26, 0.06)
											[RightHandTwist] (-0.40, 1.21, 0.06)
												RightWrist (-0.50, 1.16, 0.06)
													[RightIndexFinger1] (-0.57, 1.14, 0.05)
														[RightIndexFinger2] (-0.60, 1.12, 0.04)
															[RightIndexFinger3] (-0.61, 1.11, 0.04)
													[RightPinky1] (-0.56, 1.14, 0.09)
														[RightPinky2] (-0.58, 1.13, 0.10)
															[RightPinky3] (-0.60, 1.12, 0.10)
													[RightRingFinger1] (-0.57, 1.14, 0.08)
														[RightRingFinger2] (-0.60, 1.12, 0.08)
															[RightRingFinger3] (-0.62, 1.11, 0.09)
													[RightMiddleFinger1] (-0.57, 1.14, 0.06)
														[RightMiddleFinger2] (-0.60, 1.12, 0.06)
															[RightMiddleFinger3] (-0.62, 1.11, 0.06)
													[RightThumb0] (-0.51, 1.15, 0.05)
														[RightThumb1] (-0.54, 1.13, 0.03)
															[RightThumb2] (-0.55, 1.11, 0.02)
	LeftLegIKParent (0.08, 0.10, 0.06)
		LeftLegIK (0.08, 0.10, 0.06)
    RightLegIKParent (-0.08, 0.10, 0.06)
		RightLegIK (-0.08, 0.10, 0.06)
"""

def build_standard_skeleton():
    """
    Build a skeleton based on a standard format.
    Returns the root bone and a dictionary of all bones by name.
    """
    bones_dict = {}
    parent_stack = []

    for line in skeleton_def.split('\n'):
        original_line = line
        stripped_line = line.lstrip()
        indent = 0
        while line.startswith("\t"):
            line = line[1:]
            indent += 1
        line = stripped_line.strip()
        if not line:
            continue

        has_brackets = line.startswith("[")
        if has_brackets:
            name = line[1:line.index("]")]
        else:
            name = line[:line.index(" ")]

        pos_start = line.index("(")
        pos_end = line.index(")")
        pos_str = line[pos_start+1:pos_end]
        x, y, z = map(float, pos_str.split(", "))

        bone = Bone(name, (x, y, z), has_brackets)
        bone.globalPos = np.array([x, y, z])
        bones_dict[name] = bone

        while parent_stack and indent < len(parent_stack):
            parent_stack.pop()

        if parent_stack:
            parent = parent_stack[-1]
            parent.children.append(bone)
            bone.set_parent(parent)

        if indent >= len(parent_stack):
            parent_stack.append(bone)
        else:
            parent_stack[indent] = bone

    root = bones_dict["Master"]
    return root, bones_dict

def verify_global_positions(root_bone):
    def verify_bone(bone):
        if bone.parent:
            calculated_pos = bone.parent.globalPos + R.from_quat(bone.parent.quat).apply(bone.offset)
        for child in bone.children:
            verify_bone(child)
    verify_bone(root_bone)
