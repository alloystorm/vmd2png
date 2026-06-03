"""
T-pose -> A-pose motion conversion.

Some motions are authored for **T-pose** models (arms straight out horizontally),
but this package's standard skeleton is **A-pose** (arms angled ~27 deg down — see
the arm bone positions in skeleton.py). Playing a T-authored motion on the A-pose
skeleton leaves the arms ~27 deg too low.

This module retargets the arm chain so a T-authored motion looks correct on the
A-pose skeleton: it rotates each arm-chain bone's rest direction to horizontal (the
T-pose), then re-expresses every keyframe rotation in that corrected frame.

Math (verified against the Project Sekai "Base Model.pmx", whose arm/forearm rest
exactly horizontal): for an arm-chain bone b with rest-direction correction D_b
(the minimal rotation taking the A-pose segment direction to horizontal),

    q'_b = D_parent^-1 . q_b . D_b

where D_parent is the correction of b's parent in the chain (identity above the
arm). The post-multiply by D_b re-aims the bone's child to the T-pose direction;
the pre-multiply by D_parent^-1 cancels the rotation the corrected parent already
introduced, so the chain stays consistent. Bones with no keyframes get a single
rest keyframe carrying the correction.

This corrects the dominant, clearly-defined arm drop. The shoulder (~4 deg) and
per-finger rest differences are left as-is; pass a custom `chain` to extend it.
"""
import numpy as np
from scipy.spatial.transform import Rotation as R

# Arm-chain bones to correct, per side, in hierarchy (parent-before-child) order.
# Twist bones are included so the correction passes through them cleanly.
_LEFT_CHAIN = ["LeftArm", "LeftArmTwist", "LeftElbow", "LeftHandTwist", "LeftWrist"]
_RIGHT_CHAIN = ["RightArm", "RightArmTwist", "RightElbow", "RightHandTwist", "RightWrist"]


def _immediate_child(bone):
    """The chain continues through the single non-optional child (arm bones have
    one structural child until the wrist, which branches into fingers)."""
    kids = [c for c in bone.children if not c.isOptional]
    return kids[0] if kids else None


def _horizontal_correction(bone):
    """Minimal rotation aligning this bone's rest segment direction (toward its
    child) to the horizontal plane. Identity if it has no usable child (wrist)."""
    child = _immediate_child(bone)
    if child is None:
        return R.identity()
    d = np.asarray(child.zeroPos, float) - np.asarray(bone.zeroPos, float)
    n = np.linalg.norm(d)
    if n < 1e-8:
        return R.identity()
    d = d / n
    horiz = np.array([d[0], 0.0, d[2]])
    hn = np.linalg.norm(horiz)
    if hn < 1e-6:
        return R.identity()                 # bone points straight up/down; skip
    return R.align_vectors([horiz / hn], [d])[0]


def build_corrections(bones, chain=None):
    """Return {bone_name: (D_parent, D_b)} scipy Rotations for the arm chains.

    D_b makes the bone's rest segment horizontal; D_parent is the correction of
    the previous chain bone (identity at the top of each chain)."""
    chains = [chain] if chain is not None else [_LEFT_CHAIN, _RIGHT_CHAIN]
    out = {}
    for names in chains:
        d_parent = R.identity()
        for name in names:
            bone = bones.get(name)
            if bone is None:
                continue
            d_b = _horizontal_correction(bone)
            out[name] = (d_parent, d_b)
            d_parent = d_b
    return out


def convert_t_to_a_pose(bones, chain=None):
    """In-place: rewrite arm-chain keyframe rotations so a T-pose-authored motion
    plays correctly on the A-pose skeleton. Returns the number of bones corrected.

    `bones` is the name->Bone dict from build_standard_skeleton (already loaded
    with the VMD's keyframes via load_vmd_to_skeleton)."""
    corrections = build_corrections(bones, chain)
    n = 0
    for name, (d_parent, d_b) in corrections.items():
        bone = bones.get(name)
        if bone is None:
            continue
        dp_inv = d_parent.inv()
        if bone.frames:
            for fr in bone.frames:
                q = R.from_quat(np.asarray(fr["rotation"], float))
                fr["rotation"] = (dp_inv * q * d_b).as_quat()
        else:
            # No keyframes -> the bone sits at rest; bake the correction as frame 0
            # so the rest pose itself becomes A-pose-correct.
            q_rest = (dp_inv * d_b).as_quat()
            bone.frames = [{"frame_num": 0, "position": (0.0, 0.0, 0.0),
                            "rotation": tuple(q_rest), "bezier": b""}]
            bone.frames_sorted = True
        n += 1
    return n
