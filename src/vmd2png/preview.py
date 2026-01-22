import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from .converter import load_motion_dict
from .skeleton import build_standard_skeleton
from .vmd import load_vmd_to_skeleton, animate_skeleton

def plot_skeleton_3d(root, ax):
    """
    Recursive skeleton plotting.
    """
    xs, ys, zs = [], [], []
    
    def collect_bones(bone, parent_pos):
        # Line from parent to current
        if parent_pos is not None:
             ax.plot([parent_pos[0], bone.globalPos[0]],
                     [parent_pos[2], bone.globalPos[2]],
                     [parent_pos[1], bone.globalPos[1]], 'b-')
             
        for child in bone.children:
            collect_bones(child, bone.globalPos)
            
    # Draw logic
    collect_bones(root, None)

def preview_motion(input_path, mode='character', fps=30):
    """
    Preview motion from VMD, NPY, or PNG file.
    """
    anim = load_motion_dict(input_path, mode)
    if not anim:
        print("Failed to load animation.")
        return
        
    root, map_bones = build_standard_skeleton()
    load_vmd_to_skeleton(anim, map_bones)
    
    total_frames = int(anim["duration"] * 30) # approx
    if total_frames == 0 and "bone_frames" in anim:
        # Estimate from max frame
        max_f = 0
        for f in anim["bone_frames"]:
            if f["frame_num"] > max_f: max_f = f["frame_num"]
        total_frames = max_f + 1
        
    print(f"Previewing {total_frames} frames...")
    
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Setup axis limits (approximate for MMD models)
    ax.set_xlim(-10, 10)
    ax.set_ylim(-10, 10)
    ax.set_zlim(0, 20)
    ax.set_xlabel('X')
    ax.set_ylabel('Z (Depth)')
    ax.set_zlabel('Y (Up)')
    
    center = root.find("Center")

    def update(frame):
        ax.clear()
        ax.set_xlim(-10, 10)
        ax.set_ylim(-10, 10)
        ax.set_zlim(0, 20)
        
        # Animate
        animate_skeleton(root, frame)
        center.update_world_pos() # Ensure global positions are updated
        
        # Plot
        plot_skeleton_3d(root, ax)
        ax.set_title(f"Frame: {frame}")
        
    ani = animation.FuncAnimation(fig, update, frames=range(0, total_frames, 1), interval=1000/fps)
    plt.show()

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        preview_motion(sys.argv[1])
