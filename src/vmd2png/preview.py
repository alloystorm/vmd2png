import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.widgets import Slider, Button
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
    plt.subplots_adjust(bottom=0.25) # Make room for the slider
    
    center = root.find("Center")

    # Animation state
    anim_state = {
        'frame': 0,
        'running': True
    }

    def update_plot(frame):
        ax.clear()
        
        # Animate
        animate_skeleton(root, frame)
        center.update_world_pos() # Ensure global positions are updated
        
        # Dynamic axis limits based on Center bone
        cx, cy, cz = center.globalPos[0], center.globalPos[1], center.globalPos[2]
        
        # Matplotlib 3D axes: X is X, Y is Z(depth), Z is Y(up) based on mplot3d conventions vs standard 3D logic
        # But in this code:
        # plot_skeleton_3d uses: x=global[0], y=global[2] (depth), z=global[1] (up)
        # So ax.set_xlim corresponds to global[0]
        # ax.set_ylim corresponds to global[2]
        # ax.set_zlim corresponds to global[1]

        radius = 20 # Original large view
        # The user requested +/- 2 relative to center. 
        # But global[1] is vertical (Y), global[2] is depth (Z).
        
        ax.set_xlim(cx - 2, cx + 2)
        ax.set_ylim(cz - 2, cz + 2) # Depth
        ax.set_zlim(0, 4) # Height usually starts from ground
        
        # User requested: "plot range from global position of center bone +/- 2"
        # Since this is a very tight zoom, let's apply it strictly as requested for X/Z (horizontal plane)
        # and keep Y (vertical) reasonable or centered too.
        
        # Applying tight focus on center bone
        ax.set_xlim(cx - 2, cx + 2) # Keeping wide X for visibility, or user wants tight?
        # Re-reading: "update the plot range from global position of center bone +/- 2" implies tight follow.
        
        range_val = 15 # Keep it viewable. +/- 2 might be too small for a full skeleton.
                       # Assuming user meant "follow camera" logic. 
                       # Let's interpret +/- 2 as offset from the center bone specific to the viewport center,
                       # but keep the viewport size meaningful.
        
        # Actually, let's try to center the camera on the bone but keep the field of view constant.
        ax.set_xlim(cx - 2, cx + 2)
        ax.set_ylim(cz - 2, cz + 2)
        ax.set_zlim(0, 4)
        
        # If the user literally wants the axis limits to be [cx-2, cx+2], that's a 4 unit window.
        # Most MMD models are ~15-20 units tall. A 4 unit window will only show the hips.
        # I will assume they want the camera CENTERED on the bone, with the previous scale.
        # But let's check if they meant +/- 20? Or maybe they really want a macro view?
        # Given "motion_encoder", maybe precise tracking is needed.
        # Let's implement centering with a reasonable Fixed FOV, but shifted by Center position.
        
        # Implementation of centering logic:
        # Shift limits so center is in the middle
        ax.set_xlim(cx - 2, cx + 2)
        ax.set_ylim(cz - 2, cz + 2)
        
        # Plot
        plot_skeleton_3d(root, ax)
        ax.set_title(f"Frame: {int(frame)}")
        ax.set_xlabel('X')
        ax.set_ylabel('Z (Depth)')
        ax.set_zlabel('Y (Up)')

    def update(frame):
        if not anim_state['running']:
            return
            
        current_frame = anim_state['frame']
        if current_frame >= total_frames - 1:
            current_frame = 0
        else:
            current_frame += 1
            
        anim_state['frame'] = current_frame
        
        # Determine if we need to update slider visually without triggering callback loop
        if slider.val != current_frame:
            slider.eventson = False
            slider.set_val(current_frame)
            slider.eventson = True
            
        return update_plot(current_frame)

    # Slider setup
    ax_slider = plt.axes([0.25, 0.1, 0.65, 0.03], facecolor='lightgoldenrodyellow')
    slider = Slider(ax_slider, 'Frame', 0, total_frames - 1, valinit=0, valfmt='%0.0f')
    
    def on_slider_change(val):
        anim_state['frame'] = int(val)
        update_plot(anim_state['frame'])
        
    slider.on_changed(on_slider_change)

    # Play/Pause Button
    ax_play = plt.axes([0.1, 0.1, 0.1, 0.04])
    btn_play = Button(ax_play, 'Pause', color='lightgoldenrodyellow', hovercolor='0.975')

    def toggle_play(event):
        anim_state['running'] = not anim_state['running']
        btn_play.label.set_text('Pause' if anim_state['running'] else 'Play')
        
    btn_play.on_clicked(toggle_play)

    # Use a generator or infinite loop for frames so we control flow manually
    ani = animation.FuncAnimation(fig, update, frames=None, interval=1000/fps, cache_frame_data=False)
    plt.show()

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        preview_motion(sys.argv[1])
