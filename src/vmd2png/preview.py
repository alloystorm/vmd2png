import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.widgets import Slider, Button, RadioButtons
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from scipy.spatial.transform import Rotation as R
from .converter import load_motion_dict
from .skeleton import build_standard_skeleton
from .vmd import load_vmd_to_skeleton, animate_skeleton, Camera

def draw_camera_frustum(ax, camera, scale=5.0):
    if not camera.frames:
        return

    pos = camera.global_pos
    rot = camera.global_rot
    fov = camera.current_fov
    
    r = R.from_quat(rot)
    
    # Camera looks towards -Z in its local frame (assuming standard convention relative to the 'forward' Z vector from target)
    
    fov_rad = np.radians(fov)
    # scale is distance to far plane
    h = 2.0 * scale * np.tan(fov_rad / 2.0)
    w = h * (16/9) # Assume 16:9
    
    # Frustum far plane corners in local space (Z = -scale)
    corners = np.array([
        [-w/2, h/2, scale],  # Top Left
        [w/2, h/2, scale],   # Top Right
        [w/2, -h/2, scale],  # Bottom Right
        [-w/2, -h/2, scale]  # Bottom Left
    ])
    
    corners_world = r.apply(corners) + pos
    
    # Use helper to map global to plot coordinates: X, Z(depth), Y(up)
    def to_plot(v): return v[0], v[2], v[1]
    
    px, pz, py = to_plot(pos)
    
    for c in corners_world:
        cx, cz, cy = to_plot(c)
        ax.plot([px, cx], [pz, cz], [py, cy], 'k-', lw=1, alpha=0.3)
        
    for i in range(4):
        p1 = to_plot(corners_world[i])
        p2 = to_plot(corners_world[(i+1)%4])
        ax.plot([p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]], 'k-', lw=1, alpha=0.3)
        
    # Draw Up vector
    up_local = np.array([0, 0.5 * scale, 0])
    up_world = r.apply(up_local) + pos
    ux, uz, uy = to_plot(up_world)
    ax.plot([px, ux], [pz, uz], [py, uy], 'r-', lw=1)

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

def _collect_positions(bone, out):
    """Recursively gather every bone's global position."""
    out.append(np.asarray(bone.globalPos, dtype=float))
    for child in bone.children:
        _collect_positions(child, out)


def _equal_box(ax):
    try:
        ax.set_box_aspect((1, 1, 1))
    except Exception:
        pass


def _fit_focused(ax, waist, camera):
    """Focused actor view: center on the actor, range tracks actor-camera distance."""
    pc = waist.globalPos
    radius = 1 #max(float(np.linalg.norm(camera.global_pos - pc)), 1.0)
    ax.set_xlim(pc[0] - radius, pc[0] + radius)
    ax.set_ylim(pc[2] - radius, pc[2] + radius)
    ax.set_zlim(0, radius * 2)
    ax.set_proj_type('ortho')
    _equal_box(ax)


def _fit_overall(ax, positions, camera):
    """Overall view: fit the whole actor and the camera in frame."""
    pts = [np.asarray(p, float) for p in positions]
    if camera.frames:
        pts.append(np.asarray(camera.global_pos, float))
        pts.append(np.asarray(camera.target_pos, float))
    pts = np.array(pts)
    mn, mx = pts.min(0), pts.max(0)
    c = (mn + mx) / 2.0
    half = max(float((mx - mn).max()) / 2.0, 1.0) * 1.1
    ax.set_xlim(c[0] - half, c[0] + half)
    ax.set_ylim(c[2] - half, c[2] + half)
    ax.set_zlim(c[1] - half, c[1] + half)
    ax.set_proj_type('ortho')
    _equal_box(ax)


def _fit_camera(ax, camera):
    """Camera view: look through the camera (orientation + FOV)."""
    import math
    fwd = np.asarray(camera.target_pos, float) - np.asarray(camera.global_pos, float)
    n = float(np.linalg.norm(fwd))
    if n < 1e-6:
        fwd = np.array([0.0, 0.0, 1.0]); n = 1.0
    fwd = fwd / n
    # Plot axes are (worldX, worldZ, worldY); the viewer sits opposite the look dir.
    vd = -np.array([fwd[0], fwd[2], fwd[1]])
    elev = math.degrees(math.asin(float(np.clip(vd[2], -1.0, 1.0))))
    azim = math.degrees(math.atan2(vd[1], vd[0]))
    ax.view_init(elev=elev, azim=azim)

    fov = max(float(camera.current_fov), 1.0)
    half = max(float(camera.distance) * math.tan(math.radians(fov / 2.0)), 1.0)
    c = np.asarray(camera.target_pos, float)
    ax.set_xlim(c[0] - half, c[0] + half)
    ax.set_ylim(c[2] - half, c[2] + half)
    ax.set_zlim(c[1] - half, c[1] + half)
    try:
        ax.set_proj_type('persp', focal_length=1.0 / math.tan(math.radians(fov / 2.0)))
    except TypeError:
        ax.set_proj_type('persp')
    _equal_box(ax)


def preview_motion(input_path, fps=30, leg_ik=False, camera_vmd_path=None, t_pose=False):
    """
    Preview motion from VMD, NPY, or PNG file.

    t_pose: if True, retarget a T-pose-authored motion to the A-pose skeleton
            before playback (arms straight out -> A-pose).
    """
    anim = load_motion_dict(input_path, leg_ik=leg_ik, camera_vmd_path=camera_vmd_path)
    if not anim:
        print("Failed to load animation.")
        return

    root, map_bones = build_standard_skeleton()
    load_vmd_to_skeleton(anim, map_bones)
    if t_pose:
        from .pose_convert import convert_t_to_a_pose
        convert_t_to_a_pose(map_bones)
    
    # Setup Camera
    camera = Camera(anim.get("camera_frames", []))
    
    total_frames = int(anim["duration"] * 30) # approx
    if total_frames == 0:
        max_f = 0
        if "bone_frames" in anim:
            for f in anim["bone_frames"]:
                if f["frame_num"] > max_f: max_f = f["frame_num"]
        if "camera_frames" in anim:
            for f in anim["camera_frames"]:
                if f["frame_num"] > max_f: max_f = f["frame_num"]
        total_frames = max_f + 1
        
    print(f"Previewing {total_frames} frames...")
    
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    plt.subplots_adjust(bottom=0.25) # Make room for the slider
    
    center = root.find("Center")
    waist = root.find("Waist")
    # Animation state
    anim_state = {
        'frame': 0,
        'running': True,
        'view': 'focused'
    }

    def update_plot(frame, leg_ik):
        ax.clear()
        
        # Animate Skeleton
        animate_skeleton(root, frame, leg_ik)
        root.update_world_pos() # Ensure global positions are updated
        
        # Animate Camera
        camera.update(frame)
        
        # Plot Skeleton
        plot_skeleton_3d(center, ax)

        view = anim_state['view']

        # Draw the camera frustum except when looking through the camera itself.
        if view != 'camera':
            draw_camera_frustum(ax, camera, scale=2.0)

        # Frame the scene according to the selected view mode.
        # Plot coordinates: x=X, y=Z(Depth), z=Y(Height)
        if view == 'overall':
            positions = []
            _collect_positions(root, positions)
            _fit_overall(ax, positions, camera)
        elif view == 'camera' and camera.frames:
            _fit_camera(ax, camera)
        else:
            _fit_focused(ax, waist, camera)

        ax.set_title(f"Frame: {int(frame)}  [{view}]")
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
            
        return update_plot(current_frame, leg_ik)

    # Slider setup
    ax_slider = plt.axes([0.25, 0.1, 0.65, 0.03], facecolor='lightgoldenrodyellow')
    slider = Slider(ax_slider, '', 0, total_frames - 1, valinit=0, valfmt='%0.0f')
    
    def on_slider_change(val):
        anim_state['frame'] = int(val)
        update_plot(anim_state['frame'], leg_ik)
        
    slider.on_changed(on_slider_change)

    # Play/Pause Button
    ax_play = plt.axes([0.1, 0.1, 0.1, 0.04])
    btn_play = Button(ax_play, 'Pause', color='lightgoldenrodyellow', hovercolor='0.975')

    def toggle_play(event):
        anim_state['running'] = not anim_state['running']
        btn_play.label.set_text('Pause' if anim_state['running'] else 'Play')

    btn_play.on_clicked(toggle_play)

    # View mode selector
    ax_view = plt.axes([0.015, 0.45, 0.14, 0.16], facecolor='lightgoldenrodyellow')
    view_radio = RadioButtons(ax_view, ('Focused', 'Overall', 'Camera'), active=0)
    view_labels = {'Focused': 'focused', 'Overall': 'overall', 'Camera': 'camera'}

    def on_view_change(label):
        anim_state['view'] = view_labels[label]
        update_plot(anim_state['frame'], leg_ik)
        fig.canvas.draw_idle()

    view_radio.on_clicked(on_view_change)

    # Use a generator or infinite loop for frames so we control flow manually
    ani = animation.FuncAnimation(fig, update, frames=None, interval=1000/fps, cache_frame_data=False)
    plt.show()

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        preview_motion(sys.argv[1])
