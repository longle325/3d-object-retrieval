import os
import argparse
import trimesh
import pyrender
import numpy as np
import imageio
from PIL import Image
from tqdm import tqdm

# Set up environment variables for rendering
os.environ['PYOPENGL_PLATFORM'] = 'egl'
os.environ['LD_PRELOAD'] = os.path.join(os.environ['CONDA_PREFIX'], 'lib', 'libstdc++.so.6')

views = []
for angle in np.linspace(0, 360, 20):
    rad = np.deg2rad(angle)
    views.append({'position': [np.cos(rad) * 3, np.sin(rad) * 3, 0], 'target': [0, 0, 0]})

def remove_file(file_path: str):
    if os.path.exists(file_path):
        os.remove(file_path)
    return

def look_at(eye, target, up=[0, 0, 1]):
    """Generate a look-at view matrix."""
    f = np.array(target) - np.array(eye)
    f = f / np.linalg.norm(f)
    u = np.array(up)
    s = np.cross(f, u)
    s = s / np.linalg.norm(s)
    u = np.cross(s, f)
    m = np.eye(4)
    m[:3, 0] = s
    m[:3, 1] = u
    m[:3, 2] = -f
    m[:3, 3] = eye
    return m

def render_glb_to_components(file_path_glb: str, data_gen_output_dir: str):
    mesh = trimesh.load(file_path_glb, force='mesh')

    width = 1024
    height = 1024
    print(f"Using dimensions: width = {width}, height = {height}")

    mesh.apply_translation(-mesh.centroid)
    scale_factor = 2 / max(mesh.extents)
    mesh.apply_scale(scale_factor)

    rotation_matrix = trimesh.transformations.rotation_matrix(np.radians(90), [1, 0, 0])
    mesh.apply_transform(rotation_matrix)

    r = pyrender.OffscreenRenderer(viewport_width=width, viewport_height=height)
    scene = pyrender.Scene(bg_color=[0, 0, 0, 0], ambient_light=[0.3, 0.3, 0.3])

    if isinstance(mesh, trimesh.Trimesh):
        try:
            mesh = pyrender.Mesh.from_trimesh(mesh)
            scene.add(mesh)
        except Exception as e:
            print(f"Error adding mesh: {e}")
            return

    light = pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=2.0)
    scene.add(light, pose=np.eye(4))

    uid = os.path.basename(os.path.dirname(file_path_glb))
    object_output_dir = os.path.join(data_gen_output_dir, uid)
    os.makedirs(object_output_dir, exist_ok=True)

    for view_idx, view in enumerate(views):
        camera_pose = look_at(view['position'], view['target'])
        camera = pyrender.IntrinsicsCamera(fx=900, fy=900, cx=width/2, cy=height/2, znear=0.1, zfar=100)
        scene_camera = pyrender.Node(camera=camera, matrix=camera_pose)
        scene.add_node(scene_camera)

        color, depth = r.render(scene)

        alpha_channel = (depth != 0).astype(np.uint8) * 255
        color = np.dstack((color, alpha_channel))

        center_x = width // 2
        start_x = max(center_x - height // 2, 0)
        end_x = start_x + height
        cropped_color = color[:, start_x:end_x]

        cropped_color = Image.fromarray(cropped_color, mode='RGBA')
        cropped_color = cropped_color.resize((1024, 1024), Image.LANCZOS)
        cropped_color = np.array(cropped_color)

        image_path = os.path.join(object_output_dir, f'{view_idx:03d}.png')
        imageio.imwrite(image_path, cropped_color)

        scene.remove_node(scene_camera)

    r.delete()

def format_to_six_digits(number: int) -> str:
    return f"{number:06d}"

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Render GLB files to images from multiple views.")
    parser.add_argument("--root_dir", type=str, required=True, help="Root directory containing GLB files.")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory for rendered images.")
    args = parser.parse_args()

    root_dir = args.root_dir
    output_dir = args.output_dir
    target_filename = "raw_model.glb"

    for subdir in tqdm(os.listdir(root_dir)):
        full_path = os.path.join(root_dir, subdir, target_filename)
        if os.path.isfile(full_path):
            print(f"Rendering {full_path}")
            render_glb_to_components(full_path, output_dir)

    print("Result saved!")
