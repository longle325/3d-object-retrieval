import os
import argparse
from tqdm import tqdm
import bpy

def main(root_dir, output_dir):
    target_filename = "raw_model.obj"

    for subdir in tqdm(os.listdir(root_dir)):
        full_path = os.path.join(root_dir, subdir, target_filename)
        if os.path.isfile(full_path):
            original_obj_path = full_path
            converted_glb_path = os.path.join(output_dir, subdir, "raw_model.glb")
            os.makedirs(os.path.dirname(converted_glb_path), exist_ok=True)

            # Clear all objects
            bpy.ops.object.select_all(action='SELECT')
            bpy.ops.object.delete(use_global=False)

            # Import the OBJ file
            bpy.ops.wm.obj_import(filepath=original_obj_path, use_split_groups=True)

            # Export to GLB
            bpy.ops.export_scene.gltf(filepath=converted_glb_path, export_format='GLB')

    print("Result saved!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert OBJ files to GLB format using Blender.")
    parser.add_argument("--root_dir", type=str, required=True, help="Path to the root directory containing objects.")
    parser.add_argument("--output_dir", type=str, required=True, help="Path to save converted GLB files.")
    args = parser.parse_args()

    main(args.root_dir, args.output_dir)
