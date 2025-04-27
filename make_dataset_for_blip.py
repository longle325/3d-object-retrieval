import os
import shutil
import argparse
from tqdm import tqdm

def main(src_root, dst_root):
    for dirpath, dirnames, filenames in os.walk(src_root):
        relative_path = os.path.relpath(dirpath, src_root)
        dst_dir = os.path.join(dst_root, relative_path)
        os.makedirs(dst_dir, exist_ok=True)

        for filename in filenames:
            if filename in ["image.jpg", "text.txt"]:
                src_file = os.path.join(dirpath, filename)
                dst_file = os.path.join(dst_dir, filename)
                print(f"Processing {src_file} -> {dst_file}")
                shutil.copy2(src_file, dst_file)

    print("The training dataset is completed.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Copy specific files to new structure")
    parser.add_argument("--src_root", type=str, help="Public dataset folder")
    parser.add_argument("--dst_root", type=str, help="Root directory containing all trainning data for blip")
    args = parser.parse_args()

    main(args.src_root, args.dst_root)  
