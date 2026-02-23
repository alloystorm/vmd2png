import sys
import os

# Add src to path to ensure we import from the local source
sys.path.insert(0, os.path.abspath('src'))

from vmd2png.converter import load_from_png_16bit, read_png_metadata
from vmd2png.skeleton import build_standard_skeleton

def main():
    png_path = "test_output/conqueror_test.png"
    print(f"Loading {png_path}...")
    
    # Read and print metadata directly to verify it works
    try:
        metadata = read_png_metadata(png_path)
        print("\n--- PNG Metadata ---")
        for key, value in metadata.items():
            # Truncate long values for display
            display_val = value if len(value) < 100 else value[:97] + "..."
            print(f"{key}: {display_val}")
        print("--------------------\n")
    except Exception as e:
        print(f"Error reading metadata: {e}")

    # Calculate expected stride
    root_skel, _ = build_standard_skeleton()
    actor_bones_list = root_skel.export_bones()
    stride_actor = 4 + len(actor_bones_list) * 4
    stride_cam = 8
    stride = stride_actor + stride_cam
    
    try:
        data = load_from_png_16bit(png_path, min_val=-1.0, max_val=1.0, stride=stride)
        print(f"Successfully loaded data with shape: {data.shape}")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()
