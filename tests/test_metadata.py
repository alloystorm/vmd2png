import os
from PIL import Image
import sys

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))

from vmd2png.converter import export_vmd_to_files

def main():
    vmd_path = "data/conqueror.vmd"
    output_path = "test_output/conqueror_test.png"
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Remove existing file if it exists
    if os.path.exists(output_path):
        os.remove(output_path)
        
    print(f"Exporting {vmd_path} to {output_path}...")
    success = export_vmd_to_files(vmd_path, output_path, out_type='png')
    
    if not success:
        print("Export failed.")
        return
        
    print("Export successful. Reading metadata...")
    
    # Read metadata using PIL
    try:
        with Image.open(output_path) as img:
            img.load() # Ensure image is loaded
            metadata = img.info
            print("\n--- PNG Metadata ---")
            for key, value in metadata.items():
                print(f"Key: {key}")
                print(f"Value: {value}")
                print("-" * 20)
            print("--------------------")
    except Exception as e:
        print(f"Error reading metadata: {e}")

if __name__ == "__main__":
    main()
