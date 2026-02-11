import argparse
import sys
import os
from .preview import preview_motion
from .converter import export_vmd_to_files, convert_motion_to_vmd

def main():
    parser = argparse.ArgumentParser(description="vmd2png: VMD motion utility")
    subparsers = parser.add_subparsers(dest="command", help="Sub-command to run")
    
    # Preview Command
    parser_preview = subparsers.add_parser("preview", help="Preview motion file (.vmd, .png, .npy)")
    parser_preview.add_argument("path", help="Path to motion file")
    parser_preview.add_argument("--fps", type=int, default=30, help="Playback FPS")
    parser_preview.add_argument("--ik", action="store_true", help="Use leg IK")
    parser_preview.add_argument("--camera-motion", help="Path to camera VMD file to overlay")
    
    # Convert Command
    parser_convert = subparsers.add_parser("convert", help="Convert between VMD and PNG/NPY")
    parser_convert.add_argument("input", help="Input file path")
    parser_convert.add_argument("-o", "--output", help="Output path or directory")
    # We can infer direction from extension, but flags help
    parser_convert.add_argument("--npy", action="store_true", help="Export NPY (when input is VMD)")
    parser_convert.add_argument("--png", action="store_true", help="Export PNG (when input is VMD)")
    
    args = parser.parse_args()
    
    if args.command == "preview":
        if not os.path.exists(args.path):
            print(f"Error: File not found: {args.path}")
            sys.exit(1)
        preview_motion(args.path, fps=args.fps, camera_vmd_path=args.camera_motion)
        
    elif args.command == "convert":
        if not os.path.exists(args.input):
            print(f"Error: Input file not found: {args.input}")
            sys.exit(1)
            
        ext = os.path.splitext(args.input)[1].lower()
        
        if ext == ".vmd":
            # Export to files
            output_dir = args.output if args.output else os.path.dirname(args.input)
            if not output_dir: output_dir = "."
            
            print(f"Converting VMD to PNG/NPY in {output_dir}...")
            # Ideally export_vmd_to_files should allow filtering NPY/PNG?
            # Currently it exports both.
            # We'll just run it.
            success = export_vmd_to_files(args.input, output_dir, leg_ik=args.ik)
            if success:
                print("Done.")
            else:
                print("Conversion failed.")
                sys.exit(1)
                
        elif ext in [".png", ".npy"]:
            # Convert to VMD
            output_path = args.output
            if not output_path:
                output_path = os.path.splitext(args.input)[0] + ".vmd"
                
            print(f"Converting {ext} to VMD: {output_path}...")
            success = convert_motion_to_vmd(args.input, output_path)
            if success:
                print("Done.")
            else:
                print("Conversion failed.")
                sys.exit(1)
        else:
            print(f"Unsupported extension: {ext}")
            sys.exit(1)
            
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
