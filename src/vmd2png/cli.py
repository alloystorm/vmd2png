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
    parser_preview.add_argument("--camera", help="Path of camera VMD file to be merged")
    parser_preview.add_argument("--fps", type=int, default=30, help="Playback FPS")
    parser_preview.add_argument("--ik", action="store_true", help="Use leg IK")
    
    # Convert Command
    parser_convert = subparsers.add_parser("convert", help="Convert between VMD and PNG/NPY")
    parser_convert.add_argument("input", help="Input file path")
    parser_convert.add_argument("--camera", help="Path of camera VMD file to be merged (for VMD input)")
    parser_convert.add_argument("-o", "--output", help="Output path or directory")
    parser_convert.add_argument("-t", "--type", choices=['png', 'npy', 'vmd'], default='png', help="Output format type")
    
    args = parser.parse_args()
    
    if args.command == "preview":
        if not os.path.exists(args.path):
            print(f"Error: File not found: {args.path}")
            sys.exit(1)
        preview_motion(args.path, fps=args.fps, camera_vmd_path=args.camera, leg_ik=args.ik)
        
    elif args.command == "convert":
        if not os.path.exists(args.input):
            print(f"Error: Input file not found: {args.input}")
            sys.exit(1)
            
        ext = os.path.splitext(args.input)[1].lower()
        
        # Determine output path
        output_path = args.output
        if not output_path:
            base = os.path.splitext(os.path.basename(args.input))[0]
            output_path = os.path.join(os.path.dirname(args.input) or '.', f"{base}.{args.type}")
        elif os.path.isdir(output_path) or (not os.path.splitext(output_path)[1]):
             # Directory or prefix
             os.makedirs(output_path, exist_ok=True)
             base = os.path.splitext(os.path.basename(args.input))[0]
             output_path = os.path.join(output_path, f"{base}.{args.type}")

        print(f"Converting {args.input} to {output_path} ({args.type})...")

        if ext == ".vmd" or args.type == "vmd":
            # For VMD input: Supports to NPY/PNG/VMD
            # For VMD output: Supports from VMD/NPY/PNG (via load_motion_dict in export function)
            success = export_vmd_to_files(args.input, output_path, out_type=args.type, camera_vmd_path=args.camera)
        else:
            print("Conversion from PNG/NPY to PNG/NPY is not supported yet.")
            success = False

        if success:
            print("Done.")
        else:
            print("Conversion failed.")
            sys.exit(1)
            
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
