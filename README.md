# vmd2png

A Python utility library for processing MikuMikuDance (MMD) VMD motion files.

## Features

- **Parse VMD**: Read VMD files and extract bone and camera motion data.
- **PNG Format**: Represent motion data in 16-bit PNG format for both visualization and storage.
- **NPY Format**: Store motion data in NumPy's `.npy` format for easy manipulation in Python and AI training.
- **Convert**: Convert between VMD, PNG, and NPY formats.
- **Merge**: Combine separate Actor and Camera VMD files into a single VMD.
- **Preview**: Visualize actor and camera motion without any additional software.

## Installation

```bash
pip install .
```

## CLI Usage

The package provides a `vmd2png` command line tool.

### Preview
Preview a VMD, PNG, or NPY file in 3D:
```bash
vmd2png preview path/to/motion.vmd
# With leg IK enabled
vmd2png preview path/to/motion.vmd --ik
# With separate camera motion overlay
vmd2png preview path/to/actor.vmd --camera path/to/camera.vmd
```

### Convert
Convert VMD to extracted files (PNG/NPY):
```bash
vmd2png convert path/to/file.vmd -t png
vmd2png convert path/to/file.vmd -t npy
```

Merge Actor and Camera VMDs:
```bash
vmd2png convert actor.vmd --camera camera.vmd -o combined.vmd -t vmd
```

Convert PNG/NPY back to VMD:
```bash
vmd2png convert path/to/file.png -t vmd
```

## Caveats
- The tool expects the standard MMD bone structure. Please see [skeleton.py](src/vmd2png/skeleton.py) for details. Any custom bone motions will be lost after conversion.
- All facial expressions and morphs will be lost after conversion, as they are not supported in the PNG/NPY format yet.
- We use 16-bit PNG format to achieve necessary precision for motion data. If you edit the image and save as 8-bit, the motion data will be corrupted. 
- The motion is stored frame by frame for every frame in the PNG/NPY format. When you convert PNG/NPY back to VMD, the original keyframe information will be lost, and the file size will be larger than the original.


## What's Next
- BVH support for motion capture data.
- Support for standard facial morphs.


## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.