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

## Python Usage
