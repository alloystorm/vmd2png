# vmd2png

A Python utility library for processing MikuMikuDance (MMD) VMD motion files.

## Features

- **Parse VMD**: Read VMD files and extract bone and camera motion data.
- **Export**: detailed motion data export to `.npy` and 16-bit PNG formats.
- **Convert**: Import `.npy` or `.png` data back to VMD format.
- **Preview**: Visualize motion data.
- **Standalone**: decoupled from the main motion encoder project.

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
vmd2png preview path/to/motion.png --mode character
```

### Convert
Convert VMD to extracted files (PNG/NPY):
```bash
vmd2png convert path/to/file.vmd -o output_dir
```

Convert PNG/NPY back to VMD:
```bash
vmd2png convert path/to/file.png -o output.vmd --mode character
```

## Python Usage
