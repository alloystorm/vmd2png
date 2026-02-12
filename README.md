# vmd2png

[English](README.md) | [中文](README.zh-CN.md) | [日本語](README.jp.md)

A Python utility library for processing MikuMikuDance (MMD) VMD motion files and exports as PNG image or NPY array. You can also preview the motion in 3D with the built-in viewer.

## Motion in Image Format?

![](examples/conqueror.png)

The above example image is a converted 4 minute long VMD motion. The encoding we chose ensures max storage efficiency while preserving the characteristics of the motion visually. So you can easily identify the patterns of the motion and even evaluate the quality of the motion at a glance by checking for organic smooth transitions vs pixelated details. 

The idea of representing motion data in image format may seem unconventional at first, but it offers several advantages:
1. **Visualization**: PNG images can be easily visualized, allowing users to quickly inspect motion data without needing specialized software.
2. **Storage**: PNG is a widely supported format that can efficiently store motion data in a compact form, especially when using 16-bit channels for precision. In the above example the file size of the motion is down from 17mb in the VMD format to 1.2mb in the PNG format.
3. **Data Analysis**: Storing motion data in a format that can be easily loaded into machine learning frameworks (like NumPy) facilitates training AI models on motion data, enabling possibilities such as motion prediction and generation.


## Features

- **Parse VMD**: Read VMD files and extract bone and camera motion data.
- **PNG Format**: Represent motion data in 16-bit PNG format for both visualization and storage.
- **NPY Format**: Store motion data in NumPy's `.npy` format for easy manipulation in Python and AI training.
- **Convert**: Convert between VMD, PNG, and NPY formats.
- **Merge**: Combine separate Actor and Camera VMD files into a single VMD.
- **Preview**: Visualize actor and camera motion without any additional software.

## Installation

```bash
git clone https://github.com/alloystorm/vmd2png.git
cd vmd2png
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
# With camera motion from a seperate VMD file
vmd2png preview path/to/actor.vmd --camera path/to/camera.vmd
# Preview converted PNG/NPY files
vmd2png preview path/to/motion.png
vmd2png preview path/to/motion.npy
```

### Convert
Convert VMD to PNG/NPY format:
```bash
vmd2png convert path/to/file.vmd -t png
vmd2png convert path/to/file.vmd -t npy
```

Merge Actor and Camera motion from seperate VMDs:
```bash
vmd2png convert actor.vmd --camera camera.vmd -o combined.vmd -t vmd
```

Convert PNG/NPY back to VMD:
```bash
vmd2png convert path/to/file.png -t vmd
vmd2png convert path/to/file.npy -t vmd
```

## Caveats
- The tool expects the standard MMD bone structure. Please see [skeleton.py](src/vmd2png/skeleton.py) for details. Any custom bone motions will be lost after conversion.
- All facial expressions and morphs will be lost after conversion, as they are not supported in the PNG/NPY format yet.
- We use 16-bit PNG format to achieve necessary precision for the motion data. If you edit the image or send it through email or messaging, make sure it is not recompressed to 8bit format or the motion data will be corrupted.
- The motion is stored frame by frame for every frame in the PNG/NPY format. When you convert PNG/NPY back to VMD, the original keyframe information will be lost, and the file size will be larger than the original.


## What's Next?
- BVH support for motion capture data.
- Support for standard facial morphs.


## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.