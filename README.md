# vmd2png

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

---

## vmd2png 简介（中文）

vmd2png 是一个用于处理 MikuMikuDance（MMD）VMD 动作文件的 Python 实用库。

## 图像格式中的动作数据？

![](examples/conqueror.png)

上面的示例图片是一个长度约 4 分钟的 VMD 动作被转换后的结果。  
我们采用了一种在视觉上保留动作特征的同时最大化存储效率的编码方式。  
其结果是，你可以轻松识别动作中的整体模式，甚至可以通过观察平滑的过渡与像素化的细节来快速评估动作质量。

把动作数据表示为图像格式乍一看有些反直觉，但它有几个显著优势：

1. **可视化**：PNG 图像可以直接查看，无需专门软件即可快速检查动作数据。
2. **存储**：PNG 是广泛支持的格式，在使用 16 位通道以保证精度的前提下，依然能够高效压缩存储动作数据。以上示例中，原始 VMD 文件大小约 17MB，而转换后的 PNG 只有约 1.2MB。
3. **数据分析**：将动作数据存储为便于加载到机器学习框架（如 NumPy）中的格式，能够方便地用于训练 AI 模型，实现诸如动作预测、动作生成等用途。

## 功能特性

- **解析 VMD**：读取 VMD 文件并提取骨骼和摄像机动作数据。
- **PNG 格式**：使用 16 位 PNG 图像来表示动作数据，同时兼顾可视化与存储。
- **NPY 格式**：将动作数据存储为 NumPy 的 `.npy` 格式，便于在 Python 中处理和用于 AI 训练。
- **格式转换**：在 VMD、PNG 和 NPY 格式之间相互转换。
- **合并**：将分离的角色（Actor）与摄像机（Camera）VMD 文件合并为一个 VMD。
- **预览**：无需额外软件即可预览角色和摄像机的动作。

## 安装

```bash
git clone https://github.com/alloystorm/vmd2png.git
cd vmd2png
pip install .
```

## 命令行用法（CLI Usage）

本项目提供 `vmd2png` 命令行工具。

### 预览

在 3D 界面中预览 VMD、PNG 或 NPY 文件：

```bash
vmd2png preview path/to/motion.vmd
# 启用腿部 IK
vmd2png preview path/to/motion.vmd --ik
# 使用单独 VMD 文件中的摄像机动作
vmd2png preview path/to/actor.vmd --camera path/to/camera.vmd
# 预览已转换的 PNG/NPY 文件
vmd2png preview path/to/motion.png
vmd2png preview path/to/motion.npy
```

### 转换

将 VMD 转换为 PNG/NPY 格式：

```bash
vmd2png convert path/to/file.vmd -t png
vmd2png convert path/to/file.vmd -t npy
```

将角色与摄像机的动作从两个 VMD 中合并：

```bash
vmd2png convert actor.vmd --camera camera.vmd -o combined.vmd -t vmd
```

将 PNG/NPY 转回 VMD：

```bash
vmd2png convert path/to/file.png -t vmd
vmd2png convert path/to/file.npy -t vmd
```

## 注意事项

- 工具假定模型使用标准的 MMD 骨骼结构。详情可参考 [skeleton.py](src/vmd2png/skeleton.py)。任何自定义骨骼的动作在转换后都会丢失。
- 当前 PNG/NPY 格式尚不支持表情与变形（morph），因此所有表情和变形数据在转换后都会丢失。
- 为保证动作数据精度，我们使用 16 位 PNG 格式。如果你对图像进行编辑，或者通过邮件 / 即时通讯软件发送，请确保图像不会被重新压缩为 8 位格式，否则动作数据会被破坏。
- 在 PNG/NPY 格式中，动作是逐帧存储的（每一帧都会被展开存储）。因此，当你将 PNG/NPY 再转换回 VMD 时，原始的关键帧信息将丢失，并且生成的 VMD 文件体积通常会比原始文件更大。

## 后续计划

- 支持 BVH 动捕数据。
- 支持标准表情变形（morph）。

## 许可证

本项目基于 MIT 许可证开源，详情请参见 [LICENSE](LICENSE) 文件。

## vmd2png 概要（日本語）

vmd2png は、MikuMikuDance（MMD）の VMD モーションファイルを処理するための Python 製ユーティリティライブラリです。

## 画像形式でのモーション？

![](examples/conqueror.png)

上の画像は、約 4 分間の VMD モーションを変換した例です。  
ここでは、モーションの視覚的な特徴を保ちながら、できるだけ効率よく保存できるようなエンコード方式を採用しています。  
その結果、モーションのパターンを一目で把握できるだけでなく、スムーズな遷移か、ギザギザしたピクセル状の部分が多いかを見ることで、モーションの品質をざっくり評価することもできます。

モーションデータを画像として表現するという発想は、一見すると少し奇妙に思えるかもしれませんが、次のような利点があります。

1. **可視化**: PNG 画像として簡単に表示できるため、専用ソフトを使わずにモーションデータを素早く確認できます。
2. **保存形式**: PNG は広くサポートされているフォーマットであり、16bit チャンネルを使って十分な精度を保ちながら、比較的コンパクトにデータを保存できます。上の例では、VMD 形式で約 17MB のモーションが、PNG 形式では約 1.2MB まで小さくなっています。
3. **データ解析**: モーションを、NumPy などの機械学習フレームワークに直接読み込める形式で保存できるため、モーション予測やモーション生成など、AI モデルの学習に活用しやすくなります。

## 機能

- **VMD のパース**: VMD ファイルを読み込み、ボーンおよびカメラのモーションデータを抽出します。
- **PNG 形式**: モーションデータを 16bit PNG 画像として表現し、可視化と保存の両方に利用できます。
- **NPY 形式**: モーションデータを NumPy の `.npy` 形式で保存し、Python からの操作や機械学習用途に使いやすくします。
- **変換**: VMD / PNG / NPY の各形式のあいだで相互に変換できます。
- **マージ**: 役者（Actor）用 VMD とカメラ用 VMD を 1 つの VMD に結合できます。
- **プレビュー**: 追加のソフトウェアなしで、役者およびカメラのモーションをプレビューできます。

## インストール

```bash
git clone https://github.com/alloystorm/vmd2png.git
cd vmd2png
pip install .
```

## CLI の使い方

このパッケージは `vmd2png` というコマンドラインツールを提供します。

### プレビュー

VMD / PNG / NPY ファイルを 3D でプレビューします。

```bash
vmd2png preview path/to/motion.vmd
# 足 IK を有効にする
vmd2png preview path/to/motion.vmd --ik
# 別ファイルのカメラモーションを使用する
vmd2png preview path/to/actor.vmd --camera path/to/camera.vmd
# 変換済み PNG / NPY をプレビューする
vmd2png preview path/to/motion.png
vmd2png preview path/to/motion.npy
```

### 変換

VMD を PNG / NPY 形式に変換します。

```bash
vmd2png convert path/to/file.vmd -t png
vmd2png convert path/to/file.vmd -t npy
```

役者とカメラのモーションを、別々の VMD から 1 つの VMD に統合します。

```bash
vmd2png convert actor.vmd --camera camera.vmd -o combined.vmd -t vmd
```

PNG / NPY を VMD に戻します。

```bash
vmd2png convert path/to/file.png -t vmd
vmd2png convert path/to/file.npy -t vmd
```

## 注意点

- 本ツールは標準的な MMD のボーン構造を前提としています。詳細は [skeleton.py](src/vmd2png/skeleton.py) を参照してください。カスタムボーンに対するモーションは、変換後には失われます。
- 表情・モーフは、現時点では PNG / NPY 形式に対応していないため、変換時にすべて失われます。
- モーションデータの精度を確保するために 16bit PNG を使用しています。画像を編集したり、メールやメッセンジャーで送信する際に 8bit に再圧縮されると、モーションデータが破損するので注意してください。
- PNG / NPY 形式では、すべてのフレームを 1 フレームずつ展開して保存します。そのため、PNG / NPY から VMD に再変換すると元のキー情報は失われ、生成される VMD ファイルは元のファイルよりも大きくなる場合があります。

## 今後の予定

- モーションキャプチャデータ用の BVH 対応。
- 標準的な表情モーフへの対応。

## ライセンス

このプロジェクトは MIT ライセンスのもとで公開されています。詳細は [LICENSE](LICENSE) ファイルを参照してください。