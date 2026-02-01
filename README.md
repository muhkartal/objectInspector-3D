# Object Inspector 3D

![License](https://img.shields.io/badge/License-MIT-green.svg)
![Python](https://img.shields.io/badge/Python-3.9%2B-blue.svg)
![Pygame](https://img.shields.io/badge/Pygame-2.5%2B-orange.svg)
![Code%20Style](https://img.shields.io/badge/code%20style-black-000000.svg)

A pygame-based 3D object inspection tool with multiple visualization modes, exploded-view assemblies, and optional ML scaffolding for analysis workflows.

Overview • Features • Quick Start • Controls • Model Import • ML Models • Configuration • Project Layout • Troubleshooting • Contributing • License

## Overview
Object Inspector 3D combines a lightweight real-time renderer with interactive camera controls, multiple visualization modes, and procedural demo assemblies. It is designed as a practical sandbox for exploring 3D models, experimenting with rendering effects, and wiring in ML analysis pipelines when needed.

## Features
- Real-time 3D rendering with orbit, pan, and zoom controls.
- Visualization modes: solid, wireframe, points, and exploded with smooth transitions.
- Procedural demo assemblies (engine, gearbox, watch, plus complex/architectural sets).
- Post-processing effects (glow, vignette, scanlines presets).
- Import external models: .obj, .gltf, .glb.
- Optional input sources for ML pipelines (synthetic renderer or webcam).
- ONNX-based ML modules for classification, depth, pose, and feature detection.

## Quick Start

### Prerequisites
- Python 3.9+
- pip

### Install

Full install (includes optional ML, webcam, and GLTF support):

```bash
pip install -r requirements.txt
```

Editable install with extras:

```bash
pip install -e .

# Optional extras
pip install -e .[cv]
pip install -e .[ml]
pip install -e .[full]
```

### Run

```bash
python -m src
```

If installed as a package:

```bash
object-inspector-3d
```

## Controls

### Mouse
- Left drag: orbit camera
- Right or middle drag: pan
- Scroll: zoom

### Keyboard

| Key | Action |
| --- | --- |
| 1-7 | Select primitive shape |
| Tab | Cycle visualization mode |
| E | Toggle exploded view |
| A | Cycle all assemblies |
| F1-F12 | Load demo assemblies |
| L | Load a model file (.obj, .gltf, .glb) |
| C | Toggle webcam vs synthetic input |
| G | Toggle glow effect |
| V | Toggle vignette effect |
| Space | Toggle auto-rotate |
| R | Reset camera |
| H | Toggle help overlay |
| Esc | Quit |

## Model Import
- Press **L** to open a file dialog and load a model.
- Supported formats: **.obj**, **.gltf**, **.glb**.
- OBJ groups become exploded parts; GLTF meshes become parts.
- GLTF/GLB loading requires `pygltflib`.

## ML Models (Optional)
ML components live in `src/ml` and can be wired into your own pipeline.

Expected model files (place in `models/pretrained`):
- `mobilenet_v2.onnx` (classifier)
- `pose_estimation.onnx` (pose)
- `midas_small.onnx` (depth)
- `feature_detector.onnx` (features)

Enable ONNX Runtime:

```bash
pip install onnxruntime
```

## Configuration
Tweak defaults in `config/settings.py`:
- Window size, FPS, and performance settings
- Camera limits and sensitivity
- Color palettes, lighting, and post-processing presets
- Visualization modes and transitions
- Webcam and ML settings

## Project Layout

```
config/                 App settings
models/pretrained/      ONNX models (optional)
src/
  main.py               App entry point
  geometry/             Primitives and procedural assemblies
  rendering/            Camera, projector, and renderer
  visualization/        View modes and exploded visualizer
  effects/              Post-processing pipeline
  ui/                   Panels, overlays, sliders
  input/                Synthetic + webcam input sources
  loaders/              OBJ/GLTF model loaders
  ml/                   ONNX model wrappers and helpers
```

## Troubleshooting
- Webcam not available: install `opencv-python` and verify your device is detected.
- GLTF/GLB load fails: install `pygltflib`.
- File dialog does not open: make sure `tkinter` is installed for your Python build.
- ML outputs missing: install `onnxruntime` and add models under `models/pretrained`.

## Contributing
Issues and pull requests are welcome. Please include clear repro steps and screenshots when relevant.

## License
MIT. See `LICENSE` for details.
