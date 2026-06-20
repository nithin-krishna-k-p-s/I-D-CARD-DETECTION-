# ID Card Detection System

Real-time ID card detection system using YOLO and ONNX Runtime Web.

## Features
- Real-time ID card detection
- Webcam support
- Image and video upload
- Alarm system for missing ID cards
- Professional web interface

## Setup

1. Place your optimized ONNX model (`best_quantized.onnx`) in the root directory
2. Enable GitHub Pages in repository settings
3. Access your application at `https://[username].github.io/[repository]`

## Model Optimization

To optimize your model size:
```bash
python convert_optimize_model.py
