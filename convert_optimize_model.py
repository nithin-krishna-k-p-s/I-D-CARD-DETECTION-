# convert_and_optimize_model.py
import torch
from ultralytics import YOLO
import onnx
from onnxruntime.quantization import quantize_dynamic, QuantType
import onnxruntime as ort
import numpy as np
import os

def export_to_onnx():
    print("Loading YOLO model...")
    model = YOLO('best.pt')
    
    print("Exporting to ONNX format...")
    # Export with optimizations for web
    success = model.export(
        format='onnx',
        imgsz=640,
        half=False,  # Keep as float32 for better web compatibility
        optimize=True,
        simplify=True,
        opset=12,  # Use opset 12 for better ONNX Runtime Web support
        dynamic=False,
        nms=False  # We'll handle NMS in JavaScript
    )
    
    if not success:
        raise Exception("Failed to export model to ONNX")
    
    print("ONNX export complete")

def quantize_model():
    print("Quantizing model...")
    
    input_model_path = 'best.onnx'
    output_model_path = 'best_quantized.onnx'
    
    # Use dynamic quantization
    quantize_dynamic(
        model_input=input_model_path,
        model_output=output_model_path,
        weight_type=QuantType.QUInt8,
        optimize_model=True
    )
    
    # Check sizes
    original_size = os.path.getsize(input_model_path) / (1024 * 1024)
    quantized_size = os.path.getsize(output_model_path) / (1024 * 1024)
    
    print(f"Original ONNX size: {original_size:.2f} MB")
    print(f"Quantized ONNX size: {quantized_size:.2f} MB")
    print(f"Size reduction: {((original_size - quantized_size) / original_size * 100):.1f}%")
    
    return output_model_path

def verify_model(model_path):
    print("Verifying quantized model...")
    try:
        session = ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])
        
        # Create dummy input
        input_name = session.get_inputs()[0].name
        input_shape = session.get_inputs()[0].shape
        dummy_input = np.random.randn(*input_shape).astype(np.float32)
        
        # Run inference
        outputs = session.run(None, {input_name: dummy_input})
        print("Model verification successful!")
        print(f"Input shape: {input_shape}")
        print(f"Output names: {[output.name for output in session.get_outputs()]}")
        
    except Exception as e:
        print(f"Model verification failed: {e}")
        return False
    
    return True

def main():
    print("=" * 50)
    print("YOLO Model Optimization for Web Deployment")
    print("=" * 50)
    
    # Step 1: Export to ONNX
    export_to_onnx()
    
    # Step 2: Quantize
    quantized_model = quantize_model()
    
    # Step 3: Verify
    if verify_model(quantized_model):
        print("\n✅ Model optimization complete!")
        print(f"Optimized model saved as: {quantized_model}")
        print("\nCopy this file to your GitHub repository root directory.")
    else:
        print("\n❌ Model optimization failed. Please check the errors above.")

if __name__ == "__main__":
    main()
