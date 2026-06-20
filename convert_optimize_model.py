# convert_optimize_model.py
from ultralytics import YOLO
import onnx
import onnxruntime as ort
from onnxruntime.quantization import quantize_dynamic, QuantType
import os

def convert_and_optimize():
    # Load your trained model
    model = YOLO('best.pt')
    
    # Export to ONNX with optimizations
    model.export(
        format='onnx',
        imgsz=640,
        half=True,  # FP16 precision
        optimize=True,
        int8=True,  # Further quantization
        dynamic=False,
        simplify=True
    )
    
    # Quantize the model to reduce size further
    print("Quantizing model...")
    quantize_dynamic(
        model_input='best.onnx',
        model_output='best_quantized.onnx',
        weight_type=QuantType.QInt8
    )
    
    # Check sizes
    original_size = os.path.getsize('best.onnx') / (1024 * 1024)
    quantized_size = os.path.getsize('best_quantized.onnx') / (1024 * 1024)
    
    print(f"Original ONNX size: {original_size:.2f} MB")
    print(f"Quantized ONNX size: {quantized_size:.2f} MB")
    
    return 'best_quantized.onnx' if os.path.exists('best_quantized.onnx') else 'best.onnx'

if __name__ == "__main__":
    final_model = convert_and_optimize()
    print(f"Final model ready: {final_model}")
