import os
import onnx
from src.utils.tensorrt_utils import onnx_to_tensorrt

# --- Conversion Settings ---

# ONNX models to convert
ONNX_MODELS = {
    "antispoof_2.7": "models/antispoof/2.7_80x80_MiniFASNetV2.onnx",
    "antispoof_4.0": "models/antispoof/4_0_0_80x80_MiniFASNetV1SE.onnx",
    "mobilefacenet": "models/mobilefacenet/MobileFaceNet.onnx",
}

# Output directory for TensorRT engines
ENGINE_DIR = "models/tensorrt_engines"

# --- Conversion Script ---

def main():
    if not os.path.exists(ENGINE_DIR):
        os.makedirs(ENGINE_DIR)

    for name, onnx_path in ONNX_MODELS.items():
        if not os.path.exists(onnx_path):
            print(f"ONNX model not found: {onnx_path}")
            continue

        engine_path = os.path.join(ENGINE_DIR, f"{name}.engine")
        print(f"Converting {onnx_path} to {engine_path}...")

        onnx_to_tensorrt(onnx_path, engine_path)

        print(f"Successfully converted {name} to a TensorRT engine.")

if __name__ == "__main__":
    main()
