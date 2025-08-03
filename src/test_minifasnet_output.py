import onnxruntime
import numpy as np
import torch

def test_onnx_model_output(model_path, img_size=(80, 80)):
    session = onnxruntime.InferenceSession(model_path)
    input_name = session.get_inputs()[0].name

    # Create a dummy input
    dummy_img = np.random.randint(0, 255, (1, 3, img_size[0], img_size[1]), dtype=np.uint8)
    dummy_img = dummy_img.astype(np.float32) / 255.0

    # Run inference
    output = session.run(None, {input_name: dummy_img})

    print(f"Model: {model_path}")
    print(f"Raw output: {output}")
    
    # Apply softmax to get probabilities
    softmax = torch.nn.functional.softmax(torch.from_numpy(output[0]), dim=1).numpy()
    print(f"Softmax output: {softmax}")
    print(f"Predicted class: {np.argmax(softmax, axis=1)[0]}")
    print(f"Class probabilities: {softmax.tolist()[0]}")

if __name__ == '__main__':
    test_onnx_model_output("D:/Study/Home_work/COS30082/Project/models/antispoof/2.7_80x80_MiniFASNetV2.onnx")
    test_onnx_model_output("D:/Study/Home_work/COS30082/Project/models/antispoof/4_0_0_80x80_MiniFASNetV1SE.onnx")