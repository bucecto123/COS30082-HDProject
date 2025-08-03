import torch
import numpy as np
import cv2
from model_lib.MiniFASNet import MiniFASNetV1, MiniFASNetV2, MiniFASNetV1SE, MiniFASNetV2SE

# Choose which model to test
def get_model(model_type='MiniFASNetV2', num_classes=3):
    if model_type == 'MiniFASNetV1':
        return MiniFASNetV1(num_classes=num_classes)
    elif model_type == 'MiniFASNetV2':
        return MiniFASNetV2(num_classes=num_classes)
    elif model_type == 'MiniFASNetV1SE':
        return MiniFASNetV1SE(num_classes=num_classes)
    elif model_type == 'MiniFASNetV2SE':
        return MiniFASNetV2SE(num_classes=num_classes)
    else:
        raise ValueError('Unknown model type')

def test_model_output(model_type='MiniFASNetV2', num_classes=3, img_size=(80, 80)):
    model = get_model(model_type, num_classes)
    model.eval()
    # Create a dummy input (batch_size=1, channels=3, height, width)
    dummy_img = np.random.randint(0, 255, (1, 3, img_size[0], img_size[1]), dtype=np.uint8)
    dummy_img = torch.tensor(dummy_img, dtype=torch.float32) / 255.0
    with torch.no_grad():
        output = model(dummy_img)
        print(f"Model: {model_type}, num_classes: {num_classes}")
        print(f"Output shape: {output.shape}")
        print(f"Raw output: {output}")
        softmax = torch.nn.functional.softmax(output, dim=1)
        print(f"Softmax output: {softmax}")
        print(f"Predicted class: {torch.argmax(softmax, dim=1).item()}")
        print(f"Class probabilities: {softmax.numpy().tolist()[0]}")

if __name__ == '__main__':
    # Test all model variants
    test_model_output('MiniFASNetV1', num_classes=3)
    test_model_output('MiniFASNetV2', num_classes=3)
    test_model_output('MiniFASNetV1SE', num_classes=3)
    test_model_output('MiniFASNetV2SE', num_classes=4)
