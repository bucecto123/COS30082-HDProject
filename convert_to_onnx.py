

import torch
from SilentFaceAntiSpoofing.src.model_lib.MiniFASNet import MiniFASNetV1SE, MiniFASNetV2

from collections import OrderedDict

def remove_module_prefix(state_dict):
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:] # remove `module.`
        new_state_dict[name] = v
    return new_state_dict

def convert_to_onnx(model, model_path, output_path):
    """Converts a PyTorch model to ONNX format."""
    state_dict = torch.load(model_path, map_location=torch.device('cpu'))
    if list(state_dict.keys())[0].startswith('module.'):
        state_dict = remove_module_prefix(state_dict)
    model.load_state_dict(state_dict)
    model.eval()
    dummy_input = torch.randn(1, 3, 80, 80)
    torch.onnx.export(model, dummy_input, output_path, verbose=False, input_names=['input'], output_names=['output'])

if __name__ == '__main__':
    # Convert the first model
    model_v1se = MiniFASNetV1SE(conv6_kernel=(5, 5))
    model_path_v1se = "D:/Study/Home_work/COS30082/Project/models/antispoof/4_0_0_80x80_MiniFASNetV1SE.pth"
    output_path_v1se = "D:/Study/Home_work/COS30082/Project/models/antispoof/4_0_0_80x80_MiniFASNetV1SE.onnx"
    convert_to_onnx(model_v1se, model_path_v1se, output_path_v1se)
    print(f"Model converted to {output_path_v1se}")

    # Convert the second model
    model_v2 = MiniFASNetV2(conv6_kernel=(5, 5))
    model_path_v2 = "D:/Study/Home_work/COS30082/Project/models/antispoof/2.7_80x80_MiniFASNetV2.pth"
    output_path_v2 = "D:/Study/Home_work/COS30082/Project/models/antispoof/2.7_80x80_MiniFASNetV2.onnx"
    convert_to_onnx(model_v2, model_path_v2, output_path_v2)
    print(f"Model converted to {output_path_v2}")

