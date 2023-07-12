import torchvision
# from torchvision.models import vit_b_16
# from torchvision.models import ViT_B_16_Weights
import os
import onnx_tool
import torch
import numpy
from transformers import ViTImageProcessor, ViTForImageClassification

vit_b_16_dir = "vit_b_16"
os.makedirs(vit_b_16_dir, exist_ok=True)
vit_b_16_path = vit_b_16_dir + "/model.onnx"

processor = ViTImageProcessor.from_pretrained('google/vit-base-patch16-384')
model = ViTForImageClassification.from_pretrained('google/vit-base-patch16-384')

# model = vit_b_16(weights=ViT_B_16_Weights.IMAGENET1K_SWAG_E2E_V1)
N = 1
H = 384
W = 384
C = 3
inputs = torch.rand((N, C, H, W))

torch.onnx.export(
    model,
    inputs,
    vit_b_16_path,
    input_names = ["image_sample"],
    output_names = ["output"],
    dynamic_axes = {
        "image_sample": {0: 'batch', 1: 'channels', 2: 'height', 3: 'width'}
    },
    opset_version=14
)

onnx_tool.model_profile(vit_b_16_path, saveshapesmodel= vit_b_16_path + "_shapes.onnx", savenode=vit_b_16_path + "_profile.csv", dynamic_shapes={'image_sample': (numpy.random.rand(N, C, H, W))})

