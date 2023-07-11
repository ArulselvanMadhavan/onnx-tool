import onnx_tool
import torch
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
import numpy

model_id = "stabilityai/stable-diffusion-2-1"
device = 'cpu'
sd_pipeline = StableDiffusionPipeline.from_pretrained(model_id)
sd_pipeline = sd_pipeline.to(device)

# text_input = sd_pipeline.tokenizer(
#     "A sample prompt",
#     padding="max_length",
#     max_length=sd_pipeline.tokenizer.model_max_length,
#     truncation=True,
#     return_tensors="pt",
# )

enc_model_path = "onnx-stable-diffusion-v2-1/text_encoder_model.onnx"
unet_model_path = "onnx-stable-diffusion-v2-1/unet_model.onnx"
vae_decoder_path = "onnx-stable-diffusion-v2-1/vae_decoder.onnx"
# torch.onnx.export(
#     sd_pipeline.text_encoder,
#     # casting to torch.int32 until the CLIP fix is released: https://github.com/huggingface/transformers/pull/18515/files
#     (text_input.input_ids.to(device=device, dtype=torch.int32)),
#     enc_model_path,
#     input_names=["input_ids"],
#     output_names=["last_hidden_state", "pooler_output"],
#     # dynamic_axes={
#     #     "input_ids": {0: "batch", 1: "sequence"},
#     # },
#     opset_version=14,
#     export_params=False
# )

# def export_unet(sd_pipeline):
#     unet_in_channels = sd_pipeline.unet.config.in_channels
#     unet_sample_size = sd_pipeline.unet.config.sample_size
#     unet_path = unet_model_path
#     num_tokens = sd_pipeline.text_encoder.config.max_position_embeddings
#     text_hidden_size = sd_pipeline.text_encoder.config.hidden_size
#     torch.onnx.export(
#         sd_pipeline.unet,
#         (
#             torch.randn(1, unet_in_channels, unet_sample_size, unet_sample_size).to(device=device),
#             torch.randn(1).to(device=device),
#             torch.randn(1, num_tokens, text_hidden_size).to(device=device),
#             False,
#         ),
#         f=unet_path,
#         input_names=["sample", "timestep", "encoder_hidden_states", "return_dict"],
#         output_names=["out_sample"],  # has to be different from "sample" for correct tracing
#         dynamic_axes={
#             "sample": {0: "batch", 1: "channels", 2: "height", 3: "width"},
#             "timestep": {0: "batch"},
#             "encoder_hidden_states": {0: "batch", 1: "sequence"},
#         },
#         do_constant_folding=True,
#         opset_version=14,
#         export_params=False
#     )

def export_vae_encoder_decoder(sd_pipeline):
    vae_decoder = sd_pipeline.vae
    vae_in_channels = vae_decoder.config.in_channels
    vae_sample_size = vae_decoder.config.sample_size    
    vae_latent_channels = vae_decoder.config.latent_channels
    unet_sample_size = sd_pipeline.unet.config.sample_size
    vae_out_channels = vae_decoder.config.out_channels

    # forward only through the decoder part
    vae_decoder.forward = vae_decoder.decode
    torch.onnx.export(
        vae_decoder,
        (
            torch.randn(1, vae_latent_channels, unet_sample_size, unet_sample_size).to(device=device),
            False,
        ),
        vae_decoder_path,
        input_names=["latent_sample", "return_dict"],
        output_names=["sample"],
        dynamic_axes={
            "latent_sample": {0: "batch", 1: "channels", 2: "height", 3: "width"},
        },
        opset_version=14,
    )

# onnx_tool.model_simplify_names(enc_model_path, savemodel=enc_model_path + '_renamed.onnx', custom_inputs={'input':'batchxsequence'})
# onnx_tool.model_profile(enc_model_path,
#                         saveshapesmodel= enc_model_path + '_shapes.onnx',
#                         savenode = enc_model_path + "_profile.csv")
# export_unet(sd_pipeline)
# onnx_tool.model_profile(unet_model_path, saveshapesmodel = unet_model_path + '_shapes.onnx', savenode = unet_model_path + "_profile.csv", dynamic_shapes={"sample": (numpy.random.rand(1, 4, 96, 96)), "timestep":(numpy.random.rand(1)), "encoder_hidden_states":(numpy.random.rand(1, 77, 1024))})

# export_vae_encoder_decoder(sd_pipeline)
onnx_tool.model_profile(vae_decoder_path, saveshapesmodel = vae_decoder_path + '_shapes.onnx', savenode = vae_decoder_path + "_profile.csv", dynamic_shapes={
    "latent_sample": (numpy.random.rand(1, 4, 96, 96))
  }
)
