from autosam_utils import *
from PIL import Image
import torch
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
import numpy as np
device ='cuda'
HOME = '/teamspace/studios/this_studio'
image = Image.open(f'{HOME}/sam2/notebooks/images/truck.jpg')
image = np.array(image.convert("RGB"))

sam2_checkpoint = f"{HOME}/sam2/checkpoints/sam2.1_hiera_base_plus.pt"
model_cfg = "configs/sam2.1/sam2.1_hiera_b+.yaml"

sam2_model = build_sam2(model_cfg, sam2_checkpoint, device=device)

predictor = SAM2ImagePredictor(sam2_model)
predictor.set_image(image)
encoder_outputs = predictor.get_image_embedding()
print(encoder_outputs.shape)
print(sam2_model)

# building autosam with SAM2 Encoder

def _build_sam_seg_model(
    image_encoder,  # Now passed in!
    num_classes,
    checkpoint=None,
):
    prompt_embed_dim = 256

    sam_seg = AutoSamSeg(
        image_encoder=image_encoder,
        seg_decoder=MaskDecoder(
            num_multimask_outputs=1,
            transformer=TwoWayTransformer(
                depth=2,
                embedding_dim=prompt_embed_dim,
                mlp_dim=2048,
                num_heads=8,
            ),
            transformer_dim=prompt_embed_dim,
            iou_head_depth=3,
            iou_head_hidden_dim=256,
            num_classes=num_classes,
        ),
    )

    if checkpoint is not None:
        with open(checkpoint, "rb") as f:
            state_dict = torch.load(f)

        loaded_keys = {
            k: v for k, v in state_dict.items()
            if k in sam_seg.state_dict().keys() and 'iou' not in k and "mask_tokens" not in k
        }
        sam_seg.load_state_dict(loaded_keys, strict=False)
        print("loaded keys:", loaded_keys.keys())

    return sam_seg


def build_sam_vit_h_seg_cnn(num_classes=14, checkpoint=None):
    return _build_sam_seg_model(
        encoder_embed_dim=1280,
        encoder_depth=32,
        encoder_num_heads=16,
        encoder_global_attn_indexes=[7, 15, 23, 31],
        num_classes=num_classes,
        checkpoint=checkpoint,
    )


build_sam_seg = build_sam_vit_h_seg_cnn


def build_sam_vit_l_seg_cnn(num_classes=14, checkpoint=None):
    return _build_sam_seg_model(
        encoder_embed_dim=1024,
        encoder_depth=24,
        encoder_num_heads=16,
        encoder_global_attn_indexes=[5, 11, 17, 23],
        num_classes=num_classes,
        checkpoint=checkpoint,
    )


def build_sam_vit_b_seg_cnn(num_classes=14, checkpoint=None):
    return _build_sam_seg_model(
        encoder_embed_dim=768,
        encoder_depth=12,
        encoder_num_heads=12,
        encoder_global_attn_indexes=[2, 5, 8, 11],
        num_classes=num_classes,
        checkpoint=checkpoint,
    )


sam_seg_model_registry = {
    "default": build_sam_seg,
    "vit_h": build_sam_seg,
    "vit_l": build_sam_vit_l_seg_cnn,
    "vit_b": build_sam_vit_b_seg_cnn,
}
sam2_encoder = sam2_model.image_encoder.to(device)
autosam = _build_sam_seg_model(sam2_encoder, 2).to(device)

# Create input on the same device
images = torch.rand(4, 3, 512, 512, device=device)

# Run model
outputs = autosam(images)

for i, out in enumerate(outputs):
    if isinstance(out, torch.Tensor):
        print(f"Output {i}: Tensor of shape {out.shape}")
    else:
        print(f"Output {i}: {type(out)}")
'''
for k, v in outputs.items():
    if isinstance(v, torch.Tensor):
        print(f"{k}: {v.shape}")
    else:
        print(f"{k}: {type(v)}")

'''