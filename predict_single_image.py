import torch
from autosam_utils import *
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
import torch.nn as nn
import torch.nn.parallel
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torchvision.models as models
from PIL import Image
import numpy as np
import torch
import matplotlib.pyplot as plt
from functools import partial

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


#from segment_anything.modeling import ImageEncoderViT, TwoWayTransformer

def predict_segmentation(model, image_path, device='cuda'):
    # Load and prepare the image
    img = Image.open(image_path).convert('RGB')
    
    # Apply the same transforms used during training (except augmentations)
    transform = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Transform and add batch dimension
    img_tensor = transform(img).unsqueeze(0).to(device)
    
    # Set model to evaluation mode
    #model.eval()
    
    # Inference
    with torch.no_grad():
        mask_pred, iou_pred = model(img_tensor)
        
        # Reshape the output to [batch, classes, height, width]
        b = img_tensor.shape[0]  # should be 1 for single image
        h, w = img_tensor.shape[2], img_tensor.shape[3]
        mask_pred = mask_pred.view(b, -1, h, w)
        
        # Get probabilities with softmax
        prob = torch.softmax(mask_pred, dim=1)
        
        # Get the predicted class (foreground = class 1)
        # For binary segmentation, we take the second channel (index 1)
        pred_mask = (prob[:, 1] > 0.5).float()
        
        # Convert to numpy for visualization
        pred_mask_np = pred_mask.squeeze().cpu().numpy()
    
    return pred_mask_np, img

def segment_image(model, image_path, save_path='output.png', device='cuda'):
    
    # Predict mask
    mask, original_img = predict_segmentation(model, image_path, device)
    
    # Visualize results and save to path
    visualize(original_img, mask, save_path)
    
    return mask

def visualize(image, mask, save_path='output.png'):
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.imshow(image)
    plt.title('Original Image')
    plt.axis('off')
    
    plt.subplot(1, 2, 2)
    plt.imshow(mask, cmap='gray')
    plt.title('Predicted Mask')
    plt.axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()  # Close the figure to free memory
    
    print(f"Visualization saved to {save_path}")


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
build_sam_seg = build_sam_vit_h_seg_cnn
HOME = '/teamspace/studios/this_studio'
sam2_checkpoint = f"{HOME}/sam2/checkpoints/sam2.1_hiera_base_plus.pt"
model_cfg = "configs/sam2.1/sam2.1_hiera_b+.yaml"
sam2_model = build_sam2(model_cfg, sam2_checkpoint, device=device)
sam2_encoder = sam2_model.image_encoder.to(device)

model = _build_sam_seg_model(sam2_encoder, 2).to(device)
checkpoint_path = "/teamspace/studios/this_studio/sam2/output_experiment/leaf_segmentation/checkpoints/model_best.pth"
checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

model = model.to(device)
state_dict = checkpoint["state_dict"]

# Strip "module." prefix if it exists
from collections import OrderedDict
new_state_dict = OrderedDict()
for k, v in state_dict.items():
    new_key = k.replace("module.", "")  # or adjust if your keys are different
    new_state_dict[new_key] = v

# Load into model
model.load_state_dict(new_state_dict, strict=False)

# Set model to evaluation mode
model.eval()

image_path = "/teamspace/studios/this_studio/greenstand_segmentation_model_build/segmentation_model_build/Deeplab/india_sam_dino_annotations_large/samples/india_carica papaya (pawpaw)2020.09.18.16.22.33_27.94776972848922_77.28961287997663_2c1d4f93-843b-4432-9e07-57c879c02dc2_img_20200918_155300_512968206581480670.jpg"

mask = segment_image(model, image_path)
