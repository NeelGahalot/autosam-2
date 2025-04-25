import torch
import numpy as np
import cv2
import time
import matplotlib.pyplot as plt

from segment_anything import SamPredictor, sam_model_registry
from segment_anything.utils.transforms import ResizeLongestSide
from torch.nn.functional import threshold, normalize


def threshold(mask, threshold_val=0.0, above_val=0):
    return (mask > threshold_val).float()

# Setup
model_type = 'vit_l'
checkpoint = '/teamspace/studios/this_studio/greenstand_segmentation_model_build/segmentation_model_build/AutoSAM/scripts/sam_vit_l_0b3195.pth'
device = 'cuda:0'

sam_model = sam_model_registry[model_type](checkpoint=checkpoint)
sam_model.to(device)
sam_model.eval()

image = cv2.imread('truck.jpg')
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

transform = ResizeLongestSide(sam_model.image_encoder.img_size)
input_image = transform.apply_image(image)
input_image_torch = torch.as_tensor(input_image, device=device)
transformed_image = input_image_torch.permute(2, 0, 1).contiguous()[None, :, :, :]
input_image = sam_model.preprocess(transformed_image)
original_image_size = image.shape[:2]
input_size = tuple(transformed_image.shape[-2:])

# Define bounding box (replace or loop if needed)
bbox_coords = np.array([[425, 600, 700, 875]])

# Timing containers
image_encoder_times = []
prompt_encoder_times = []
mask_decoder_times = []

for _ in range(3):
    # Image encoder
    start = time.time()
    with torch.no_grad():
        image_embedding = sam_model.image_encoder(input_image)
    image_encoder_times.append(time.time() - start)

    # Prompt encoder
    start = time.time()
    with torch.no_grad():
        box = transform.apply_boxes(bbox_coords, original_image_size)
        box_torch = torch.as_tensor(box, dtype=torch.float, device=device)[None, :]
        sparse_embeddings, dense_embeddings = sam_model.prompt_encoder(
            points=None,
            boxes=box_torch,
            masks=None,
        )
    prompt_encoder_times.append(time.time() - start)

    # Mask decoder
    start = time.time()
    with torch.no_grad():
        low_res_masks, iou_predictions = sam_model.mask_decoder(
            image_embeddings=image_embedding,
            image_pe=sam_model.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=False,
        )
    mask_decoder_times.append(time.time() - start)

    # Optional: postprocess to get binary masks
    upscaled_masks = sam_model.postprocess_masks(low_res_masks, input_size, original_image_size).to(device)
    binary_mask = normalize(threshold(upscaled_masks, 0.0, 0))

# Plotting the timing results
components = ['Image Encoder', 'Prompt Encoder', 'Mask Decoder']
timings = [image_encoder_times, prompt_encoder_times, mask_decoder_times]
print(f'image_encoder_times - {image_encoder_times}')
print(f'prompt_encoder_times - {prompt_encoder_times}')
print(f'mask_decoder_times - {mask_decoder_times}')
plt.figure(figsize=(8, 5))
plt.boxplot(timings, labels=components)
plt.ylabel('Time (seconds)')
plt.title('SAM Component Timing Over 3 Runs')
plt.grid(True)
plt.tight_layout()
plt.savefig('SAM_Timing_large_2.jpg')
