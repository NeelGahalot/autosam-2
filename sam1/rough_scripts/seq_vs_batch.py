import torch
import numpy as np
import cv2
import time
import matplotlib.pyplot as plt
from segment_anything import SamPredictor, sam_model_registry
from segment_anything.utils.transforms import ResizeLongestSide
from torch.nn.functional import normalize

def threshold(mask, threshold_val=0.0):
    return (mask > threshold_val).float()

# Setup
model_type = 'vit_l'
checkpoint = '/teamspace/studios/this_studio/greenstand_segmentation_model_build/segmentation_model_build/AutoSAM/scripts/sam_vit_l_0b3195.pth'
device = 'cuda:0'
sam_model = sam_model_registry[model_type](checkpoint=checkpoint)
sam_model.to(device)
sam_model.eval()

# Load images
image1 = cv2.imread('truck.jpg')
image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2RGB)
image2 = cv2.imread('groceries.jpg')
image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2RGB)
images = [image1, image2]

# Define bounding boxes (one for each image)
bbox_coords = [
    np.array([[425, 600, 700, 875]]),  # for truck.jpg
    np.array([[200, 300, 400, 500]])   # for groceries.jpg - adjust as needed
]

# Performance tracking
sequential_image_encoder_times = []
sequential_prompt_encoder_times = []
sequential_mask_decoder_times = []
sequential_total_times = []

batch_image_encoder_times = []
batch_prompt_encoder_times = []
batch_mask_decoder_times = []
batch_total_times = []

num_runs = 3
transform = ResizeLongestSide(sam_model.image_encoder.img_size)

# =====================
# SEQUENTIAL PROCESSING
# =====================
print("Running sequential processing...")
for run in range(num_runs):
    run_total_start = time.time()
    
    for i, image in enumerate(images):
        # Preprocess image
        input_image = transform.apply_image(image)
        input_image_torch = torch.as_tensor(input_image, device=device)
        transformed_image = input_image_torch.permute(2, 0, 1).contiguous()[None, :, :, :]
        input_image_processed = sam_model.preprocess(transformed_image)
        original_image_size = image.shape[:2]
        input_size = tuple(transformed_image.shape[-2:])
        
        # Image encoder
        start = time.time()
        with torch.no_grad():
            image_embedding = sam_model.image_encoder(input_image_processed)
        image_encoder_time = time.time() - start
        sequential_image_encoder_times.append(image_encoder_time)
        
        # Prompt encoder
        start = time.time()
        with torch.no_grad():
            box = transform.apply_boxes(bbox_coords[i], original_image_size)
            box_torch = torch.as_tensor(box, dtype=torch.float, device=device)[None, :]
            sparse_embeddings, dense_embeddings = sam_model.prompt_encoder(
                points=None,
                boxes=box_torch,
                masks=None,
            )
        prompt_encoder_time = time.time() - start
        sequential_prompt_encoder_times.append(prompt_encoder_time)
        
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
        mask_decoder_time = time.time() - start
        sequential_mask_decoder_times.append(mask_decoder_time)
        
        # Optional: postprocess
        upscaled_masks = sam_model.postprocess_masks(low_res_masks, input_size, original_image_size).to(device)
        binary_mask = normalize(threshold(upscaled_masks, 0.0))
    
    run_total_time = time.time() - run_total_start
    sequential_total_times.append(run_total_time)



# ==================
# BATCH PROCESSING
# ==================
print("Running batch processing with batch size 4...")
for run in range(num_runs):
    run_total_start = time.time()
    
    # Preprocess both images
    processed_images = []
    transformed_images = []
    original_sizes = []
    input_sizes = []
    
    for image in images:
        input_image = transform.apply_image(image)
        input_image_torch = torch.as_tensor(input_image, device=device)
        transformed_image = input_image_torch.permute(2, 0, 1).contiguous()[None, :, :, :]
        processed_images.append(sam_model.preprocess(transformed_image))
        transformed_images.append(transformed_image)
        original_sizes.append(image.shape[:2])
        input_sizes.append(tuple(transformed_image.shape[-2:]))
    
    # Batch process using image encoder - process 4 images by duplicating the 2 original images
    start = time.time()
    with torch.no_grad():
        # Duplicate the processed images to create a batch of 4
        duplicated_images = processed_images + processed_images  # This creates a list with 4 images
        batched_input = torch.cat(duplicated_images, dim=0)  # Now batched_input has 4 images
        batched_embeddings = sam_model.image_encoder(batched_input)
    batch_image_encoder_time = time.time() - start
    batch_image_encoder_times.append(batch_image_encoder_time)
    
    # Now process each prompt and mask sequentially - only process the 2 original images
    for i in range(len(images)):
        # Get the corresponding embedding from batch - use only the first 2 embeddings
        image_embedding = batched_embeddings[i:i+1]
        
        # Prompt encoder (still sequential)
        start = time.time()
        with torch.no_grad():
            box = transform.apply_boxes(bbox_coords[i], original_sizes[i])
            box_torch = torch.as_tensor(box, dtype=torch.float, device=device)[None, :]
            sparse_embeddings, dense_embeddings = sam_model.prompt_encoder(
                points=None,
                boxes=box_torch,
                masks=None,
            )
        prompt_encoder_time = time.time() - start
        batch_prompt_encoder_times.append(prompt_encoder_time)
        
        # Mask decoder (still sequential)
        start = time.time()
        with torch.no_grad():
            low_res_masks, iou_predictions = sam_model.mask_decoder(
                image_embeddings=image_embedding,
                image_pe=sam_model.prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=sparse_embeddings,
                dense_prompt_embeddings=dense_embeddings,
                multimask_output=False,
            )
        mask_decoder_time = time.time() - start
        batch_mask_decoder_times.append(mask_decoder_time)
        
        # Optional: postprocess
        upscaled_masks = sam_model.postprocess_masks(low_res_masks, input_sizes[i], original_sizes[i]).to(device)
        binary_mask = normalize(threshold(upscaled_masks, 0.0))
    
    run_total_time = time.time() - run_total_start
    batch_total_times.append(run_total_time)

# Print results
print("\nRESULTS:")
print("===== SEQUENTIAL =====")
print(f"Image Encoder: {sequential_image_encoder_times} - Avg: {np.mean(sequential_image_encoder_times):.4f}s")
print(f"Prompt Encoder: {sequential_prompt_encoder_times} - Avg: {np.mean(sequential_prompt_encoder_times):.4f}s")
print(f"Mask Decoder: {sequential_mask_decoder_times} - Avg: {np.mean(sequential_mask_decoder_times):.4f}s")
print(f"Total: {sequential_total_times} - Avg: {np.mean(sequential_total_times):.4f}s")

print("\n===== BATCH =====")
print(f"Image Encoder: {batch_image_encoder_times} - Avg: {np.mean(batch_image_encoder_times):.4f}s")
print(f"Prompt Encoder: {batch_prompt_encoder_times} - Avg: {np.mean(batch_prompt_encoder_times):.4f}s")
print(f"Mask Decoder: {batch_mask_decoder_times} - Avg: {np.mean(batch_mask_decoder_times):.4f}s")
print(f"Total: {batch_total_times} - Avg: {np.mean(batch_total_times):.4f}s")

# Create comparison plot
plt.figure(figsize=(14, 8))

# Data preparation
components = ['Image Encoder', 'Prompt Encoder', 'Mask Decoder', 'Total']
seq_avg = [np.mean(sequential_image_encoder_times), np.mean(sequential_prompt_encoder_times), 
           np.mean(sequential_mask_decoder_times), np.mean(sequential_total_times)]
batch_avg = [np.mean(batch_image_encoder_times), np.mean(batch_prompt_encoder_times), 
             np.mean(batch_mask_decoder_times), np.mean(batch_total_times)]

# Plotting
x = np.arange(len(components))
width = 0.35

fig, ax = plt.subplots(figsize=(12, 7))
rects1 = ax.bar(x - width/2, seq_avg, width, label='Sequential')
rects2 = ax.bar(x + width/2, batch_avg, width, label='Batch')

# Labels and formatting
ax.set_ylabel('Time (seconds)')
ax.set_title('SAM Component Timing Comparison: Sequential vs Batch')
ax.set_xticks(x)
ax.set_xticklabels(components)
ax.legend()

# Add timing labels on bars
def autolabel(rects):
    for rect in rects:
        height = rect.get_height()
        ax.annotate(f'{height:.4f}s',
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')

autolabel(rects1)
autolabel(rects2)

# Calculate and display speedup
speedup = [s/b if b > 0 else 0 for s, b in zip(seq_avg, batch_avg)]
plt.figure(figsize=(10, 6))
plt.bar(components, speedup)
plt.axhline(y=1.0, color='r', linestyle='-', alpha=0.3)
plt.ylabel('Speedup Factor (Sequential/Batch)')
plt.title('Batch Processing Speedup Over Sequential')
for i, v in enumerate(speedup):
    plt.text(i, v + 0.05, f'{v:.2f}x', ha='center')

plt.tight_layout()
plt.savefig('SAM_Timing_Batch_vs_Sequential.jpg')
plt.figure(1)
plt.savefig('SAM_Component_Comparison.jpg')

print("Benchmark complete. Results saved to image files.")