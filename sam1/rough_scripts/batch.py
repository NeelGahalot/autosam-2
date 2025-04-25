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

def benchmark_sam(batch_size=4, num_runs=5, warmup_runs=3):
    # Setup
    model_type = 'vit_l'
    checkpoint = '/teamspace/studios/this_studio/greenstand_segmentation_model_build/segmentation_model_build/AutoSAM/scripts/sam_vit_l_0b3195.pth'
    device = 'cuda:0'
    sam_model = sam_model_registry[model_type](checkpoint=checkpoint)
    sam_model.to(device)
    sam_model.eval()
    transform = ResizeLongestSide(sam_model.image_encoder.img_size)

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

    # Preprocess images once
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

    # Create a batch of images by duplicating to reach desired batch size
    duplicated_processed_images = []
    for i in range(batch_size):
        duplicated_processed_images.append(processed_images[i % len(processed_images)])

    # Prepare batched input once
    batched_input = torch.cat(duplicated_processed_images, dim=0)

    # ----------------------
    # Warmup runs for both sequential and batch
    # ----------------------
    print(f"Performing {warmup_runs} warmup runs...")
    
    # Warmup for sequential
    for _ in range(warmup_runs):
        for i, image in enumerate(images):
            with torch.no_grad():
                # Image encoder
                image_embedding = sam_model.image_encoder(processed_images[i])
                
                # Prompt encoder
                box = transform.apply_boxes(bbox_coords[i], original_sizes[i])
                box_torch = torch.as_tensor(box, dtype=torch.float, device=device)[None, :]
                sparse_embeddings, dense_embeddings = sam_model.prompt_encoder(
                    points=None, boxes=box_torch, masks=None,
                )
                
                # Mask decoder
                low_res_masks, _ = sam_model.mask_decoder(
                    image_embeddings=image_embedding,
                    image_pe=sam_model.prompt_encoder.get_dense_pe(),
                    sparse_prompt_embeddings=sparse_embeddings,
                    dense_prompt_embeddings=dense_embeddings,
                    multimask_output=False,
                )
    
    # Warmup for batch
    for _ in range(warmup_runs):
        with torch.no_grad():
            # Image encoder - batch
            batched_embeddings = sam_model.image_encoder(batched_input)
            
            # Process individual results
            for i in range(len(images)):
                # Prompt encoder
                box = transform.apply_boxes(bbox_coords[i], original_sizes[i])
                box_torch = torch.as_tensor(box, dtype=torch.float, device=device)[None, :]
                sparse_embeddings, dense_embeddings = sam_model.prompt_encoder(
                    points=None, boxes=box_torch, masks=None,
                )
                
                # Mask decoder
                low_res_masks, _ = sam_model.mask_decoder(
                    image_embeddings=batched_embeddings[i:i+1],
                    image_pe=sam_model.prompt_encoder.get_dense_pe(),
                    sparse_prompt_embeddings=sparse_embeddings,
                    dense_prompt_embeddings=dense_embeddings,
                    multimask_output=False,
                )
    
    # Synchronize GPU to ensure all previous operations are complete
    torch.cuda.synchronize()
    
    # =====================
    # SEQUENTIAL PROCESSING
    # =====================
    print(f"\nRunning sequential processing for {num_runs} iterations...")
    for run in range(num_runs):
        run_total_start = time.time()
        
        for i, image in enumerate(images):
            # Image encoder
            torch.cuda.synchronize()
            start = time.time()
            with torch.no_grad():
                image_embedding = sam_model.image_encoder(processed_images[i])
            torch.cuda.synchronize()
            image_encoder_time = time.time() - start
            sequential_image_encoder_times.append(image_encoder_time)
            
            # Prompt encoder
            torch.cuda.synchronize()
            start = time.time()
            with torch.no_grad():
                box = transform.apply_boxes(bbox_coords[i], original_sizes[i])
                box_torch = torch.as_tensor(box, dtype=torch.float, device=device)[None, :]
                sparse_embeddings, dense_embeddings = sam_model.prompt_encoder(
                    points=None, boxes=box_torch, masks=None,
                )
            torch.cuda.synchronize()
            prompt_encoder_time = time.time() - start
            sequential_prompt_encoder_times.append(prompt_encoder_time)
            
            # Mask decoder
            torch.cuda.synchronize()
            start = time.time()
            with torch.no_grad():
                low_res_masks, iou_predictions = sam_model.mask_decoder(
                    image_embeddings=image_embedding,
                    image_pe=sam_model.prompt_encoder.get_dense_pe(),
                    sparse_prompt_embeddings=sparse_embeddings,
                    dense_prompt_embeddings=dense_embeddings,
                    multimask_output=False,
                )
            torch.cuda.synchronize()
            mask_decoder_time = time.time() - start
            sequential_mask_decoder_times.append(mask_decoder_time)
        
        torch.cuda.synchronize()
        run_total_time = time.time() - run_total_start
        sequential_total_times.append(run_total_time)

    # ==================
    # BATCH PROCESSING
    # ==================
    print(f"Running batch processing (batch size {batch_size}) for {num_runs} iterations...")
    for run in range(num_runs):
        run_total_start = time.time()
        
        # Batch process using image encoder
        torch.cuda.synchronize()
        start = time.time()
        with torch.no_grad():
            batched_embeddings = sam_model.image_encoder(batched_input)
        torch.cuda.synchronize()
        batch_image_encoder_time = time.time() - start
        batch_image_encoder_times.append(batch_image_encoder_time)
        
        # Now process each prompt and mask sequentially
        for i in range(len(images)):
            # Get the corresponding embedding from batch
            image_embedding = batched_embeddings[i:i+1]
            
            # Prompt encoder (still sequential)
            torch.cuda.synchronize()
            start = time.time()
            with torch.no_grad():
                box = transform.apply_boxes(bbox_coords[i], original_sizes[i])
                box_torch = torch.as_tensor(box, dtype=torch.float, device=device)[None, :]
                sparse_embeddings, dense_embeddings = sam_model.prompt_encoder(
                    points=None, boxes=box_torch, masks=None,
                )
            torch.cuda.synchronize()
            prompt_encoder_time = time.time() - start
            batch_prompt_encoder_times.append(prompt_encoder_time)
            
            # Mask decoder (still sequential)
            torch.cuda.synchronize()
            start = time.time()
            with torch.no_grad():
                low_res_masks, iou_predictions = sam_model.mask_decoder(
                    image_embeddings=image_embedding,
                    image_pe=sam_model.prompt_encoder.get_dense_pe(),
                    sparse_prompt_embeddings=sparse_embeddings,
                    dense_prompt_embeddings=dense_embeddings,
                    multimask_output=False,
                )
            torch.cuda.synchronize()
            mask_decoder_time = time.time() - start
            batch_mask_decoder_times.append(mask_decoder_time)
        
        torch.cuda.synchronize()
        run_total_time = time.time() - run_total_start
        batch_total_times.append(run_total_time)

    # Calculate per-image times for fair comparison
    sequential_per_image = [t/len(images) for t in sequential_total_times]
    batch_per_image = [t/len(images) for t in batch_total_times]

    # Print results
    print("\nRESULTS:")
    print("===== SEQUENTIAL (per image) =====")
    print(f"Image Encoder: {sequential_image_encoder_times} - Avg: {np.mean(sequential_image_encoder_times):.4f}s")
    print(f"Prompt Encoder: {sequential_prompt_encoder_times} - Avg: {np.mean(sequential_prompt_encoder_times):.4f}s")
    print(f"Mask Decoder: {sequential_mask_decoder_times} - Avg: {np.mean(sequential_mask_decoder_times):.4f}s")
    print(f"Total (per image): {sequential_per_image} - Avg: {np.mean(sequential_per_image):.4f}s")

    print(f"\n===== BATCH (batch size {batch_size}, per image) =====")
    print(f"Image Encoder: {batch_image_encoder_times} - Avg: {np.mean(batch_image_encoder_times)/len(images):.4f}s")
    print(f"Prompt Encoder: {batch_prompt_encoder_times} - Avg: {np.mean(batch_prompt_encoder_times):.4f}s")
    print(f"Mask Decoder: {batch_mask_decoder_times} - Avg: {np.mean(batch_mask_decoder_times):.4f}s")
    print(f"Total (per image): {batch_per_image} - Avg: {np.mean(batch_per_image):.4f}s")

    # Calculate speedup
    encoder_speedup = (np.mean(sequential_image_encoder_times) * len(images)) / np.mean(batch_image_encoder_times)
    total_speedup = np.mean(sequential_per_image) / np.mean(batch_per_image)
    
    print(f"\nImage Encoder Speedup: {encoder_speedup:.2f}x")
    print(f"Total Speedup (per image): {total_speedup:.2f}x")

    # Create comparison plot
    plt.figure(figsize=(14, 8))

    # Data preparation
    components = ['Image Encoder', 'Prompt Encoder', 'Mask Decoder', 'Total (per image)']
    
    # Calculate per-image timings for image encoder with batch
    seq_avg = [
        np.mean(sequential_image_encoder_times), 
        np.mean(sequential_prompt_encoder_times), 
        np.mean(sequential_mask_decoder_times), 
        np.mean(sequential_per_image)
    ]
    
    batch_avg = [
        np.mean(batch_image_encoder_times) / len(images),  # Divide by number of images for fair comparison
        np.mean(batch_prompt_encoder_times), 
        np.mean(batch_mask_decoder_times), 
        np.mean(batch_per_image)
    ]

    # Plotting
    x = np.arange(len(components))
    width = 0.35

    fig, ax = plt.subplots(figsize=(12, 7))
    rects1 = ax.bar(x - width/2, seq_avg, width, label='Sequential')
    rects2 = ax.bar(x + width/2, batch_avg, width, label=f'Batch (size {batch_size})')

    # Labels and formatting
    ax.set_ylabel('Time (seconds)')
    ax.set_title(f'SAM Component Timing Comparison: Sequential vs Batch (size {batch_size})')
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
    plt.title(f'Batch Processing Speedup (size {batch_size}) Over Sequential')
    for i, v in enumerate(speedup):
        plt.text(i, v + 0.05, f'{v:.2f}x', ha='center')

    plt.tight_layout()
    plt.savefig(f'SAM_Timing_Batch{batch_size}_vs_Sequential.jpg')
    plt.figure(1)
    plt.savefig(f'SAM_Component_Comparison_Batch{batch_size}.jpg')

    print("Benchmark complete. Results saved to image files.")
    
    return {
        'sequential': {
            'image_encoder': np.mean(sequential_image_encoder_times),
            'prompt_encoder': np.mean(sequential_prompt_encoder_times),
            'mask_decoder': np.mean(sequential_mask_decoder_times),
            'total_per_image': np.mean(sequential_per_image)
        },
        'batch': {
            'image_encoder_per_image': np.mean(batch_image_encoder_times) / len(images),
            'prompt_encoder': np.mean(batch_prompt_encoder_times),
            'mask_decoder': np.mean(batch_mask_decoder_times),
            'total_per_image': np.mean(batch_per_image)
        },
        'speedup': {
            'image_encoder': encoder_speedup,
            'total_per_image': total_speedup
        }
    }

if __name__ == "__main__":
    # Run with different batch sizes
    batch_sizes = [4, 8, 16]
    results = {}
    
    for batch_size in batch_sizes:
        print(f"\n{'='*50}")
        print(f"RUNNING BENCHMARK WITH BATCH SIZE {batch_size}")
        print(f"{'='*50}")
        results[batch_size] = benchmark_sam(batch_size=batch_size)
    
    # Print summary of all results
    print("\n\nSUMMARY OF ALL BATCH SIZES:")
    print(f"{'='*50}")
    print(f"{'Batch Size':<10} | {'Image Encoder Speedup':<25} | {'Total Speedup':<15}")
    print(f"{'-'*10} | {'-'*25} | {'-'*15}")
    
    for batch_size, result in results.items():
        print(f"{batch_size:<10} | {result['speedup']['image_encoder']:<25.2f}x | {result['speedup']['total_per_image']:<15.2f}x")