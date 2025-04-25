import torch
import numpy as np
import time
from torch import nn
from segment_anything import sam_model_registry, SamPredictor
import matplotlib.pyplot as plt

# Utility function for timing
def time_operation(operation_name, func, *args, **kwargs):
    start_time = time.time()
    result = func(*args, **kwargs)
    end_time = time.time()
    elapsed = end_time - start_time
    print(f"{operation_name} took {elapsed:.4f} seconds")
    return result, elapsed

# Create a dummy SAM model for testing
def get_dummy_sam():
    # Replace with actual model initialization if available
    try:
        sam = sam_model_registry["vit_l"](checkpoint="/teamspace/studios/this_studio/greenstand_segmentation_model_build/segmentation_model_build/AutoSAM/scripts/sam_vit_l_0b3195.pth")
        print('using actual sam checkpoint')
    except:
        # Mock SAM for testing if real model isn't available
        class MockSAM(nn.Module):
            def __init__(self):
                super().__init__()
                self.image_encoder = nn.Conv2d(3, 256, kernel_size=1)
                self.prompt_encoder = nn.Module()
                self.prompt_encoder.get_dense_pe = lambda: torch.rand(1, 256, 64, 64)
                self.mask_decoder = nn.Module()
                self.device = "cuda" if torch.cuda.is_available() else "cpu"
                
            def prompt_encoder_forward(self, points, boxes, masks):
                time.sleep(0.01)  # Simulate some processing time
                return torch.rand(boxes.shape[0], 2, 256).to(boxes.device), torch.rand(boxes.shape[0], 256, 64, 64).to(boxes.device)
                
            def mask_decoder_forward(self, image_embeddings, image_pe, sparse_prompt_embeddings, dense_prompt_embeddings, multimask_output):
                time.sleep(0.02)  # Simulate some processing time
                batch_size = image_embeddings.shape[0]
                return torch.rand(batch_size, 1, 256, 256).to(image_embeddings.device), torch.rand(batch_size, 1).to(image_embeddings.device)
                
            def postprocess_masks(self, masks, input_size, original_size):
                return masks
                
        sam = MockSAM()
        
        # Monkey patch for timing measurements
        sam.image_encoder.forward = lambda x: torch.rand(x.shape[0], 256, 64, 64).to(x.device)
        sam.prompt_encoder.forward = sam.prompt_encoder_forward
        sam.mask_decoder.forward = sam.mask_decoder_forward
        
    return sam

# Main benchmarking function
def benchmark_sam(batch_sizes=[1, 2, 4, 8, 16], num_iterations=10):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Running on {device}")
    
    sam_model = get_dummy_sam().to(device)
    
    results = {
        'batch_size': [],
        'sequential': {
            'image_encoder': [],
            'prompt_encoder': [],
            'mask_decoder': [],
            'total': []
        },
        'batched': {
            'image_encoder': [],
            'prompt_encoder': [],
            'mask_decoder': [],
            'total': []
        }
    }
    
    image_size = (1024, 1024)
    
    for batch_size in batch_sizes:
        print(f"\n--- Batch Size: {batch_size} ---")
        
        # Create dummy batch data
        input_images = torch.rand(batch_size, 3, image_size[0], image_size[1]).to(device)
        boxes_torch = torch.rand(batch_size, 1, 4).to(device)  # [batch_size, 1, 4]
        
        # Store times for this batch size
        seq_encoder_time = 0
        seq_prompt_time = 0 
        seq_decoder_time = 0
        seq_total_time = 0
        
        batch_encoder_time = 0
        batch_prompt_time = 0
        batch_decoder_time = 0
        batch_total_time = 0
        
        for i in range(num_iterations):
            print(f"\nIteration {i+1}/{num_iterations}")
            
            # ---------- Sequential Processing ----------
            print("\nSequential Processing:")
            seq_start = time.time()
            
            # Sequential image encoding
            all_image_embeddings = []
            for j in range(batch_size):
                _, t_enc = time_operation(
                    f"  Image Encoder (item {j+1}/{batch_size})", 
                    lambda: sam_model.image_encoder(input_images[j:j+1])
                )
                seq_encoder_time += t_enc
                all_image_embeddings.append(_[0])
            
            # Sequential prompt encoding
            all_sparse_embeddings = []
            all_dense_embeddings = []
            for j in range(batch_size):
                (sparse_emb, dense_emb), t_prompt = time_operation(
                    f"  Prompt Encoder (item {j+1}/{batch_size})",
                    lambda: sam_model.prompt_encoder(
                        points=None,
                        boxes=boxes_torch[j:j+1],
                        masks=None
                    )
                )
                seq_prompt_time += t_prompt
                all_sparse_embeddings.append(sparse_emb)
                all_dense_embeddings.append(dense_emb)
            
            # Sequential mask decoding
            for j in range(batch_size):
                _, t_dec = time_operation(
                    f"  Mask Decoder (item {j+1}/{batch_size})",
                    lambda: sam_model.mask_decoder(
                        image_embeddings=all_image_embeddings[j],
                        image_pe=sam_model.prompt_encoder.get_dense_pe(),
                        sparse_prompt_embeddings=all_sparse_embeddings[j],
                        dense_prompt_embeddings=all_dense_embeddings[j],
                        multimask_output=False
                    )
                )
                seq_decoder_time += t_dec
                
            seq_end = time.time()
            seq_total = seq_end - seq_start
            print(f"  Total Sequential Time: {seq_total:.4f} seconds")
            seq_total_time += seq_total
            
            # ---------- Batched Processing ----------
            print("\nBatched Processing:")
            batch_start = time.time()
            
            # Batched image encoding
            image_embeddings, t_enc_batch = time_operation(
                "  Image Encoder (batch)",
                lambda: sam_model.image_encoder(input_images)
            )
            batch_encoder_time += t_enc_batch
            
            # Let's simulate both approaches for prompt encoder:
            
            # Option 1: Sequential prompt encoding but with batched image encoding
            prompt_start = time.time()
            all_sparse_emb = []
            all_dense_emb = []
            for j in range(batch_size):
                sparse_emb, dense_emb = sam_model.prompt_encoder(
                    points=None,
                    boxes=boxes_torch[j:j+1],
                    masks=None
                )
                all_sparse_emb.append(sparse_emb)
                all_dense_emb.append(dense_emb)
            sparse_embeddings = torch.cat(all_sparse_emb, dim=0)
            dense_embeddings = torch.cat(all_dense_emb, dim=0)
            prompt_end = time.time()
            t_prompt_batch = prompt_end - prompt_start
            print(f"  Prompt Encoder (sequential within batch): {t_prompt_batch:.4f} seconds")
            batch_prompt_time += t_prompt_batch
            
            # Batched mask decoding
            _, t_dec_batch = time_operation(
                "  Mask Decoder (batch)",
                lambda: sam_model.mask_decoder(
                    image_embeddings=image_embeddings,
                    image_pe=sam_model.prompt_encoder.get_dense_pe(),
                    sparse_prompt_embeddings=sparse_embeddings,
                    dense_prompt_embeddings=dense_embeddings,
                    multimask_output=False
                )
            )
            batch_decoder_time += t_dec_batch
            
            batch_end = time.time()
            batch_total = batch_end - batch_start
            print(f"  Total Batch Time: {batch_total:.4f} seconds")
            print(f"  Speedup: {seq_total/batch_total:.2f}x")
            batch_total_time += batch_total
        
        # Average times over iterations
        results['batch_size'].append(batch_size)
        
        results['sequential']['image_encoder'].append(seq_encoder_time / num_iterations)
        results['sequential']['prompt_encoder'].append(seq_prompt_time / num_iterations)
        results['sequential']['mask_decoder'].append(seq_decoder_time / num_iterations)
        results['sequential']['total'].append(seq_total_time / num_iterations)
        
        results['batched']['image_encoder'].append(batch_encoder_time / num_iterations)
        results['batched']['prompt_encoder'].append(batch_prompt_time / num_iterations)
        results['batched']['mask_decoder'].append(batch_decoder_time / num_iterations)
        results['batched']['total'].append(batch_total_time / num_iterations)
    
    return results

def plot_results(results):
    batch_sizes = results['batch_size']
    
    # Plot component times
    fig, axs = plt.subplots(2, 2, figsize=(15, 12))
    
    # Sequential vs Batch total
    axs[0, 0].plot(batch_sizes, results['sequential']['total'], 'o-', label='Sequential')
    axs[0, 0].plot(batch_sizes, results['batched']['total'], 's-', label='Batched')
    axs[0, 0].set_title('Total Processing Time')
    axs[0, 0].set_xlabel('Batch Size')
    axs[0, 0].set_ylabel('Time (seconds)')
    axs[0, 0].legend()
    axs[0, 0].grid(True)
    
    # Image encoder
    axs[0, 1].plot(batch_sizes, results['sequential']['image_encoder'], 'o-', label='Sequential')
    axs[0, 1].plot(batch_sizes, results['batched']['image_encoder'], 's-', label='Batched')
    axs[0, 1].set_title('Image Encoder Time')
    axs[0, 1].set_xlabel('Batch Size')
    axs[0, 1].set_ylabel('Time (seconds)')
    axs[0, 1].legend()
    axs[0, 1].grid(True)
    
    # Prompt encoder
    axs[1, 0].plot(batch_sizes, results['sequential']['prompt_encoder'], 'o-', label='Sequential')
    axs[1, 0].plot(batch_sizes, results['batched']['prompt_encoder'], 's-', label='Batched')
    axs[1, 0].set_title('Prompt Encoder Time')
    axs[1, 0].set_xlabel('Batch Size')
    axs[1, 0].set_ylabel('Time (seconds)')
    axs[1, 0].legend()
    axs[1, 0].grid(True)
    
    # Mask decoder
    axs[1, 1].plot(batch_sizes, results['sequential']['mask_decoder'], 'o-', label='Sequential')
    axs[1, 1].plot(batch_sizes, results['batched']['mask_decoder'], 's-', label='Batched')
    axs[1, 1].set_title('Mask Decoder Time')
    axs[1, 1].set_xlabel('Batch Size')
    axs[1, 1].set_ylabel('Time (seconds)')
    axs[1, 1].legend()
    axs[1, 1].grid(True)
    
    plt.tight_layout()
    plt.savefig('sam_timing_results.png')
    plt.show()
    
    # Breakdown chart
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Sequential breakdown
    seq_encoder = results['sequential']['image_encoder'][-1]
    seq_prompt = results['sequential']['prompt_encoder'][-1]
    seq_decoder = results['sequential']['mask_decoder'][-1]
    seq_total = seq_encoder + seq_prompt + seq_decoder
    
    seq_labels = ['Image Encoder', 'Prompt Encoder', 'Mask Decoder']
    seq_sizes = [seq_encoder/seq_total*100, seq_prompt/seq_total*100, seq_decoder/seq_total*100]
    
    ax1.pie(seq_sizes, labels=seq_labels, autopct='%1.1f%%', startangle=90)
    ax1.set_title('Sequential Processing Time Breakdown')
    
    # Batched breakdown
    batch_encoder = results['batched']['image_encoder'][-1]  
    batch_prompt = results['batched']['prompt_encoder'][-1]
    batch_decoder = results['batched']['mask_decoder'][-1]
    batch_total = batch_encoder + batch_prompt + batch_decoder
    
    batch_labels = ['Image Encoder', 'Prompt Encoder', 'Mask Decoder']
    batch_sizes = [batch_encoder/batch_total*100, batch_prompt/batch_total*100, batch_decoder/batch_total*100]
    
    ax2.pie(batch_sizes, labels=batch_labels, autopct='%1.1f%%', startangle=90)
    ax2.set_title('Batched Processing Time Breakdown')
    
    plt.tight_layout()
    plt.savefig('sam_time_breakdown.png')
    plt.show()

if __name__ == "__main__":
    print("Starting SAM component timing benchmark...")
    results = benchmark_sam(batch_sizes=[1, 2], num_iterations=3)
    plot_results(results)
    print("Benchmark complete!")