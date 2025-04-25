import torch
import torch.nn as nn
from torch.profiler import profile, record_function, ProfilerActivity
import numpy as np
import cv2
import time
import matplotlib.pyplot as plt
from segment_anything import SamPredictor, sam_model_registry
from segment_anything.utils.transforms import ResizeLongestSide

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

def profile_image_encoder_with_batch(model, batch_sizes=[1, 2, 4, 8], image_size=1024, device='cuda:0', num_warmup=3):
    model.eval()
    model.to(device)
    
    # Results storage
    batch_results = {}
    
    for batch_size in batch_sizes:
        print(f"\n=== Testing batch size: {batch_size} ===")
        
        # Create dummy input of appropriate size
        dummy_input = torch.randn(batch_size, 3, image_size, image_size, device=device)
        
        # Warmup runs
        print(f"Performing {num_warmup} warmup runs...")
        for _ in range(num_warmup):
            with torch.no_grad():
                _ = model.image_encoder(dummy_input)
                torch.cuda.synchronize()
        
        print("Warmup complete, starting profiling...")
        
        # Layer-wise timing setup
        layer_times = {}
        start_times = {}
        hooks = []
        
        def time_hook(name):
            def hook(module, input, output):
                if name not in layer_times:
                    layer_times[name] = []
                torch.cuda.synchronize()
                end_time = time.time()
                duration = end_time - start_times[name]
                layer_times[name].append(duration)
            return hook
        
        # Register hooks for all ViT blocks
        for name, module in model.image_encoder.named_modules():
            if isinstance(module, nn.ModuleList):  # ViT blocks
                for idx, block in enumerate(module):
                    block_name = f"block_{idx}"
                    
                    # Create pre-hook to record start time
                    def pre_hook_factory(block_name):
                        def pre_hook(module, input):
                            torch.cuda.synchronize()
                            start_times[block_name] = time.time()
                        return pre_hook
                    
                    hooks.append(block.register_forward_pre_hook(pre_hook_factory(block_name)))
                    hooks.append(block.register_forward_hook(time_hook(block_name)))
        
        # Profiling with torch.profiler
        with torch.no_grad():
            # Total time measurement
            torch.cuda.synchronize()
            total_start = time.time()
            
            with profile(
                activities=[ProfilerActivity.CUDA],
                record_shapes=True,
                with_flops=True,
                profile_memory=True,
            ) as prof:
                with record_function(f"batch_size_{batch_size}"):
                    output = model.image_encoder(dummy_input)
            
            torch.cuda.synchronize()
            total_time = time.time() - total_start
        
        # Remove hooks
        for hook in hooks:
            hook.remove()
        
        # Store results
        batch_results[batch_size] = {
            'layer_times': layer_times,
            'total_time': total_time,
            'profiler': prof
        }
        
        # Print summary
        print(f"Batch size {batch_size} - Total time: {total_time*1000:.2f} ms")
        print(f"Throughput: {batch_size/total_time:.2f} images/sec")
        
        # Layer timing summary
        print("\n--- Layer-wise Time Distribution ---")
        for name, times in layer_times.items():
            times_ms = np.array(times) * 1000  # Convert to milliseconds
            print(f"{name}: {times_ms.mean():.2f} ± {times_ms.std():.2f} ms")
    
    return batch_results

# Usage
batch_sizes = [1, 2, 4]  # Adjust based on your GPU memory
batch_profile_results = profile_image_encoder_with_batch(sam_model, 
                                                         batch_sizes=batch_sizes,
                                                         image_size=1024,  # SAM's default image size
                                                         device=device)


# Optional: Create comparison plots
def plot_batch_layer_timings(batch_results):
    batch_sizes = sorted(batch_results.keys())
    
    # Get consistent layer names across all batches
    all_layers = set()
    for batch_size in batch_sizes:
        all_layers.update(batch_results[batch_size]['layer_times'].keys())
    
    # Sort layers by block number
    layers = sorted(list(all_layers), key=lambda x: int(x.split('_')[1]))
    
    # Create a figure with subplots - one row per batch size
    fig, axes = plt.subplots(len(batch_sizes), 1, figsize=(12, 4*len(batch_sizes)), sharex=True)
    
    # If only one batch size, axes won't be an array
    if len(batch_sizes) == 1:
        axes = [axes]
    
    # Plot each batch size as a separate subplot
    for i, batch_size in enumerate(batch_sizes):
        ax = axes[i]
        layer_times = batch_results[batch_size]['layer_times']
        
        # Prepare data for this batch
        times = []
        for layer in layers:
            if layer in layer_times:
                times.append(np.mean(np.array(layer_times[layer]) * 1000))  # ms
            else:
                times.append(0)  # Layer not present in this batch
        
        # Plot horizontal bar chart
        ax.barh(layers, times, color='steelblue')
        ax.set_ylabel('Layer')
        ax.set_title(f'Batch Size: {batch_size}')
        ax.grid(axis='x', linestyle='--', alpha=0.7)
    
    # Add common xlabel
    fig.text(0.5, 0.04, 'Time (ms)', ha='center', va='center', fontsize=12)
    fig.suptitle('Layer Execution Times by Batch Size', fontsize=16)
    
    plt.tight_layout(rect=[0, 0.05, 1, 0.97])  # Adjust for the common xlabel
    plt.savefig('benchmarking/batch_layer_timings.png')
    plt.show()
    
    # Also create a heatmap for comparison
    plt.figure(figsize=(12, 8))
    data = np.zeros((len(layers), len(batch_sizes)))
    
    for j, batch_size in enumerate(batch_sizes):
        layer_times = batch_results[batch_size]['layer_times']
        for i, layer in enumerate(layers):
            if layer in layer_times:
                data[i, j] = np.mean(np.array(layer_times[layer]) * 1000)  # ms
    
    plt.imshow(data, aspect='auto', interpolation='nearest', cmap='viridis')
    plt.colorbar(label='Time (ms)')
    plt.yticks(range(len(layers)), layers)
    plt.xticks(range(len(batch_sizes)), [f'Batch {b}' for b in batch_sizes])
    plt.xlabel('Batch Size')
    plt.ylabel('Layer')
    plt.title('Layer Timing Heatmap Across Batch Sizes')
    plt.tight_layout()
    plt.savefig('benchmarking/batch_layer_heatmap.png')
    plt.show()


# Plot the results
# After running batch_profile_results = profile_image_encoder_with_batch(...)
plot_batch_layer_timings(batch_profile_results)

def profile_image_encoder_layers(model, input_image, device, num_warmup=5):
    model.eval()
    model.to(device)
    
    print(f"Performing {num_warmup} warmup runs...")
    # Warmup runs
    for _ in range(num_warmup):
        with torch.no_grad():
            _ = model.image_encoder(input_image)
            torch.cuda.synchronize()
    
    print("Warmup complete, starting profiling...")
    
    # Hook to track layer-wise timings
    layer_times = {}
    
    def time_hook(name):
        def hook(module, input, output):
            if name not in layer_times:
                layer_times[name] = []
            torch.cuda.synchronize()  # Make sure previous ops are completed
            end_time = time.time()
            duration = end_time - start_times[name]
            layer_times[name].append(duration)
        return hook
    
    # Register hooks for all ViT blocks
    hooks = []
    start_times = {}  # Dictionary to store start times
    
    for name, module in model.image_encoder.named_modules():
        if isinstance(module, nn.ModuleList):  # ViT blocks
            for idx, block in enumerate(module):
                block_name = f"block_{idx}"
                
                # Create a pre-hook to record start time
                def pre_hook_factory(block_name):
                    def pre_hook(module, input):
                        torch.cuda.synchronize()
                        start_times[block_name] = time.time()
                    return pre_hook
                
                hooks.append(block.register_forward_pre_hook(pre_hook_factory(block_name)))
                hooks.append(block.register_forward_hook(time_hook(block_name)))
    
    # Run profiling
    with torch.no_grad():
        with profile(
            activities=[ProfilerActivity.CUDA],
            record_shapes=True,
            with_flops=True,
        ) as prof:
            with record_function("full_forward_pass"):
                torch.cuda.synchronize()  # Synchronize before profiling
                output = model.image_encoder(input_image)
                torch.cuda.synchronize()  # Make sure all ops complete before ending profiling
    
    # Remove hooks
    for hook in hooks:
        hook.remove()
    
    # Print layer-wise stats
    print("\n--- Layer-wise Time Distribution ---")
    for name, times in layer_times.items():
        times_ms = np.array(times) * 1000  # Convert to milliseconds
        print(f"{name}: {times_ms.mean():.2f} ± {times_ms.std():.2f} ms")
    
    # Print full profiler summary
    print("\n--- Full Profiler Summary ---")
    print(prof.key_averages().table(sort_by="cuda_time_total"))
    
    return layer_times

def plot_layer_timings(layer_times):
    # Convert to milliseconds and calculate means
    layers = []
    times = []
    
    for name, time_list in layer_times.items():
        layers.append(name)
        times.append(np.mean(np.array(time_list) * 1000))  # Convert to ms
    
    # Sort by layer number
    sorted_indices = sorted(range(len(layers)), key=lambda i: int(layers[i].split('_')[1]))
    sorted_layers = [layers[i] for i in sorted_indices]
    sorted_times = [times[i] for i in sorted_indices]
    
    # Create horizontal bar chart
    plt.figure(figsize=(10, 8))
    plt.barh(sorted_layers, sorted_times, color='steelblue')
    plt.xlabel('Time (ms)')
    plt.ylabel('Layer')
    plt.title('Per-Layer Execution Time')
    plt.grid(axis='x', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig('benchmarking/layer_timings_bar.png')
    plt.show()

# Usage (add to main script after model initialization)
layer_stats = profile_image_encoder_layers(sam_model, input_image, device)
plot_layer_timings(layer_stats)