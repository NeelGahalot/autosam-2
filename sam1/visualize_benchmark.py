import matplotlib.pyplot as plt
import numpy as np

def create_benchmark_dashboard(benchmark_results, memory_data=None):
    """
    Create a dashboard with multiple plots showing benchmark results
    
    Args:
        benchmark_results: Dictionary of benchmark results
        memory_data: Optional dictionary with memory usage data
    """
    # Create figure with subplots
    fig, axs = plt.subplots(2, 2, figsize=(18, 12))
    
    # Extract data
    batch_sizes = sorted(benchmark_results['batch'].keys())
    processing_times = [benchmark_results['batch'][size]['per_image_time'] for size in batch_sizes]
    batch_times = [benchmark_results['batch'][size]['avg_batch_time'] for size in batch_sizes]
    
    # Add sequential processing as batch size 1
    if 1 not in batch_sizes and 'sequential' in benchmark_results:
        seq_time = benchmark_results['sequential']['avg_per_image']
    else:
        seq_time = None
    
    # Calculate speedups
    if seq_time:
        speedups = [seq_time / benchmark_results['batch'][size]['per_image_time'] 
                    for size in batch_sizes]
    else:
        speedups = [1.0] + [processing_times[0] / time for time in processing_times[1:]]
    
    # 1. Processing time per image
    axs[0, 0].plot(batch_sizes, processing_times, 'o-', linewidth=2, markersize=8)
    if seq_time:
        axs[0, 0].axhline(y=seq_time, color='r', linestyle='--', 
                        label=f'Sequential: {seq_time:.4f}s')
        axs[0, 0].legend()
    axs[0, 0].set_xlabel('Batch Size')
    axs[0, 0].set_ylabel('Time per Image (seconds)')
    axs[0, 0].set_title('Processing Time per Image vs Batch Size')
    axs[0, 0].grid(True, linestyle='--', alpha=0.7)
    axs[0, 0].set_xticks(batch_sizes)
    
    # 2. Speedup factor
    bars = axs[0, 1].bar(range(len(batch_sizes)), speedups, width=0.7)
    axs[0, 1].axhline(y=1.0, color='r', linestyle='--', label='No speedup')
    axs[0, 1].legend()
    
    # Add values on top of bars
    for bar, speedup in zip(bars, speedups):
        axs[0, 1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                      f'{speedup:.2f}x', ha='center', fontsize=10)
    
    axs[0, 1].set_xlabel('Batch Size')
    axs[0, 1].set_ylabel('Speedup Factor (×)')
    axs[0, 1].set_title('Speedup Factor vs Batch Size')
    axs[0, 1].set_xticks(range(len(batch_sizes)))
    axs[0, 1].set_xticklabels(batch_sizes)
    axs[0, 1].grid(True, axis='y', linestyle='--', alpha=0.7)
    
    # 3. Total batch processing time
    bars = axs[1, 0].bar(range(len(batch_sizes)), batch_times, width=0.7)
    
    # Add values on top of bars
    for bar, time in zip(bars, batch_times):
        axs[1, 0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                      f'{time:.4f}s', ha='center', fontsize=10)
    
    axs[1, 0].set_xlabel('Batch Size')
    axs[1, 0].set_ylabel('Total Batch Time (seconds)')
    axs[1, 0].set_title('Total Batch Processing Time')
    axs[1, 0].set_xticks(range(len(batch_sizes)))
    axs[1, 0].set_xticklabels(batch_sizes)
    axs[1, 0].grid(True, axis='y', linestyle='--', alpha=0.7)
    
    # 4. Memory usage if available
    if memory_data:
        memory_batch_sizes = sorted(memory_data.keys())
        memory_usage = [memory_data[size] for size in memory_batch_sizes]
        
        axs[1, 1].plot(memory_batch_sizes, memory_usage, 'o-', linewidth=2, markersize=8, color='green')
        axs[1, 1].set_xlabel('Batch Size')
        axs[1, 1].set_ylabel('GPU Memory Usage (MB)')
        axs[1, 1].set_title('GPU Memory Usage vs Batch Size')
        axs[1, 1].grid(True, linestyle='--', alpha=0.7)
        axs[1, 1].set_xticks(memory_batch_sizes)
        
        # Add values next to points
        for i, mem in enumerate(memory_usage):
            axs[1, 1].text(memory_batch_sizes[i], mem + 20, f'{mem} MB', ha='center')
    else:
        # If no memory data, show some other useful information
        axs[1, 1].axis('off')
        text = "Benchmark Summary:\n\n"
        text += f"Model: {benchmark_results.get('model_type', 'SAM')}\n"
        text += f"Device: {benchmark_results.get('device', 'GPU')}\n\n"
        
        if seq_time:
            text += f"Sequential time: {seq_time:.4f}s per image\n\n"
        
        text += "Batch Results:\n"
        for batch_size in batch_sizes:
            text += f"- Batch {batch_size}: {benchmark_results['batch'][batch_size]['per_image_time']:.4f}s per image\n"
            
        axs[1, 1].text(0.1, 0.5, text, fontsize=12, va='center')
    
    plt.tight_layout()
    #plt.savefig('sam_benchmark_dashboard.png', dpi=300)
    plt.show()
    
def plot_total_batch_processing_time(benchmark_results):
    """
    Plot total processing time for different batch sizes
    
    Args:
        benchmark_results: Dictionary of benchmark results
    """
    # Extract data
    batch_sizes = sorted(benchmark_results['batch'].keys())
    batch_times = [benchmark_results['batch'][size]['avg_batch_time'] for size in batch_sizes]
    
    # Create figure
    plt.figure(figsize=(10, 6))
    
    # Plot bars
    bars = plt.bar(range(len(batch_sizes)), batch_times, width=0.7)
    
    # Add values on top of bars
    for bar, time in zip(bars, batch_times):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f'{time:.4f}s', ha='center', fontsize=10)
    
    # Labels and title
    plt.xlabel('Batch Size', fontsize=12)
    plt.ylabel('Total Batch Processing Time (seconds)', fontsize=12)
    plt.title('SAM Encoder: Total Batch Processing Time vs Batch Size', fontsize=14)
    
    # Set x-ticks to batch sizes
    plt.xticks(range(len(batch_sizes)), batch_sizes)
    
    # Grid
    plt.grid(True, axis='y', linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    #plt.savefig('sam_total_batch_time.png', dpi=300)
    plt.show()
    
def plot_speedup_vs_batch_size(benchmark_results):
    """
    Plot speedup factor vs batch size
    
    Args:
        benchmark_results: Dictionary of benchmark results
    """
    # Extract data
    batch_sizes = sorted(benchmark_results['batch'].keys())
    
    # Calculate speedups
    sequential_time = benchmark_results['sequential']['avg_per_image']
    speedups = [sequential_time / benchmark_results['batch'][size]['per_image_time'] 
                for size in batch_sizes]
    
    # Create figure
    plt.figure(figsize=(10, 6))
    
    # Plot bars
    bars = plt.bar(range(len(batch_sizes)), speedups, width=0.7)
    
    # Add horizontal line at y=1 (no speedup)
    plt.axhline(y=1.0, color='r', linestyle='--', label='No speedup (sequential)')
    
    # Add values on top of bars
    for bar, speedup in zip(bars, speedups):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                f'{speedup:.2f}x', ha='center', fontsize=10)
    
    # Labels and title
    plt.xlabel('Batch Size', fontsize=12)
    plt.ylabel('Speedup Factor (×)', fontsize=12)
    plt.title('SAM Encoder: Speedup Factor vs Batch Size', fontsize=14)
    
    # Set x-ticks to batch sizes
    plt.xticks(range(len(batch_sizes)), batch_sizes)
    
    # Grid and legend
    plt.grid(True, axis='y', linestyle='--', alpha=0.7)
    plt.legend()
    
    plt.tight_layout()
    #plt.savefig('sam_speedup_vs_batch_size.png', dpi=300)
    plt.show()
    
def plot_processing_time_vs_batch_size(benchmark_results):
    """
    Plot processing time per image vs batch size
    
    Args:
        benchmark_results: Dictionary of benchmark results
    """
    # Extract data
    batch_sizes = sorted(benchmark_results['batch'].keys())
    processing_times = [benchmark_results['batch'][size]['per_image_time'] for size in batch_sizes]
    
    # Add sequential processing as batch size 1
    if 1 not in batch_sizes and 'sequential' in benchmark_results:
        batch_sizes = [1] + batch_sizes
        processing_times = [benchmark_results['sequential']['avg_per_image']] + processing_times
    
    # Create figure
    plt.figure(figsize=(10, 6))
    
    # Plot line
    plt.plot(batch_sizes, processing_times, 'o-', linewidth=2, markersize=8)
    
    # Add horizontal line for sequential time
    if 'sequential' in benchmark_results:
        seq_time = benchmark_results['sequential']['avg_per_image']
        plt.axhline(y=seq_time, color='r', linestyle='--', 
                    label=f'Sequential: {seq_time:.4f}s')
    
    # Labels and title
    plt.xlabel('Batch Size', fontsize=12)
    plt.ylabel('Time per Image (seconds)', fontsize=12)
    plt.title('SAM Encoder: Processing Time per Image vs Batch Size', fontsize=14)
    
    # Set x-ticks to batch sizes
    plt.xticks(batch_sizes)
    
    # Grid and legend
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    
    plt.tight_layout()
    #plt.savefig('sam_time_vs_batch_size.png', dpi=300)
    plt.show()