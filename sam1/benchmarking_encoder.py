import os
import time
import torch
import numpy as np
from PIL import Image
from torch.utils.data import DataLoader
from segment_anything import sam_model_registry
from segment_anything.utils.transforms import ResizeLongestSide
from torch.profiler import profile, record_function, ProfilerActivity
from visualize_benchmark import *
    
class SAMEncoderBenchmark:
    def __init__(self, model_type, checkpoint, device="cuda"):
        """
        Initialize the SAM Encoder Benchmark tool
        
        Args:
            model_type: Type of SAM model ('vit_h', 'vit_l', 'vit_b')
            checkpoint: Path to model checkpoint
            device: Device to run model on
        """
        self.device = torch.device(device if torch.cuda.is_available() and device == "cuda" else "cpu")
        
        # Load SAM model
        self.sam_model = sam_model_registry[model_type](checkpoint=checkpoint)
        self.sam_model.to(self.device)
        self.sam_model.eval()
        
        # Create transform for resizing
        self.transform = ResizeLongestSide(self.sam_model.image_encoder.img_size)

    def visualize_results(self, benchmark_results, memory_data=None, output_dir='benchmarking'):
        """
        Visualize benchmark results with various plots
        
        Args:
            benchmark_results: Dictionary of benchmark results
            memory_data: Optional dictionary with memory usage data
            output_dir: Directory to save plots, default is current directory
        """
        import matplotlib.pyplot as plt
        
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        else:
            output_dir = "."
        
        # Individual plots
        plot_processing_time_vs_batch_size(benchmark_results)
        plt.savefig(os.path.join(output_dir, 'sam_time_vs_batch_size.png'), dpi=300)
        
        plot_speedup_vs_batch_size(benchmark_results)
        plt.savefig(os.path.join(output_dir, 'sam_speedup_vs_batch_size.png'), dpi=300)
        
        plot_total_batch_processing_time(benchmark_results)
        plt.savefig(os.path.join(output_dir, 'sam_total_batch_time.png'), dpi=300)
        
        if memory_data:
            plot_memory_usage(benchmark_results, memory_data)
            plt.savefig(os.path.join(output_dir, 'sam_memory_usage.png'), dpi=300)
        
        # Dashboard
        create_benchmark_dashboard(benchmark_results, memory_data)
        plt.savefig(os.path.join(output_dir, 'sam_benchmark_dashboard.png'), dpi=300)
        
        print(f"Visualization complete. Plots saved to {output_dir}")    
        
    def preprocess_image(self, image):
        """
        Preprocess a single image for SAM model
        
        Args:
            image: PIL image, numpy array or tensor
            
        Returns:
            preprocessed_image: Tensor ready for the image encoder
            original_size: Original image size
            input_size: Transformed image size
        """
        # Handle different input types
        if isinstance(image, torch.Tensor):
            # Denormalize if necessary and convert to numpy
            # Assuming image is [C, H, W] with values in range used by torchvision normalization
            image_np = image.permute(1, 2, 0).cpu().numpy()
            
            # If values are in normalized range (approximately [-1, 1] or similar)
            # we need to convert back to [0, 255] range for SAM preprocessing
            if image_np.min() < 0 or image_np.max() > 1.5:
                # Assuming standard ImageNet normalization
                mean = np.array([0.485, 0.456, 0.406])
                std = np.array([0.229, 0.224, 0.225])
                
                image_np = image_np * std + mean
                image_np = np.clip(image_np * 255, 0, 255).astype(np.uint8)
        elif isinstance(image, Image.Image):
            image_np = np.array(image)
        else:
            # Assume it's already a numpy array
            image_np = image
            
        # Store original size
        original_size = image_np.shape[:2]
        
        # Apply transform
        transformed = self.transform.apply_image(image_np)
        transformed_torch = torch.as_tensor(transformed, device=self.device)
        transformed_torch = transformed_torch.permute(2, 0, 1).contiguous()[None, :, :, :]
        
        # Preprocess
        preprocessed_image = self.sam_model.preprocess(transformed_torch)
        input_size = transformed_torch.shape[-2:]
        
        return preprocessed_image, original_size, input_size
    
    def preprocess_batch(self, images):
        """
        Preprocess a batch of images for SAM model
        
        Args:
            images: Tensor of shape [B, C, H, W]
            
        Returns:
            preprocessed_batch: Tensor ready for the image encoder
            original_sizes: List of original image sizes
            input_sizes: List of transformed image sizes
        """
        batch_size = images.shape[0]
        original_sizes = []
        transformed_images = []
        
        # Process each image in the batch
        for i in range(batch_size):
            img = images[i].cpu()
            
            # If normalized, denormalize back to [0, 255] range
            if img.min() < 0 or img.max() > 1.5:
                # Assuming standard ImageNet normalization
                mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
                std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
                
                img = img * std + mean
                img = torch.clamp(img * 255, 0, 255).to(torch.uint8)
                
            # Convert to numpy for SAM transform
            img_np = img.permute(1, 2, 0).numpy()
            original_sizes.append(img_np.shape[:2])
            
            # Apply transform
            transformed = self.transform.apply_image(img_np)
            transformed_torch = torch.as_tensor(transformed, device=self.device)
            transformed_torch = transformed_torch.permute(2, 0, 1).contiguous()
            transformed_images.append(transformed_torch)
        
        # Stack into batch
        batch_tensor = torch.stack(transformed_images, dim=0)
        input_sizes = [tensor.shape[-2:] for tensor in transformed_images]
        
        # Preprocess batch
        preprocessed_batch = self.sam_model.preprocess(batch_tensor)
        
        return preprocessed_batch, original_sizes, input_sizes
    
    def encode_images(self, preprocessed_batch):
        """
        Run the image encoder on a batch of preprocessed images
        
        Args:
            preprocessed_batch: Tensor of shape [B, C, H, W]
            
        Returns:
            image_embeddings: Image embeddings from the SAM model
        """
        with torch.no_grad():
            image_embeddings = self.sam_model.image_encoder(preprocessed_batch)
        return image_embeddings
    
    def benchmark_sequential(self, dataset, num_images=10):
        """
        Benchmark sequential processing (one image at a time)
        
        Args:
            dataset: PyTorch dataset object
            num_images: Number of images to process
            
        Returns:
            times: List of processing times for each image
            total_time: Total processing time
        """
        dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
        times = []
        
        # Get first image for warm-up
        first_image, _ = next(iter(dataloader))
        first_image = first_image[0]  # Get single image from batch
        
        # Warm-up run
        print("Performing warm-up run...")
        preprocessed, _, _ = self.preprocess_image(first_image)
        with torch.no_grad():
            _ = self.sam_model.image_encoder(preprocessed)
        
        if self.device.type == "cuda":
            torch.cuda.synchronize()
        
        print("Warm-up complete, starting benchmark...")
        
        # Reset dataloader for actual benchmark
        dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
        total_start = time.time()
        
        for i, (image, _) in enumerate(dataloader):
            if i >= num_images:
                break
                
            image = image[0]  # Get single image from batch
            
            # Process image
            # Preprocess
            preprocessed, _, _ = self.preprocess_image(image)
            
            start = time.time()
            # Encode
            with torch.no_grad():
                _ = self.sam_model.image_encoder(preprocessed)
                
            # Synchronize if using CUDA
            if self.device.type == "cuda":
                torch.cuda.synchronize()
                
            end = time.time()
            times.append(end - start)
            
            print(f"Sequential image {i+1}/{num_images}: {times[-1]:.4f}s")
        
        total_time = time.time() - total_start
        return times, total_time
    
    def benchmark_batch(self, dataset, batch_sizes=[2,4,8,16]):
        """
        Benchmark batch processing with different batch sizes
        
        Args:
            dataset: PyTorch dataset object
            batch_sizes: List of batch sizes to test
            
        Returns:
            results: Dictionary of results for each batch size
        """
        results = {}
        
        for batch_size in batch_sizes:
            # Skip batch sizes that are too large
            if batch_size > len(dataset):
                print(f"Skipping batch size {batch_size}: larger than dataset")
                continue
                
            dataloader = DataLoader(
                dataset, 
                batch_size=batch_size, 
                shuffle=False,
                num_workers=2,
                pin_memory=True
            )
            
            # Get first batch
            images, _ = next(iter(dataloader))
            
            # Skip warm-up run
            preprocessed_batch, _, _ = self.preprocess_batch(images)
            _ = self.encode_images(preprocessed_batch)
            
            # Benchmark runs
            num_runs = 5
            batch_times = []
            
            for run in range(num_runs):
                if self.device.type == "cuda":
                    torch.cuda.synchronize()
                
                # Preprocess batch
                preprocessed_batch, _, _ = self.preprocess_batch(images)
                start = time.time()
                # Encode batch
                _ = self.encode_images(preprocessed_batch)
                
                if self.device.type == "cuda":
                    torch.cuda.synchronize()
                    
                batch_time = time.time() - start
                batch_times.append(batch_time)
                
                print(f"Batch size {batch_size}, run {run+1}/{num_runs}: {batch_time:.4f}s")
            
            # Calculate statistics
            avg_time = sum(batch_times) / len(batch_times)
            per_image_time = avg_time / batch_size
            
            results[batch_size] = {
                "avg_batch_time": avg_time,
                "per_image_time": per_image_time,
                "times": batch_times
            }
            
            print(f"Batch size {batch_size} results:")
            print(f"  Average batch time: {avg_time:.4f}s")
            print(f"  Per image time: {per_image_time:.4f}s")
            if batch_size > 1 and 1 in results:
                print(f"  Speedup vs sequential: {results[1]['per_image_time'] / per_image_time:.2f}x")
            print("-" * 40)
        
        return results

    def benchmark_with_profiling(self, dataset, batch_size=4, num_runs=5):
        """
        Benchmark with detailed profiling
        
        Args:
            dataset: PyTorch dataset object
            batch_size: Batch size to test
            num_runs: Number of profiling runs
            
        Returns:
            prof: The profiler object containing results
        """
        dataloader = DataLoader(
            dataset, 
            batch_size=batch_size, 
            shuffle=False,
            num_workers=2,
            pin_memory=True
        )
        
        # Get first batch
        images, _ = next(iter(dataloader))
        
        # Warm-up run
        preprocessed_batch, _, _ = self.preprocess_batch(images)
        _ = self.encode_images(preprocessed_batch)
        
        # Setup profiler
        with profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            record_shapes=True,
            profile_memory=True,
            with_stack=True
        ) as prof:
            for run in range(num_runs):
                if self.device.type == "cuda":
                    torch.cuda.synchronize()
                
                with record_function("full_batch_processing"):
                    # Preprocess batch
                    with record_function("preprocessing"):
                        preprocessed_batch, _, _ = self.preprocess_batch(images)
                    
                    # Encode batch
                    with record_function("encoding"):
                        _ = self.encode_images(preprocessed_batch)
                
                if self.device.type == "cuda":
                    torch.cuda.synchronize()
                
                prof.step()
        
        return prof
    
    def run_comprehensive_benchmark(self, dataset, num_sequential=10):
        """
        Run a comprehensive benchmark comparing sequential vs batch processing
        
        Args:
            dataset: PyTorch dataset object
            num_sequential: Number of images for sequential benchmark
            
        Returns:
            benchmark_results: Dictionary of benchmark results
        """
        print("Running sequential benchmark...")
        sequential_times, sequential_total = self.benchmark_sequential(dataset, num_sequential)
        
        print("\nRunning batch benchmarks...")
        batch_results = self.benchmark_batch(dataset)
        
        # Compile results
        benchmark_results = {
            "sequential": {
                "per_image_times": sequential_times,
                "avg_per_image": sum(sequential_times) / len(sequential_times),
                "total_time": sequential_total
            },
            "batch": batch_results
        }
        
        # Print summary
        print("\nBenchmark Summary:")
        print(f"Sequential processing average: {benchmark_results['sequential']['avg_per_image']:.4f}s per image")
        
        for batch_size, result in batch_results.items():
            speedup = benchmark_results['sequential']['avg_per_image'] / result['per_image_time']
            print(f"Batch size {batch_size}: {result['per_image_time']:.4f}s per image (speedup: {speedup:.2f}x)")
        
        return benchmark_results

# Simple dataset class to only load images (no need for masks when benchmarking encoder)
class SimpleImageDataset(torch.utils.data.Dataset):
    def __init__(self, image_dir, extensions=('.jpg', '.jpeg', '.png'), transform=None):
        """
        A simple dataset that loads images from a directory
        
        Args:
            image_dir: Directory containing images
            extensions: Tuple of valid file extensions
            transform: Transform to apply to images
        """
        self.image_dir = image_dir
        self.transform = transform
        
        # Get all image files
        self.image_files = []
        for file in os.listdir(image_dir):
            if file.lower().endswith(extensions):
                self.image_files.append(os.path.join(image_dir, file))
        
        if not self.image_files:
            raise FileNotFoundError(f"No images found in {image_dir}")
    
    def __getitem__(self, index):
        # Load image
        img_path = self.image_files[index]
        img = Image.open(img_path).convert('RGB')
        
        if self.transform:
            img = self.transform(img)
        
        # Return dummy mask (not used for encoding benchmark)
        mask = torch.zeros((1, 1, 1))
        
        return img, mask
    
    def __len__(self):
        return len(self.image_files)


# Usage example
if __name__ == "__main__":
    import argparse
    from torchvision import transforms
    
    parser = argparse.ArgumentParser(description="Benchmark SAM image encoder with batch processing")
    parser.add_argument("--image_dir", type=str, default="/teamspace/studios/this_studio/greenstand_segmentation_model_build/segmentation_model_build/Deeplab/inhouse/samples", help="Directory containing images")
    parser.add_argument("--checkpoint", type=str, default="/teamspace/studios/this_studio/greenstand_segmentation_model_build/segmentation_model_build/segment-anything/sam_vit_b_01ec64.pth", help="Path to SAM model checkpoint")
    parser.add_argument("--model_type", type=str, default="vit_b", choices=["vit_h", "vit_l", "vit_b"], help="SAM model type")
    parser.add_argument("--device", type=str, default="cuda", help="Device to run on")
    parser.add_argument("--batch_sizes", type=int, nargs="+", default=[4], help="Batch sizes to test")
    parser.add_argument("--num_sequential", type=int, default=10, help="Number of images for sequential benchmark")
    
    args = parser.parse_args()
    
    # Define transform
    transform = transforms.Compose([
        transforms.Resize((1024, 1024)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Create dataset
    dataset = SimpleImageDataset(
        image_dir=args.image_dir,
        transform=transform
    )
    
    print(f"Found {len(dataset)} images in {args.image_dir}")
    
    # Initialize benchmark
    benchmark = SAMEncoderBenchmark(
        model_type=args.model_type,
        checkpoint=args.checkpoint,
        device=args.device
    )
    
    # Run benchmark
    results = benchmark.run_comprehensive_benchmark(dataset, args.num_sequential)
    benchmark.visualize_results(results)
    # Print speedup table
    print("\nSpeedup Table:")
    print("Batch Size | Time per Image (s) | Speedup vs Sequential")
    print("-" * 55)
    
    sequential_time = results['sequential']['avg_per_image']
    for batch_size in sorted(results['batch'].keys()):
        batch_result = results['batch'][batch_size]
        speedup = sequential_time / batch_result['per_image_time']
        print(f"{batch_size:^10} | {batch_result['per_image_time']:.6f} | {speedup:.2f}x")

    '''
    # Run profiling
    prof = benchmark.benchmark_with_profiling(dataset, batch_size=4)
    
    # Print profiling results
    print("CPU Time:")
    print(prof.key_averages(group_by_input_shape=True).table(
        sort_by="cpu_time_total", 
        row_limit=20,
        top_level_events_only=True
    ))
    
    print("\nCUDA Time:")
    print(prof.key_averages(group_by_input_shape=True).table(
        sort_by="cuda_time_total", 
        row_limit=20,
        top_level_events_only=True
    ))
    
    print("\nCUDA Memory:")
    print(prof.key_averages(group_by_input_shape=True).table(
        sort_by="self_cuda_memory_usage", 
        row_limit=20,
        top_level_events_only=True
    ))

    
    # Print speedup table
    print("\nSpeedup Table:")
    print("Batch Size | Time per Image (s) | Speedup vs Sequential")
    print("-" * 55)
    '''

    