import torch
import numpy as np

def test_cuda_availability():
    """Test if CUDA is available and print device information."""
    print("CUDA available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("CUDA device count:", torch.cuda.device_count())
        print("Current CUDA device:", torch.cuda.current_device())
        print("CUDA device name:", torch.cuda.get_device_name(0))

def test_simple_cuda_operation():
    """Perform a simple vector addition using CUDA."""
    # Create two random vectors
    x = torch.randn(1000, device='cuda')
    y = torch.randn(1000, device='cuda')
    
    # Perform addition
    z = x + y
    
    # Move result back to CPU for verification
    z_cpu = z.cpu()
    
    # Verify the result
    x_cpu = x.cpu()
    y_cpu = y.cpu()
    expected = x_cpu + y_cpu
    
    # Check if results match
    is_correct = torch.allclose(z_cpu, expected)
    print("Vector addition test passed:", is_correct)
    
    return is_correct

if __name__ == "__main__":
    print("Running CUDA tests...")
    test_cuda_availability()
    test_simple_cuda_operation()