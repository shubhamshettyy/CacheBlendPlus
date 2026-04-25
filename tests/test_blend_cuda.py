import torch
import pytest
import os
import sys

# Add the project root to sys.path to import cacheblendplus
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from cacheblendplus.blend_kernel import blend

@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_blend_cuda_correctness():
    # Setup dimensions
    L, N, H, D = 4, 128, 8, 64  # Layers, SeqLen, Heads, HeadDim
    k = 20  # Number of tokens to recompute/blend
    
    device = "cuda"
    dtype = torch.float16
    
    # 1. Create base cache (zeros)
    cached_kv = torch.zeros((L, 2, N, H, D), device=device, dtype=dtype)
    
    # 2. Create new values (ones)
    new_values = torch.ones((L, 2, k, H, D), device=device, dtype=dtype)
    
    # 3. Pick random indices to update
    indices = torch.sort(torch.randperm(N)[:k]).values.to(device=device, dtype=torch.int64)
    
    # 4. Clone for reference (PyTorch baseline)
    reference_cache = cached_kv.clone()
    reference_cache[:, :, indices, :, :] = new_values
    
    # 5. Run our optimized blend
    # Note: our implementation should modify cached_kv in-place
    result_cache = blend(cached_kv, new_values, indices)
    
    # 6. Compare
    assert torch.allclose(result_cache, reference_cache), "CUDA blend output does not match PyTorch baseline!"
    assert torch.allclose(cached_kv, reference_cache), "In-place update failed!"
    
    print("\n✓ CUDA Blend correctness verified.")

@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_blend_speed():
    # Larger dimensions for meaningful timing
    L, N, H, D = 32, 2048, 32, 128
    k = 300
    
    device = "cuda"
    dtype = torch.float16
    
    cached_kv = torch.randn((L, 2, N, H, D), device=device, dtype=dtype)
    new_values = torch.randn((L, 2, k, H, D), device=device, dtype=dtype)
    indices = torch.sort(torch.randperm(N)[:k]).values.to(device=device, dtype=torch.int64)
    
    # Warmup
    for _ in range(5):
        blend(cached_kv, new_values, indices)
    
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    
    start.record()
    for _ in range(100):
        blend(cached_kv, new_values, indices)
    end.record()
    
    torch.cuda.synchronize()
    print(f"\nAverage Blend Time (100 runs): {start.elapsed_time(end)/100:.4f} ms")

if __name__ == "__main__":
    test_blend_cuda_correctness()
    test_blend_speed()
