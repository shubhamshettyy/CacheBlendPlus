import os
import torch

# Windows Environment Setup for JIT compilation - MUST happen before importing cpp_extension
if os.name == "nt":
    os.environ["CUDA_HOME"] = r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.4"
    cl_path = r"C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Tools\MSVC\14.43.34808\bin\Hostx64\x64"
    if cl_path not in os.environ["PATH"]:
        os.environ["PATH"] = cl_path + os.pathsep + os.environ["PATH"]

from torch.utils.cpp_extension import load

# JIT compile the CUDA extension
_module = None
try:
    # Get the directory of the current file
    _curr_dir = os.path.dirname(os.path.abspath(__file__))
    _cuda_src = os.path.join(_curr_dir, "blend.cu")
    
    if os.path.exists(_cuda_src):
        _module = load(
            name="blend_cuda",
            sources=[_cuda_src],
            verbose=True
        )
except Exception as e:
    print(f"Failed to load CUDA blend kernel: {e}")
    print("Falling back to PyTorch implementation.")

def blend(cached_kv, new_values, indices):
    """
    Blends newly computed KV values into the cached KV tensor at the specified indices.
    
    Args:
        cached_kv: Tensor of shape (L, 2, N, H, D) - The baseline KV cache.
        new_values: Tensor of shape (L, 2, k, H, D) - The recomputed KV values.
        indices: Tensor of shape (k,) - The token indices that were recomputed.
        
    Returns:
        The updated cached_kv tensor.
    """
    if _module is not None and cached_kv.is_cuda and new_values.is_cuda and indices.is_cuda:
        # Use the optimized CUDA kernel
        # Note: the kernel modifies cached_kv in-place
        _module.launch_blend(cached_kv, new_values, indices)
        return cached_kv
    else:
        # Fallback to PyTorch scatter update
        # In-place scatter update across the token dimension (dim=2)
        cached_kv[:, :, indices, :, :] = new_values
        return cached_kv
