import os
import torch
from torch.utils.cpp_extension import load

# JIT compile the CUDA extension
_module = None
_load_error = None


def _load_cuda_module():
    """
    Best-effort CUDA extension loading.

    This is intentionally environment-agnostic:
    - no hardcoded CUDA/MSVC paths
    - respects user/system toolchain configuration
    - clean fallback to PyTorch scatter when build tools are unavailable
    """
    curr_dir = os.path.dirname(os.path.abspath(__file__))
    cuda_src = os.path.join(curr_dir, "blend.cu")

    if not os.path.exists(cuda_src):
        return None, "blend.cu not found"

    if not torch.cuda.is_available():
        return None, "CUDA is not available"

    # Allow users/CI to opt out of JIT compilation.
    if os.environ.get("CACHEBLEND_DISABLE_CUDA_EXT", "").lower() in {"1", "true", "yes"}:
        return None, "CUDA extension disabled by CACHEBLEND_DISABLE_CUDA_EXT"

    try:
        mod = load(
            name="blend_cuda",
            sources=[cuda_src],
            verbose=os.environ.get("CACHEBLEND_VERBOSE_EXT", "0") == "1",
        )
        return mod, None
    except Exception as exc:
        return None, str(exc)


_module, _load_error = _load_cuda_module()
if _module is None:
    print(f"Failed to load CUDA blend kernel: {_load_error}")
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
