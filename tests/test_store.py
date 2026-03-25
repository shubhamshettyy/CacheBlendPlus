import torch
import pytest
from transformers import DynamicCache
from cacheblendplus.kv_store import pack_kv, unpack_kv, KVCacheStore

# Dummy dimensions for a tiny mock model
L = 4       # num_layers
N = 16      # seq_len
H = 8       # num_heads
D = 64      # head_dim
BATCH = 1   # Always 1 for this pipeline

@pytest.fixture
def mock_hf_kv():
    """Generates a fake Hugging Face DynamicCache object."""
    cache = DynamicCache()
    for i in range(L):
        # Shape: (batch_size, num_heads, seq_len, head_dim)
        k = torch.randn(BATCH, H, N, D, dtype=torch.float16, device='cuda')
        v = torch.randn(BATCH, H, N, D, dtype=torch.float16, device='cuda')
        cache.update(k, v, i)
    return cache

def test_pack_unpack_shapes(mock_hf_kv):
    """Verifies that packing produces the exact required (L, 2, N, H, D) shape."""
    packed = pack_kv(mock_hf_kv)
    
    # Check shape contract
    assert packed.shape == (L, 2, N, H, D), f"Expected {(L, 2, N, H, D)}, got {packed.shape}"
    assert packed.dtype == torch.float16
    assert packed.device.type == 'cuda'
    
    # Check unpack shape
    unpacked = unpack_kv(packed)
    assert len(unpacked.layers) == L
    assert unpacked.layers[0].keys.shape == (BATCH, H, N, D)

def test_pack_unpack_lossless(mock_hf_kv):
    """Verifies that packing and unpacking preserves exact values."""
    packed = pack_kv(mock_hf_kv)
    unpacked = unpack_kv(packed)
    
    for i in range(L):
        # Check K
        assert torch.allclose(mock_hf_kv.layers[i].keys, unpacked.layers[i].keys), f"Layer {i} Key mismatch"
        # Check V
        assert torch.allclose(mock_hf_kv.layers[i].values, unpacked.layers[i].values), f"Layer {i} Value mismatch"

def test_kv_store_memory_backend(mock_hf_kv):
    """Verifies the store saves to CPU and loads back to CUDA."""
    packed = pack_kv(mock_hf_kv)
    store = KVCacheStore()
    
    text_chunk = "The quick brown fox jumps over the lazy dog."
    
    # Store it
    store.store(text_chunk, packed)
    
    # Verify it's stored on CPU internally to save VRAM
    cache_key = store._key(text_chunk)
    assert store._mem[cache_key].device.type == 'cpu'
    
    # Load it back
    loaded = store.load(text_chunk, device='cuda')
    assert loaded is not None
    assert loaded.device.type == 'cuda'
    assert torch.allclose(packed, loaded)
    
    # Test cache miss
    miss = store.load("Non-existent chunk", device='cuda')
    assert miss is None