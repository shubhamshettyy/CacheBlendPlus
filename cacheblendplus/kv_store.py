import hashlib
import torch
import os
from transformers import DynamicCache

# ---------------------------------------------------------------------------
# MODULE-LEVEL HELPERS
# ---------------------------------------------------------------------------

def pack_kv(hf_cache: DynamicCache) -> torch.Tensor:
    """
    Transformers DynamicCache (new layer-based API) -> (L, 2, N, H, D) float16.
    Each layer has .keys and .values of shape (1, H, N, D).
    """
    layers = []
    for layer in hf_cache.layers:
        k = layer.keys.squeeze(0)   # (H, N, D)
        v = layer.values.squeeze(0)
        
        k = k.permute(1, 0, 2)      # (N, H, D)
        v = v.permute(1, 0, 2)
        
        layers.append(torch.stack([k, v], dim=0))  # (2, N, H, D)
        
    packed = torch.stack(layers, dim=0)              # (L, 2, N, H, D)
    return packed

def unpack_kv(packed: torch.Tensor) -> DynamicCache:
    """
    Converts a single (L, 2, N, H, D) tensor back into the DynamicCache format.
    Restores the batch_size = 1 dimension.
    """
    cache = DynamicCache()
    L = packed.shape[0]
    dtype = packed.dtype
    
    for i in range(L):
        # Extract K and V, permute back to (H, N, D), and add batch dim (1, H, N, D)
        k = packed[i, 0].permute(1, 0, 2).unsqueeze(0).to(dtype)
        v = packed[i, 1].permute(1, 0, 2).unsqueeze(0).to(dtype)
        cache.update(k, v, i)
        
    # Force all stored tensors back to original dtype
    for layer in cache.layers:
        layer.keys   = layer.keys.to(dtype)
        layer.values = layer.values.to(dtype)
        
    return cache

# ---------------------------------------------------------------------------
# CACHE STORE
# ---------------------------------------------------------------------------

class KVCacheStore:
    def __init__(self, disk_path=None):
        self._mem = {}
        self.disk_path = disk_path
        if disk_path: 
            os.makedirs(disk_path, exist_ok=True)

    def _key(self, text):
        return hashlib.sha256(text.encode()).hexdigest()

    def store(self, text, kv_tensor):
        k = self._key(text)
        # Store on CPU to save VRAM
        self._mem[k] = kv_tensor.cpu()
        
        if self.disk_path:
            torch.save(kv_tensor.cpu(), f"{self.disk_path}/{k}.pt")

    def load(self, text, device='cuda'):
        k = self._key(text)
        
        # Check memory cache first
        if k in self._mem:
            return self._mem[k].to(device)
            
        # Fallback to disk cache
        if self.disk_path:
            path = f"{self.disk_path}/{k}.pt"
            if os.path.exists(path):
                return torch.load(path).to(device)
                
        return None  # Cache miss