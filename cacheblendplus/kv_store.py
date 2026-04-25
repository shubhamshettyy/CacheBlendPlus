import torch
from typing import Optional

def pack_kv(past_key_values) -> torch.Tensor:
    """HuggingFace past_key_values → (L, 2, N, H, D) float16."""
    if hasattr(past_key_values, 'key_cache'):
        key_cache = past_key_values.key_cache
        val_cache = past_key_values.value_cache
        layers = []
        for k, v in zip(key_cache, val_cache):
            # k, v are (1, H, N, D)
            k = k.squeeze(0).transpose(0, 1) # (N, H, D)
            v = v.squeeze(0).transpose(0, 1) # (N, H, D)
            layers.append(torch.stack([k, v], dim=0)) # (2, N, H, D)
        return torch.stack(layers, dim=0) # (L, 2, N, H, D)

    layers = []
    for layer_kv in past_key_values:
        k, v = layer_kv[0], layer_kv[1]
        k = k.squeeze(0).transpose(0, 1)
        v = v.squeeze(0).transpose(0, 1)
        layers.append(torch.stack([k, v], dim=0))
    return torch.stack(layers, dim=0)

def unpack_kv(kv_tensor: torch.Tensor):
    """(L, 2, N, H_kv, D) → DynamicCache for newer transformers."""
    from transformers import DynamicCache
    cache = DynamicCache()
    for i in range(kv_tensor.shape[0]):
        k = kv_tensor[i, 0].permute(1, 0, 2).unsqueeze(0).half()  # (1, H_kv, N, D)
        v = kv_tensor[i, 1].permute(1, 0, 2).unsqueeze(0).half()
        cache.update(k, v, i)
    return cache

def get_model_layers(model) -> list:
    name = type(model).__name__.lower()
    if 'gpt2' in name:
        return list(model.transformer.h)
    if 'mistral' in name or 'llama' in name or 'mistral' in str(type(model.model)):
        return list(model.model.layers)
    raise NotImplementedError(f"Model {type(model).__name__} not supported.")

def get_embeddings(model, input_ids: torch.Tensor) -> torch.Tensor:
    name = type(model).__name__.lower()
    N = input_ids.shape[1]

    if 'gpt2' in name:
        pos_ids = torch.arange(N, device=input_ids.device).unsqueeze(0)
        tok_emb = model.transformer.wte(input_ids)
        pos_emb = model.transformer.wpe(pos_ids)
        return model.transformer.drop(tok_emb + pos_emb)

    if 'mistral' in name or 'llama' in name:
        return model.model.embed_tokens(input_ids)

    raise NotImplementedError

def apply_final_norm(model, hidden: torch.Tensor) -> torch.Tensor:
    name = type(model).__name__.lower()
    if 'gpt2' in name:
        return model.transformer.ln_f(hidden)
    if 'mistral' in name or 'llama' in name:
        return model.model.norm(hidden)
    raise NotImplementedError

class KVCacheStore:
    """
    A simple dictionary-based KV cache store.
    In a real-world scenario, this might use a vector database or disk-based storage.
    """
    def __init__(self):
        self._data = {}

    def store_kv(self, key: str, kv_tensor: torch.Tensor):
        # We store a clone to avoid issues if the original tensor is modified in-place
        self._data[key] = kv_tensor.cpu().clone()

    def load_kv(self, key: str, device: str = "cuda") -> Optional[torch.Tensor]:
        kv = self._data.get(key)
        if kv is not None:
            return kv.to(device)
        return None

    # Aliases to match pipeline.py usage if needed
    def store(self, key: str, kv_tensor: torch.Tensor):
        self.store_kv(key, kv_tensor)
    
    def load(self, key: str, device: str = "cuda") -> Optional[torch.Tensor]:
        return self.load_kv(key, device)
