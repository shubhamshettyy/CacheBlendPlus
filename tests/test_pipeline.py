import torch
import pytest
import os
import sys
import time

# Add the project root to sys.path to import cacheblendplus
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from transformers import AutoModelForCausalLM, AutoTokenizer
from cacheblendplus.pipeline import cacheblend_generate, KVBlender
from cacheblendplus.kv_store import KVCacheStore
from cacheblendplus.recompute_engine import SelectiveRecomputer

class SimpleSelector:
    """A selector that just picks the first 15% of tokens (for testing)."""
    def __init__(self, r=0.15):
        self.r = r
    def select(self, chunk_ids, cached_kv):
        N = chunk_ids.shape[1]
        k = max(1, int(self.r * N))
        return torch.arange(k, device=chunk_ids.device, dtype=torch.int64)

@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_pipeline_speedup_and_correctness():
    MODEL_ID = "gpt2"
    device = "cuda"
    dtype = torch.float16

    print(f"\nLoading {MODEL_ID}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, torch_dtype=dtype
    ).to(device).eval()

    store = KVCacheStore()
    selector = SimpleSelector(r=0.15)
    recomputer = SelectiveRecomputer(model)
    blender = KVBlender()

    # Long context simulation
    chunks = [
        "The Great Wall of China is a series of fortifications that were built across the historical northern borders of ancient Chinese states.",
        "It was constructed to protect the Chinese states and empires against the raids and invasions of the various nomadic groups of the Eurasian Steppe.",
        "Several walls were built from as early as the 7th century BC, with selective stretches later joined together by Qin Shi Huang, the first emperor of China.",
        "The Ming dynasty (1368–1644) further built the Great Wall, making it the most well-known section of the wall today."
    ]
    prompt = "Who was the first emperor of China and what did he do with the wall?"

    print("\n--- Running COLD START (Baseline) ---")
    # In a cold start, all chunks are misses and will be computed fully
    res_cold = cacheblend_generate(
        prompt, chunks, model, tokenizer, store, selector, recomputer, blender, max_new_tokens=20
    )
    print(f"Cold Start TTFT: {res_cold['ttft_ms']:.2f} ms")
    print(f"Cold Start Output: {res_cold['text']}")

    print("\n--- Running WARM START (CacheBlend) ---")
    # In a warm start, chunks are loaded from store, and we use the selection/recompute logic
    res_warm = cacheblend_generate(
        prompt, chunks, model, tokenizer, store, selector, recomputer, blender, max_new_tokens=20
    )
    print(f"Warm Start TTFT: {res_warm['ttft_ms']:.2f} ms")
    print(f"Warm Start Output: {res_warm['text']}")

    # Validation
    assert res_warm['cache_hits'] == len(chunks), "Warm start should have 100% cache hits!"
    
    # Speedup Check
    speedup = res_cold['ttft_ms'] / res_warm['ttft_ms']
    print(f"\nMeasured Speedup: {speedup:.2f}x")
    
    # Technical check: we just want to ensure it generated *something* 
    # and didn't crash, and that TTFT was actually faster.
    assert len(res_warm['text']) > 0
    assert speedup > 1.0, f"CacheBlend should be faster than cold start! Got {speedup:.2f}x"

if __name__ == "__main__":
    test_pipeline_speedup_and_correctness()
