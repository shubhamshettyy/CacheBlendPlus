import os
import sys

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from cacheblendplus.adaptive_selector import AdaptiveTokenSelector
from cacheblendplus.kv_store import KVCacheStore
from cacheblendplus.pipeline import KVBlender, cacheblend_generate
from cacheblendplus.recompute_engine import SelectiveRecomputer


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is required for this smoke run.")

    device = "cuda"
    model_id = "sshleifer/tiny-gpt2"

    print(f"Loading model: {model_id}")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_id, dtype=torch.float16
    ).to(device).eval()

    store = KVCacheStore()
    selector = AdaptiveTokenSelector(model=model)
    recomputer = SelectiveRecomputer(model)
    blender = KVBlender()

    chunks = [
        "CacheBlend blends cached and recomputed KV values for faster warm starts.",
        "Adaptive selection changes recomputation budget based on observed divergence.",
    ]
    prompt = "How does CacheBlend reduce latency?"

    print("\nRunning cold call...")
    cold = cacheblend_generate(
        prompt, chunks, model, tokenizer, store, selector, recomputer, blender, max_new_tokens=20
    )
    print(f"COLD TTFT: {cold['ttft_ms']:.2f} ms | hits={cold['cache_hits']} misses={cold['cache_misses']}")

    print("\nRunning warm call...")
    warm = cacheblend_generate(
        prompt, chunks, model, tokenizer, store, selector, recomputer, blender, max_new_tokens=20
    )
    print(f"WARM TTFT: {warm['ttft_ms']:.2f} ms | hits={warm['cache_hits']} misses={warm['cache_misses']}")
    print(f"Output: {warm['text'][:200]}")
    print(f"Output repr: {repr(warm['text'][:200])}")

    stats = selector.get_last_selection_stats()
    print(f"\nAdaptive selector stats: {stats}")

    if warm["ttft_ms"] > 0:
        print(f"Speedup cold/warm: {cold['ttft_ms'] / warm['ttft_ms']:.2f}x")


if __name__ == "__main__":
    main()
