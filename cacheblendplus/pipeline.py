"""
Fixed for Transformers >= 4.45 which requires DynamicCache instead of tuple past_key_values.
"""

import time
import torch
from .kv_store import pack_kv, unpack_kv
from .blend_kernel import blend

class KVBlender:
    def blend(self, cached_kv, new_values, indices):
        # The blend_kernel.py implementation handles both CUDA and CPU
        # Note: it might modify cached_kv in-place for efficiency
        return blend(cached_kv, new_values, indices)


def cacheblend_generate(
    prompt,
    chunk_texts,
    model,
    tokenizer,
    store,
    selector,
    recomputer,
    blender=None,
    mode: str = "cacheblend",  # "cacheblend" or "standard_cache"
    max_new_tokens=64,
    min_new_tokens=8,
    do_sample=True,
    temperature=0.8,
    top_p=0.95,
    benchmark_first_token: bool = False,
):
    if blender is None:
        blender = KVBlender()

    cache_hits = 0
    cache_misses = 0
    fused_kv = None

    # End-to-end request latency includes cache/prefill work plus first token decode.
    torch.cuda.synchronize()
    t_request_start = time.perf_counter()

    for chunk_text in chunk_texts:
        chunk_enc = tokenizer(
            chunk_text,
            return_tensors="pt",
            return_attention_mask=True,
            truncation=True,
            max_length=512,
        )
        chunk_ids = chunk_enc["input_ids"].cuda()
        chunk_attn = chunk_enc["attention_mask"].cuda()

        cached = store.load(chunk_text, device="cuda")

        if cached is None:
            cache_misses += 1
            with torch.no_grad():
                out = model(chunk_ids, attention_mask=chunk_attn, use_cache=True)
            chunk_kv = pack_kv(out.past_key_values)
            store.store(chunk_text, chunk_kv)
        else:
            cache_hits += 1
            chunk_kv = cached
            if mode == "cacheblend":
                indices = selector.select(chunk_ids, chunk_kv)
                new_kv = recomputer.recompute(chunk_ids, chunk_kv, indices)
                chunk_kv = blender.blend(chunk_kv, new_kv, indices)
            elif mode == "standard_cache":
                pass
            else:
                raise ValueError(f"Unsupported mode: {mode}")

        if fused_kv is None:
            fused_kv = chunk_kv
        else:
            fused_kv = torch.cat([fused_kv, chunk_kv], dim=2)

    prompt_enc = tokenizer(
        prompt,
        return_tensors="pt",
        return_attention_mask=True,
        truncation=True,
        max_length=128,
    )
    prompt_ids = prompt_enc["input_ids"].cuda()
    prompt_attn = prompt_enc["attention_mask"].cuda()

    # Convert to DynamicCache — required by Transformers >= 4.45
    past = unpack_kv(fused_kv) if fused_kv is not None else None

    torch.cuda.synchronize()
    t_decode_start = time.perf_counter()

    # cache_position tells generate() where the prompt tokens sit
    # after the already-cached context (which has fused_kv.shape[2] tokens)
    context_len = fused_kv.shape[2] if fused_kv is not None else 0
    cache_position = torch.arange(
        context_len,
        context_len + prompt_ids.shape[1],
        device=prompt_ids.device
    )

    gen_max = 1 if benchmark_first_token else max_new_tokens
    gen_min = 1 if benchmark_first_token else min_new_tokens
    gen_sample = False if benchmark_first_token else do_sample

    with torch.no_grad():
        out_ids = model.generate(
            prompt_ids,
            attention_mask=prompt_attn,
            past_key_values=past,
            cache_position=cache_position,
            max_new_tokens=gen_max,
            min_new_tokens=gen_min,
            use_cache=True,
            do_sample=gen_sample,
            temperature=temperature,
            top_p=top_p,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    torch.cuda.synchronize()
    generate_only_ms = (time.perf_counter() - t_decode_start) * 1000
    ttft_ms = (time.perf_counter() - t_request_start) * 1000

    generated = out_ids[0][prompt_ids.shape[1]:]
    text = tokenizer.decode(generated, skip_special_tokens=True).strip()

    return {
        "text": text,
        "ttft_ms": ttft_ms,
        "e2e_ttft_ms": ttft_ms,
        "generate_only_ms": generate_only_ms,
        "cache_hits": cache_hits,
        "cache_misses": cache_misses,
    }


if __name__ == "__main__":
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from kv_store import KVCacheStore
    from recompute_engine import SelectiveRecomputer

    class StubSelector:
        def __init__(self, k_ratio=0.15):
            self.k_ratio = k_ratio
        def select(self, chunk_ids, cached_kv):
            N = chunk_ids.shape[1]
            k = max(1, int(self.k_ratio * N))
            return torch.arange(k, dtype=torch.int64).cuda()

    MODEL_ID = "gpt2"
    print(f"Loading {MODEL_ID}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, dtype=torch.float16
    ).cuda().eval()

    store      = KVCacheStore()
    selector   = StubSelector(k_ratio=0.15)
    recomputer = SelectiveRecomputer(model)
    blender    = KVBlender()

    chunks = [
        "Paris is the capital city of France, located in northern France on the Seine river.",
        "The Eiffel Tower was built in 1889 as the entrance arch for the 1889 World's Fair.",
    ]
    prompt = "What is the Eiffel Tower and where is it?"

    print("\n--- COLD START ---")
    r1 = cacheblend_generate(prompt, chunks, model, tokenizer,
                              store, selector, recomputer, blender, max_new_tokens=40)
    print(f"Output : {r1['text'][:120]}")
    print(f"TTFT   : {r1['ttft_ms']:.1f} ms")
    print(f"Hits/Misses: {r1['cache_hits']}/{r1['cache_misses']}")

    print("\n--- WARM START ---")
    r2 = cacheblend_generate(prompt, chunks, model, tokenizer,
                              store, selector, recomputer, blender, max_new_tokens=40)
    print(f"Output : {r2['text'][:120]}")
    print(f"TTFT   : {r2['ttft_ms']:.1f} ms")
    print(f"Hits/Misses: {r2['cache_hits']}/{r2['cache_misses']}")

    if r1["ttft_ms"] > 0:
        speedup = r1["ttft_ms"] / r2["ttft_ms"]
        print(f"\nSpeedup: {speedup:.2f}x")
        if speedup >= 1.5:
            print("✓ Target met (>=1.5x)")
        else:
            print("⚠  Below 1.5x — normal for tiny GPT-2. Will be larger with Mistral-7B.")
