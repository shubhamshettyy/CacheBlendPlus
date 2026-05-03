import time
import torch
from .kv_store import pack_kv, unpack_kv
from .token_selector import CacheBlendFusor

from .blend_kernel import blend

class KVBlender:
    def blend(self, cached_kv, new_values, indices):
        return blend(cached_kv, new_values, indices)

def cacheblend_generate(
    prompt,
    chunk_texts,
    model,
    tokenizer,
    store,
    selector=None, # For backward compatibility in eval_harness
    recomputer=None,
    blender=None,
    mode: str = "cacheblend",  # "cacheblend" or "standard_cache"
    max_new_tokens=64,
    min_new_tokens=8,
    do_sample=True,
    temperature=0.8,
    top_p=0.95,
    benchmark_first_token: bool = False,
    k_ratio: float = 0.15,
):
    """
    Refactored to use CacheBlendFusor for efficient, paper-faithful recomputation.
    """
    cache_hits = 0
    cache_misses = 0
    
    # 1. Gather/Prefill all chunks
    fused_kv_parts = []
    hit_mask_parts = []
    chunk_offsets = []
    current_offset = 0

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t_request_start = time.perf_counter()

    for chunk_text in chunk_texts:
        chunk_enc = tokenizer(
            chunk_text,
            return_tensors="pt",
            truncation=True,
            max_length=512,
        )
        chunk_ids = chunk_enc["input_ids"].cuda()
        N_c = chunk_ids.shape[1]
        
        cached = store.load(chunk_text, device="cuda")
        
        if cached is None:
            cache_misses += 1
            with torch.no_grad():
                out = model(chunk_ids, use_cache=True)
            chunk_kv = pack_kv(out.past_key_values)
            store.store(chunk_text, chunk_kv)
            # A miss means we computed it "fresh" in context 0, but since we just 
            # computed it, it's technically a "hit" for the fusion logic 
            # (or we treat it as something that needs recompute if we want to be safe)
            # Paper: misses are recomputed fully.
            hit_mask_parts.append(torch.zeros(N_c, dtype=torch.bool, device="cuda"))
        else:
            cache_hits += 1
            chunk_kv = cached
            hit_mask_parts.append(torch.ones(N_c, dtype=torch.bool, device="cuda"))

        fused_kv_parts.append(chunk_kv)
        chunk_offsets.append(current_offset)
        current_offset += N_c

    fused_kv = torch.cat(fused_kv_parts, dim=2)
    hit_mask = torch.cat(hit_mask_parts, dim=0)
    
    # Full prompt construction
    all_chunks_text = " ".join(chunk_texts)
    full_prompt_text = all_chunks_text + " " + prompt
    full_ids = tokenizer(full_prompt_text, return_tensors="pt").input_ids.cuda()
    
    # 2. Fuse and Recompute using the optimized Fusor
    if mode == "cacheblend":
        # If selector is an AdaptiveSelector, we can use its k_ratio
        r = k_ratio
        if hasattr(selector, "k_ratio"):
            r = selector.k_ratio
        elif hasattr(selector, "base_k_ratio"):
            r = selector.base_k_ratio
            
        fusor = CacheBlendFusor(model, r=r)
        # Note: fused_kv is updated in-place by the fusor
        fused_kv, _ = fusor.fuse(
            full_ids[:, :current_offset], 
            fused_kv, 
            hit_mask, 
            chunk_offsets=chunk_offsets
        )
    
    # 3. Final Decode
    # Ensure we have at least the last token for generate if the prompt was fully cached
    if current_offset >= full_ids.shape[1]:
        # Prompt was part of chunks? Should not happen in RAG, but safety first.
        prompt_ids = full_ids[:, -1:] 
        current_offset -= 1
    else:
        prompt_ids = full_ids[:, current_offset:]

    past = unpack_kv(fused_kv)
    
    # Mistral requires a full attention mask covering the KV cache + current tokens
    kv_len = fused_kv.shape[2]
    total_len = kv_len + prompt_ids.shape[1]
    attention_mask = torch.ones((1, total_len), device=prompt_ids.device)

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t_decode_start = time.perf_counter()

    gen_kwargs = {
        "past_key_values": past,
        "attention_mask": attention_mask,
        "max_new_tokens": 1 if benchmark_first_token else max_new_tokens,
        "min_new_tokens": 1 if benchmark_first_token else min_new_tokens,
        "use_cache": True,
        "do_sample": False if benchmark_first_token else do_sample,
        "pad_token_id": tokenizer.eos_token_id,
        "eos_token_id": tokenizer.eos_token_id,
    }
    if gen_kwargs["do_sample"]:
        gen_kwargs["temperature"] = temperature
        gen_kwargs["top_p"] = top_p

    with torch.no_grad():
        out_ids = model.generate(prompt_ids, **gen_kwargs)

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    ttft_ms = (time.perf_counter() - t_request_start) * 1000
    
    generated = out_ids[0]
    text = tokenizer.decode(generated, skip_special_tokens=True).strip()

    return {
        "text": text,
        "ttft_ms": ttft_ms,
        "cache_hits": cache_hits,
        "cache_misses": cache_misses,
    }
