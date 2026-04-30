import os
import sys

import pytest
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Add the project root to sys.path to import cacheblendplus
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from cacheblendplus.adaptive_selector import AdaptiveTokenSelector
from cacheblendplus.kv_store import KVCacheStore
from cacheblendplus.pipeline import KVBlender, cacheblend_generate
from cacheblendplus.recompute_engine import SelectiveRecomputer


MODEL_ID = "sshleifer/tiny-gpt2"


def _load_model_and_tokenizer(device: str = "cuda"):
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, dtype=torch.float16
    ).to(device).eval()
    return model, tokenizer


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_adaptive_selector_output_contract():
    model, tokenizer = _load_model_and_tokenizer(device="cuda")

    text = (
        "CacheBlend recomputes only highly divergent tokens to reduce prefill latency. "
        "This is a compact smoke test for Colab."
    )
    enc = tokenizer(text, return_tensors="pt", return_attention_mask=True)
    chunk_ids = enc["input_ids"].to("cuda")
    chunk_mask = enc["attention_mask"].to("cuda")

    with torch.no_grad():
        out = model(chunk_ids, attention_mask=chunk_mask, use_cache=True)

    from cacheblendplus.kv_store import pack_kv

    cached_kv = pack_kv(out.past_key_values).to("cuda")

    selector = AdaptiveTokenSelector(
        model=model,
        low_thresh=0.05,
        high_thresh=0.20,
        min_k_ratio=0.05,
        max_k_ratio=0.30,
    )

    indices = selector.select(chunk_ids, cached_kv)
    stats = selector.get_last_selection_stats()

    assert indices.dtype == torch.int64
    assert indices.device.type == "cuda"
    assert int(indices.numel()) >= 1
    assert torch.all(indices[1:] >= indices[:-1]), "Indices must be sorted ascending."
    assert 0.05 <= stats["selected_k_ratio"] <= 0.30
    assert stats["selected_k"] == float(indices.numel())


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_colab_pipeline_with_adaptive_selector():
    model, tokenizer = _load_model_and_tokenizer(device="cuda")

    store = KVCacheStore()
    selector = AdaptiveTokenSelector(model=model)
    recomputer = SelectiveRecomputer(model)
    blender = KVBlender()

    chunks = [
        "Paris is the capital of France and has many famous landmarks.",
        "The Eiffel Tower is one of the most visited monuments in the world.",
    ]
    prompt = "What is the Eiffel Tower and where is it located?"

    cold = cacheblend_generate(
        prompt,
        chunks,
        model,
        tokenizer,
        store,
        selector,
        recomputer,
        blender,
        max_new_tokens=12,
    )

    warm = cacheblend_generate(
        prompt,
        chunks,
        model,
        tokenizer,
        store,
        selector,
        recomputer,
        blender,
        max_new_tokens=12,
    )

    assert cold["cache_misses"] == len(chunks)
    assert warm["cache_hits"] == len(chunks)
    assert isinstance(warm["text"], str)
    assert warm["ttft_ms"] > 0.0
