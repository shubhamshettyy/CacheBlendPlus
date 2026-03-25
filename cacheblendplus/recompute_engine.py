"""
dependency:pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

Interface:
  recompute(chunk_ids, cached_kv, indices) → Tensor[L, 2, k, H, D]  float16, CUDA
"""

import torch
from kv_store import pack_kv, unpack_kv # Adi's helpers
from pipeline import _to_dynamic_cache

class SelectiveRecomputer:
    """
    Runs a minimal forward pass over k selected tokens, using the full
    cached KV as past_key_values context so attention is computed correctly
    over the entire chunk and not just the k selected tokens.
    """

    def __init__(self, model):
        self.model = model

    def recompute(
        self,
        chunk_ids: torch.Tensor,   # (1, N)  full chunk token IDs
        cached_kv: torch.Tensor,   # (L, 2, N, H, D)  pre-computed KV
        indices: torch.Tensor,     # (k,)  int64, sorted ascending — from TokenSelector (Shubham's module)
    ) -> torch.Tensor:
        """
        Returns the freshly-computed KV entries for only the k selected tokens.
        Shape: (L, 2, k, H, D)

        The caller (pipeline.py) is responsible for scattering
        these back into the full cached_kv at positions `indices`.
        """
        assert indices.dtype == torch.int64, "indices must be int64"
        assert cached_kv.dtype == torch.float16, "cached_kv must be float16"
        assert cached_kv.device.type == "cuda", "cached_kv must be on CUDA"

        # Extract just the selected token IDs
        selected_ids = chunk_ids[:, indices]  # (1, k)

        # Unpack to HuggingFace tuple format so we can pass as past_key_values as this makes the model attend over the FULL cached context, not just k tokens
        #past = unpack_kv(cached_kv)
        past = _to_dynamic_cache(cached_kv) #added to fix transformers 4.45+ compatibility, which switched to DynamicCache for past_key_values instead of tuples

        with torch.no_grad():
            out = self.model(
                selected_ids,
                past_key_values=past,
                use_cache=True,
            )

        # HuggingFace appends the new k tokens at the END of past_key_values
        # We only want those fresh k entries, not the full N+k tensor
        new_kv_full = pack_kv(out.past_key_values)  # (L, 2, N+k, H, D)
        k = indices.shape[0]
        new_kv = new_kv_full[:, :, -k:, :, :]       # (L, 2, k, H, D)

        return new_kv


# ---------------------------------------------------------------------------
# Quick smoke test:
# Requires transformers, torch, kv_store.py (Adi's) in the same dir
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import time

    MODEL_ID = "gpt2"
    print(f"Loading {MODEL_ID}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, torch_dtype=torch.float16
    ).cuda().eval()

    text = "The quick brown fox jumps over the lazy dog. " * 4
    inputs = tokenizer(text, return_tensors="pt").to("cuda")
    chunk_ids = inputs["input_ids"]  # (1, N)
    N = chunk_ids.shape[1]
    print(f"Chunk length: {N} tokens")

    # Build a fake cached KV by running a real forward pass
    with torch.no_grad():
        out = model(chunk_ids, use_cache=True)
    cached_kv = pack_kv(out.past_key_values)
    print(f"cached_kv shape: {cached_kv.shape}  (L, 2, N, H, D)")

    # Pick ~15% of tokens as "high divergence" (random stand-in for the selector)
    k = max(1, int(0.15 * N))
    indices = torch.randint(0, N, (k,)).unique().sort().values.to(torch.int64).cuda()
    print(f"Recomputing {len(indices)}/{N} tokens ({len(indices)/N:.0%})")

    recomputer = SelectiveRecomputer(model)

    t0 = time.perf_counter()
    new_kv = recomputer.recompute(chunk_ids, cached_kv, indices)
    elapsed = (time.perf_counter() - t0) * 1000

    print(f"new_kv shape: {new_kv.shape}  (L, 2, k, H, D)")
    print(f"Recompute time: {elapsed:.1f} ms")
    assert new_kv.shape[2] == len(indices), "k dimension mismatch!"
    assert new_kv.dtype == torch.float16
    print("✓ Smoke test passed")
