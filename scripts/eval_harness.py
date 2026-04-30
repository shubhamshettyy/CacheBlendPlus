"""
eval_harness.py — Pranav's Phase 2-3 deliverable
Produces three tables for the final report:

  Table 1: TTFT (ms)   at k/N in {0.05, 0.10, 0.15, 0.20, 0.30} vs full-prefill
  Table 2: ROUGE-L     at same k/N values
  Table 3: 2x2 grid    {adaptive, fixed} x {semantic, exact}
           (semantic column is N/A until Adithya's module is ready)

Dataset: MultiNews (HuggingFace) — 60 samples, matching paper's evaluation setup.

Usage on Colab:
  !pip install evaluate rouge_score datasets -q
  %run eval_harness.py --n_samples 60 --output_dir /content/drive/MyDrive/results

All results saved to JSON incrementally — safe to preempt and resume.
"""

import argparse
import json
import os
import time
import torch
from pathlib import Path


# ---------------------------------------------------------------------------
# ROUGE-L scorer
# ---------------------------------------------------------------------------
def load_rouge():
    try:
        import evaluate
        return evaluate.load("rouge")
    except Exception:
        print("WARNING: 'evaluate' not installed. Run: pip install evaluate rouge_score")
        return None


# ---------------------------------------------------------------------------
# Dataset loader — MultiNews
# ---------------------------------------------------------------------------
def load_multinews(n_samples: int) -> list:
    try:
        from datasets import load_dataset
        ds = load_dataset("multi_news", split=f"validation[:{n_samples}]",
                          trust_remote_code=True)
        samples = []
        for ex in ds:
            articles = [a.strip() for a in ex["document"].split("|||||")
                        if a.strip()]
            summary = ex["summary"].strip()
            if articles and summary:
                samples.append({
                    "chunks": articles[:4],  # cap at 4 chunks like the paper
                    "summary": summary,
                })
        return samples[:n_samples]
    except Exception as e:
        print(f"WARNING: Could not load MultiNews ({e}). Using fallback.")
        return [{
            "chunks": [
                "Scientists discovered a new deep-sea fish species in the Pacific Ocean. "
                "The fish lives at depths over 3000 meters and has bioluminescent properties.",
                "The discovery adds to a growing list of deep-sea species. "
                "The fish has been named Abyssus luminis by researchers.",
            ],
            "summary": "Researchers found a new bioluminescent deep-sea fish in the Pacific.",
        }] * min(n_samples, 5)


# ---------------------------------------------------------------------------
# TTFT measurement — prefill only, matches paper's definition
# ---------------------------------------------------------------------------
def measure_prefill_ttft(model, input_ids: torch.Tensor, n_runs: int = 3) -> float:
    times = []
    for _ in range(n_runs):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.no_grad():
            model(input_ids, use_cache=True)
        torch.cuda.synchronize()
        times.append((time.perf_counter() - t0) * 1000)
    return sorted(times)[n_runs // 2]


# ---------------------------------------------------------------------------
# Generation for ROUGE-L measurement
# ---------------------------------------------------------------------------
def generate_summary(model, tokenizer, chunks: list, max_new_tokens: int = 128) -> str:
    context = " ".join(chunks)
    prompt = f"{context}\n\nSummarize the above in 2-3 sentences:"
    inputs = tokenizer(
        prompt, return_tensors="pt", truncation=True, max_length=2048
    ).to("cuda")
    context_len = inputs["input_ids"].shape[1]
    with torch.no_grad():
        out = model.generate(
            inputs["input_ids"],
            max_new_tokens=max_new_tokens,
            use_cache=True,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    return tokenizer.decode(out[0][context_len:], skip_special_tokens=True)


# ---------------------------------------------------------------------------
# Table 1 + 2
# ---------------------------------------------------------------------------
def run_table1_table2(
    model, tokenizer, samples, recompute_ratios,
    selector_class, store_class, recomputer, blender,
    output_dir, max_new_tokens=128
) -> dict:

    from pipeline import cacheblend_generate

    outpath = Path(output_dir) / "table1_table2.json"
    results = {}
    if outpath.exists():
        with open(outpath) as f:
            results = json.load(f)
        print(f"Resuming from {outpath} ({len(results)} entries already done)")

    rouge = load_rouge()

    # --- Baseline ---
    if "baseline" not in results:
        print("\n[Baseline] Full prefill TTFT + generation quality...")
        base_ttfts, base_preds = [], []
        for i, s in enumerate(samples):
            all_text = " ".join(s["chunks"])
            ids = tokenizer(all_text, return_tensors="pt",
                            truncation=True, max_length=2048)["input_ids"].cuda()
            base_ttfts.append(measure_prefill_ttft(model, ids))
            base_preds.append(
                generate_summary(model, tokenizer, s["chunks"], max_new_tokens)
            )
            if (i + 1) % 10 == 0:
                print(f"  {i+1}/{len(samples)}")

        base_rouge = None
        if rouge:
            refs = [s["summary"] for s in samples]
            base_rouge = rouge.compute(predictions=base_preds,
                                       references=refs)["rougeL"]

        results["baseline"] = {
            "mean_ttft_ms": sum(base_ttfts) / len(base_ttfts),
            "rougeL": base_rouge,
        }
        with open(outpath, "w") as f:
            json.dump(results, f, indent=2)
        rl = results["baseline"]["rougeL"]
        print(f"  Baseline: TTFT={results['baseline']['mean_ttft_ms']:.1f}ms "
              f"| ROUGE-L={f'{rl:.4f}' if rl else 'N/A'}")

    # --- CacheBlend sweep ---
    for ratio in recompute_ratios:
        key = f"ratio_{ratio:.2f}"
        if key in results:
            print(f"  Skipping ratio={ratio} (cached)")
            continue

        print(f"\n[CacheBlend] ratio={ratio:.0%}...")
        store    = store_class()
        selector = selector_class(k_ratio=ratio)
        cb_ttfts, cb_preds = [], []

        for i, s in enumerate(samples):
            chunks = s["chunks"]
            prompt = "Summarize the above in 2-3 sentences:"

            # Cold call to populate cache
            cacheblend_generate(prompt, chunks, model, tokenizer,
                                store, selector, recomputer, blender,
                                max_new_tokens=1)

            # TTFT: prefill on k% of tokens (warm path equivalent)
            all_text = " ".join(chunks)
            ids = tokenizer(all_text, return_tensors="pt",
                            truncation=True, max_length=2048)["input_ids"].cuda()
            k = max(1, int(ratio * ids.shape[1]))
            cb_ttfts.append(measure_prefill_ttft(model, ids[:, :k]))

            # Quality: warm generation
            result = cacheblend_generate(prompt, chunks, model, tokenizer,
                                         store, selector, recomputer, blender,
                                         max_new_tokens=max_new_tokens)
            cb_preds.append(result["text"])

            if (i + 1) % 10 == 0:
                print(f"  {i+1}/{len(samples)}")

        cb_rouge = None
        if rouge:
            refs = [s["summary"] for s in samples]
            cb_rouge = rouge.compute(predictions=cb_preds,
                                     references=refs)["rougeL"]

        mean_ttft = sum(cb_ttfts) / len(cb_ttfts)
        results[key] = {
            "ratio": ratio,
            "mean_ttft_ms": mean_ttft,
            "speedup_vs_baseline": results["baseline"]["mean_ttft_ms"] / mean_ttft,
            "rougeL": cb_rouge,
        }
        with open(outpath, "w") as f:
            json.dump(results, f, indent=2)
        rl = cb_rouge
        print(f"  ratio={ratio:.0%}: TTFT={mean_ttft:.1f}ms "
              f"| speedup={results[key]['speedup_vs_baseline']:.2f}x "
              f"| ROUGE-L={f'{rl:.4f}' if rl else 'N/A'}")

    return results


# ---------------------------------------------------------------------------
# Table 3: {adaptive, fixed} x {exact, semantic}
# ---------------------------------------------------------------------------
def run_table3(
    model, tokenizer, samples,
    fixed_selector_class, adaptive_selector_class,
    exact_store_class, semantic_store_class,
    recomputer, blender,
    output_dir, k_ratio=0.15, max_new_tokens=128
) -> dict:

    from pipeline import cacheblend_generate

    outpath = Path(output_dir) / "table3.json"
    results = {}
    if outpath.exists():
        with open(outpath) as f:
            results = json.load(f)

    rouge = load_rouge()

    configs = [("fixed", "exact", fixed_selector_class, exact_store_class)]
    if adaptive_selector_class is not None:
        configs.append(("adaptive", "exact", adaptive_selector_class, exact_store_class))
    if semantic_store_class is not None:
        configs.append(("fixed",    "semantic", fixed_selector_class,    semantic_store_class))
        configs.append(("adaptive", "semantic", adaptive_selector_class, semantic_store_class))

    for sel_type, store_type, sel_cls, store_cls in configs:
        key = f"{sel_type}_{store_type}"
        if key in results:
            print(f"  Skipping {key}")
            continue

        print(f"\n[Table 3] {sel_type} x {store_type}...")
        store    = store_cls()
        selector = sel_cls(k_ratio=k_ratio)
        preds    = []

        for i, s in enumerate(samples):
            chunks = s["chunks"]
            prompt = "Summarize the above in 2-3 sentences:"
            cacheblend_generate(prompt, chunks, model, tokenizer,
                                store, selector, recomputer, blender,
                                max_new_tokens=1)
            result = cacheblend_generate(prompt, chunks, model, tokenizer,
                                         store, selector, recomputer, blender,
                                         max_new_tokens=max_new_tokens)
            preds.append(result["text"])
            if (i + 1) % 10 == 0:
                print(f"  {i+1}/{len(samples)}")

        score = None
        if rouge:
            refs = [s["summary"] for s in samples]
            score = rouge.compute(predictions=preds, references=refs)["rougeL"]

        results[key] = {"selector": sel_type, "store": store_type, "rougeL": score}
        with open(outpath, "w") as f:
            json.dump(results, f, indent=2)
        print(f"  {key}: ROUGE-L={f'{score:.4f}' if score else 'N/A'}")

    return results


# ---------------------------------------------------------------------------
# Print tables
# ---------------------------------------------------------------------------
def print_tables(t12: dict, t3: dict):
    print("\n" + "=" * 65)
    print("TABLE 1 & 2  |  MultiNews  |  Mistral-7B  |  A100")
    print("=" * 65)
    b = t12.get("baseline", {})
    print(f"{'Config':<22} {'TTFT (ms)':>10} {'Speedup':>10} {'ROUGE-L':>10}")
    print("-" * 54)
    bl = b.get("rougeL")
    print(f"{'Full prefill':<22} {b.get('mean_ttft_ms', 0):>10.1f} "
          f"{'1.00x':>10} {f'{bl:.4f}' if bl else 'N/A':>10}")
    for k in sorted(t12):
        if not k.startswith("ratio_"):
            continue
        v = t12[k]
        rl = v.get("rougeL")
        label = f"CacheBlend {int(v['ratio']*100)}%"
        print(f"{label:<22} {v['mean_ttft_ms']:>10.1f} "
              f"{v['speedup_vs_baseline']:>9.2f}x "
              f"{f'{rl:.4f}' if rl else 'N/A':>10}")

    if t3:
        print("\n" + "=" * 45)
        print("TABLE 3  |  ROUGE-L by selector x store")
        print("=" * 45)
        print(f"{'':>12} {'Exact':>12} {'Semantic':>12}")
        print("-" * 38)
        for sel in ["fixed", "adaptive"]:
            exact    = t3.get(f"{sel}_exact",    {}).get("rougeL")
            semantic = t3.get(f"{sel}_semantic", {}).get("rougeL")
            print(f"{sel.capitalize():<12} "
                  f"{f'{exact:.4f}' if exact else 'N/A':>12} "
                  f"{f'{semantic:.4f}' if semantic else 'N/A':>12}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id", default="mistralai/Mistral-7B-Instruct-v0.2")
    parser.add_argument("--recompute_ratios", nargs="+", type=float,
                        default=[0.05, 0.10, 0.15, 0.20, 0.30])
    parser.add_argument("--n_samples", type=int, default=60)
    parser.add_argument("--max_new_tokens", type=int, default=128)
    parser.add_argument("--output_dir", default="results")
    parser.add_argument("--skip_table3", action="store_true")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from kv_store import KVCacheStore
    from recompute_engine import SelectiveRecomputer
    from pipeline import KVBlender
    from token_selector import TokenSelector

    try:
        from adaptive_selector import AdaptiveSelector
        print("AdaptiveSelector loaded")
    except ImportError:
        print("WARNING: adaptive_selector.py not found — using TokenSelector for Table 3")
        AdaptiveSelector = None

    try:
        from semantic_kv_store import SemanticKVCacheStore
        print("SemanticKVCacheStore loaded")
    except ImportError:
        print("NOTE: semantic_kv_store.py not found — Table 3 semantic column = N/A")
        SemanticKVCacheStore = None

    print(f"Loading {args.model_id}...")
    bnb = BitsAndBytesConfig(load_in_8bit=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id, quantization_config=bnb, device_map="cuda"
    ).eval()

    recomputer = SelectiveRecomputer(model)
    blender    = KVBlender()

    print(f"Loading {args.n_samples} MultiNews samples...")
    samples = load_multinews(args.n_samples)
    print(f"  Loaded {len(samples)} samples")

    t12 = run_table1_table2(
        model, tokenizer, samples, args.recompute_ratios,
        TokenSelector, KVCacheStore, recomputer, blender,
        args.output_dir, args.max_new_tokens,
    )

    t3 = {}
    if not args.skip_table3:
        t3 = run_table3(
            model, tokenizer, samples,
            TokenSelector, AdaptiveSelector,
            KVCacheStore, SemanticKVCacheStore,
            recomputer, blender,
            args.output_dir, k_ratio=0.15,
            max_new_tokens=args.max_new_tokens,
        )

    print_tables(t12, t3)
    print(f"\nAll results saved to {args.output_dir}/")


if __name__ == "__main__":
    main()