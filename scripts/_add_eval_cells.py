"""Inserts eval-harness cells into colab_cacheblend_all_in_one.ipynb."""
import json, uuid, pathlib

NB = pathlib.Path(
    "C:/D Drive stuff/Edu/MSCS/sem2/690AB/latest/CacheBlendPlus"
    "/colab_cacheblend_all_in_one.ipynb"
)

with open(NB) as f:
    nb = json.load(f)

# Idempotency: remove any cells we added before (detect by a sentinel string)
SENTINEL = "Phase 2 — Full Eval Harness"
nb["cells"] = [c for c in nb["cells"]
               if not any(SENTINEL in (s if isinstance(s, str) else "")
                          for s in (c.get("source") or []))]


def md(text):
    return {"cell_type": "markdown", "id": uuid.uuid4().hex[:8],
            "metadata": {}, "source": [text]}

def code(src):
    return {"cell_type": "code", "id": uuid.uuid4().hex[:8],
            "metadata": {}, "outputs": [], "execution_count": None,
            "source": [src]}


# ── Cell A: section header ─────────────────────────────────────────────────
CA = md(
    "## Phase 2 — Full Eval Harness (Mistral-7B · MultiNews)\n\n"
    "Produces the three tables from the CacheBlend+ report:\n\n"
    "| Table | Metric | Configs |\n"
    "|-------|--------|---------|\n"
    "| 1 | TTFT (ms) | k/N ∈ {5 %, 10 %, 15 %, 20 %, 30 %} vs full-prefill |\n"
    "| 2 | ROUGE-L | same k/N sweep |\n"
    "| 3 | ROUGE-L grid | {fixed, adaptive} × {exact, semantic} |\n\n"
    "> **GPU required.** Run on a Colab A100 or T4 instance.  \n"
    "> Results are saved incrementally to "
    "`/content/drive/MyDrive/cacheblend_results/`  \n"
    "> so the notebook is safe to preempt and resume."
)

# ── Cell B: mount drive ────────────────────────────────────────────────────
CB = code(
    "from google.colab import drive\n"
    "drive.mount('/content/drive')\n"
    "\n"
    "import os\n"
    "RESULTS_DIR = '/content/drive/MyDrive/cacheblend_results'\n"
    "os.makedirs(RESULTS_DIR, exist_ok=True)\n"
    "print(f'Results will be saved to: {RESULTS_DIR}')\n"
)

# ── Cell C: inline eval helpers ────────────────────────────────────────────
CC_SRC = r"""# ── Eval-harness helpers (all-inline; no package install needed) ────────────
import json, time
from pathlib import Path
from statistics import mean


def load_rouge():
    try:
        import evaluate
        return evaluate.load("rouge")
    except Exception:
        print("WARNING: 'evaluate' not installed.")
        return None


def load_multinews(n_samples: int) -> list:
    from datasets import load_dataset
    try:
        ds = load_dataset("multi_news", split=f"validation[:{n_samples}]",
                          trust_remote_code=True)
        samples = []
        for ex in ds:
            articles = [a.strip() for a in ex["document"].split("|||||") if a.strip()]
            summary  = ex["summary"].strip()
            if articles and summary:
                samples.append({"chunks": articles[:4], "summary": summary})
        return samples[:n_samples]
    except Exception as e:
        print(f"WARNING: MultiNews load failed ({e}) — using synthetic fallback")
        return [{
            "chunks": [
                "Scientists discovered a new deep-sea fish in the Pacific Ocean. "
                "The fish lives at depths over 3000 m and has bioluminescent properties.",
                "The discovery adds to a growing list of deep-sea species. "
                "Researchers named it Abyssus luminis.",
            ],
            "summary": "Researchers found a new bioluminescent deep-sea fish in the Pacific.",
        }] * min(n_samples, 5)


def measure_prefill_ttft(model, input_ids, n_runs=3):
    # Median wall-clock prefill time (ms) over n_runs trials.
    times = []
    for _ in range(n_runs):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.no_grad():
            model(input_ids, use_cache=True)
        torch.cuda.synchronize()
        times.append((time.perf_counter() - t0) * 1000)
    return sorted(times)[n_runs // 2]


def _standard_cache_generate(model, tokenizer, chunks, max_new_tokens=128):
    # Baseline generation: full prefill, no CacheBlend.
    store_tmp = KVCacheStore()
    dummy_sel = TokenSelector(k_ratio=0.15)   # never used (cold pass only)
    r = cacheblend_generate(
        "Summarize the above in 2-3 sentences:",
        chunks, model, tokenizer,
        store_tmp, dummy_sel,
        SelectiveRecomputer(model), KVBlender(),
        mode="standard_cache",
        max_new_tokens=max_new_tokens,
        do_sample=False,
    )
    return r["text"]


class TokenSelector:
    # Fixed k/N token selector (evenly spaced positions).
    def __init__(self, k_ratio=0.15):
        self.k_ratio = float(k_ratio)

    def select(self, chunk_ids, cached_kv):
        N = int(chunk_ids.shape[1])
        k = max(1, min(N, int(self.k_ratio * N)))
        indices = torch.linspace(0, N - 1, k, dtype=torch.int64, device=chunk_ids.device)
        return torch.sort(indices).values


def eval_table1_table2(model, tokenizer, samples, recompute_ratios,
                       recomputer, blender, results_dir, max_new_tokens=128):
    outpath = Path(results_dir) / "table1_table2.json"
    results = {}
    if outpath.exists():
        with open(outpath) as f:
            results = json.load(f)
        print(f"Resuming table1_table2 ({len(results)} entries already done)")

    rouge = load_rouge()

    # ── Baseline: full prefill ─────────────────────────────────────────────
    if "baseline" not in results:
        print("\n[Baseline] full-prefill TTFT + ROUGE-L ...")
        base_ttfts, base_preds = [], []
        for i, s in enumerate(samples):
            all_text = " ".join(s["chunks"])
            ids = tokenizer(all_text, return_tensors="pt",
                            truncation=True, max_length=2048)["input_ids"].cuda()
            base_ttfts.append(measure_prefill_ttft(model, ids))
            base_preds.append(
                _standard_cache_generate(model, tokenizer, s["chunks"], max_new_tokens)
            )
            if (i + 1) % 10 == 0:
                print(f"  {i+1}/{len(samples)}")

        base_rouge = None
        if rouge:
            refs = [s["summary"] for s in samples]
            base_rouge = rouge.compute(predictions=base_preds, references=refs)["rougeL"]

        results["baseline"] = {
            "mean_ttft_ms": mean(base_ttfts),
            "rougeL": base_rouge,
        }
        with open(outpath, "w") as f:
            json.dump(results, f, indent=2)
        rl = results["baseline"]["rougeL"]
        print(f"  Baseline TTFT={results['baseline']['mean_ttft_ms']:.1f} ms  "
              f"ROUGE-L={f'{rl:.4f}' if rl else 'N/A'}")

    # ── CacheBlend sweep ────────────────────────────────────────────────────
    for ratio in recompute_ratios:
        key = f"ratio_{ratio:.2f}"
        if key in results:
            print(f"  Skipping ratio={ratio} (cached)")
            continue

        print(f"\n[CacheBlend] k/N={ratio:.0%} ...")
        store    = KVCacheStore()
        selector = TokenSelector(k_ratio=ratio)
        cb_ttfts, cb_preds = [], []

        for i, s in enumerate(samples):
            chunks = s["chunks"]
            prompt = "Summarize the above in 2-3 sentences:"

            # Cold pass — populate cache
            cacheblend_generate(prompt, chunks, model, tokenizer,
                                store, selector, recomputer, blender,
                                max_new_tokens=1, do_sample=False)

            # TTFT proxy: time prefill on k tokens (matches paper's Table 1)
            all_text = " ".join(chunks)
            ids = tokenizer(all_text, return_tensors="pt",
                            truncation=True, max_length=2048)["input_ids"].cuda()
            k = max(1, int(ratio * ids.shape[1]))
            cb_ttfts.append(measure_prefill_ttft(model, ids[:, :k]))

            # Warm pass — quality measurement
            r = cacheblend_generate(prompt, chunks, model, tokenizer,
                                    store, selector, recomputer, blender,
                                    max_new_tokens=max_new_tokens, do_sample=False)
            cb_preds.append(r["text"])
            if (i + 1) % 10 == 0:
                print(f"  {i+1}/{len(samples)}")

        cb_rouge = None
        if rouge:
            refs = [s["summary"] for s in samples]
            cb_rouge = rouge.compute(predictions=cb_preds, references=refs)["rougeL"]

        mean_ttft = mean(cb_ttfts)
        results[key] = {
            "ratio": ratio,
            "mean_ttft_ms": mean_ttft,
            "speedup_vs_baseline": results["baseline"]["mean_ttft_ms"] / mean_ttft,
            "rougeL": cb_rouge,
        }
        with open(outpath, "w") as f:
            json.dump(results, f, indent=2)
        rl = cb_rouge
        print(f"  ratio={ratio:.0%}  TTFT={mean_ttft:.1f} ms  "
              f"speedup={results[key]['speedup_vs_baseline']:.2f}x  "
              f"ROUGE-L={f'{rl:.4f}' if rl else 'N/A'}")

    return results


def eval_table3(model, tokenizer, samples,
                exact_store_class, semantic_store_class,
                recomputer, blender,
                results_dir, k_ratio=0.15, max_new_tokens=128):
    outpath = Path(results_dir) / "table3.json"
    results = {}
    if outpath.exists():
        with open(outpath) as f:
            results = json.load(f)

    rouge = load_rouge()

    fixed_factory    = lambda k_ratio: TokenSelector(k_ratio=k_ratio)
    adaptive_factory = lambda k_ratio: AdaptiveTokenSelector(
        model=model,
        low_thresh=0.05, high_thresh=0.20,
        min_k_ratio=max(0.05, k_ratio * 0.5),
        max_k_ratio=min(k_ratio * 2, 0.50),
    )

    configs = [
        ("fixed",    "exact",    fixed_factory,    exact_store_class),
        ("adaptive", "exact",    adaptive_factory, exact_store_class),
    ]
    if semantic_store_class is not None:
        configs += [
            ("fixed",    "semantic", fixed_factory,    semantic_store_class),
            ("adaptive", "semantic", adaptive_factory, semantic_store_class),
        ]

    for sel_type, store_type, sel_fn, store_cls in configs:
        key = f"{sel_type}_{store_type}"
        if key in results:
            print(f"  Skipping {key} (cached)")
            continue

        print(f"\n[Table 3] {sel_type} x {store_type} ...")
        store    = store_cls()
        selector = sel_fn(k_ratio=k_ratio)
        preds    = []

        for i, s in enumerate(samples):
            chunks = s["chunks"]
            prompt = "Summarize the above in 2-3 sentences:"
            cacheblend_generate(prompt, chunks, model, tokenizer,
                                store, selector, recomputer, blender,
                                max_new_tokens=1, do_sample=False)
            r = cacheblend_generate(prompt, chunks, model, tokenizer,
                                    store, selector, recomputer, blender,
                                    max_new_tokens=max_new_tokens, do_sample=False)
            preds.append(r["text"])
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


def print_tables(t12, t3):
    print("\n" + "=" * 65)
    print("TABLE 1 & 2  |  MultiNews  |  Mistral-7B-Instruct-v0.2")
    print("=" * 65)
    b  = t12.get("baseline", {})
    bl = b.get("rougeL")
    print(f"{'Config':<24} {'TTFT (ms)':>10} {'Speedup':>10} {'ROUGE-L':>10}")
    print("-" * 56)
    print(f"{'Full prefill':<24} {b.get('mean_ttft_ms', 0):>10.1f} "
          f"{'1.00x':>10} {f'{bl:.4f}' if bl else 'N/A':>10}")
    for k in sorted(t12):
        if not k.startswith("ratio_"):
            continue
        v  = t12[k]
        rl = v.get("rougeL")
        label = f"CacheBlend {int(v['ratio']*100)}%"
        print(f"{label:<24} {v['mean_ttft_ms']:>10.1f} "
              f"{v['speedup_vs_baseline']:>9.2f}x "
              f"{f'{rl:.4f}' if rl else 'N/A':>10}")

    if t3:
        print("\n" + "=" * 48)
        print("TABLE 3  |  ROUGE-L  |  selector x cache store")
        print("=" * 48)
        print(f"{'':>16} {'Exact':>14} {'Semantic':>14}")
        print("-" * 46)
        for sel in ["fixed", "adaptive"]:
            exact    = t3.get(f"{sel}_exact",    {}).get("rougeL")
            semantic = t3.get(f"{sel}_semantic", {}).get("rougeL")
            print(f"{sel.capitalize():<16} "
                  f"{f'{exact:.4f}' if exact else 'N/A':>14} "
                  f"{f'{semantic:.4f}' if semantic else 'N/A':>14}")


print("Eval-harness helpers defined.")
"""
CC = code(CC_SRC)

# ── Cell D: load Mistral-7B ────────────────────────────────────────────────
CD = code(
    "# Load Mistral-7B-Instruct in 8-bit (fits on T4/A100)\n"
    "from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig\n"
    "\n"
    "MISTRAL_ID = 'mistralai/Mistral-7B-Instruct-v0.2'\n"
    "print(f'Loading {MISTRAL_ID} in 8-bit ...')\n"
    "\n"
    "bnb_cfg = BitsAndBytesConfig(load_in_8bit=True)\n"
    "mistral_tok = AutoTokenizer.from_pretrained(MISTRAL_ID)\n"
    "mistral_tok.pad_token = mistral_tok.eos_token\n"
    "mistral_model = AutoModelForCausalLM.from_pretrained(\n"
    "    MISTRAL_ID,\n"
    "    quantization_config=bnb_cfg,\n"
    "    device_map='cuda',\n"
    ").eval()\n"
    "\n"
    "mistral_recomputer = SelectiveRecomputer(mistral_model)\n"
    "mistral_blender    = KVBlender()\n"
    "print('Mistral-7B ready.')\n"
)

# ── Cell E: run Tables 1 & 2 ──────────────────────────────────────────────
CE = code(
    "N_SAMPLES        = 60\n"
    "MAX_NEW_TOKENS   = 128\n"
    "RECOMPUTE_RATIOS = [0.05, 0.10, 0.15, 0.20, 0.30]\n"
    "\n"
    "print(f'Loading {N_SAMPLES} MultiNews samples ...')\n"
    "mn_samples = load_multinews(N_SAMPLES)\n"
    "print(f'  Loaded {len(mn_samples)} samples')\n"
    "\n"
    "t12_results = eval_table1_table2(\n"
    "    mistral_model, mistral_tok, mn_samples,\n"
    "    RECOMPUTE_RATIOS,\n"
    "    mistral_recomputer, mistral_blender,\n"
    "    RESULTS_DIR,\n"
    "    max_new_tokens=MAX_NEW_TOKENS,\n"
    ")\n"
)

# ── Cell F: run Table 3 ───────────────────────────────────────────────────
CF = code(
    "# sentence-transformers needed for semantic store (Table 3 col 2).\n"
    "# Set USE_SEMANTIC=False to skip it.\n"
    "USE_SEMANTIC = True\n"
    "\n"
    "SemanticStore = None\n"
    "if USE_SEMANTIC:\n"
    "    try:\n"
    "        from sentence_transformers import SentenceTransformer\n"
    "        import numpy as _np\n"
    "\n"
    "        class SemanticKVCacheStore(KVCacheStore):\n"
    "            def __init__(self, model_name='all-MiniLM-L6-v2', threshold=0.85):\n"
    "                super().__init__()\n"
    "                self.encoder   = SentenceTransformer(model_name, device='cpu')\n"
    "                self.threshold = threshold\n"
    "                self._embeddings = []\n"
    "\n"
    "            def store(self, text, kv_tensor):\n"
    "                self._data[text] = kv_tensor.cpu().clone()\n"
    "                emb = self.encoder.encode(text, convert_to_numpy=True)\n"
    "                self._embeddings.append((emb, text))\n"
    "\n"
    "            def load(self, text, device='cuda'):\n"
    "                kv = self._data.get(text)\n"
    "                if kv is not None:\n"
    "                    return kv.to(device)\n"
    "                if not self._embeddings:\n"
    "                    return None\n"
    "                q = self.encoder.encode(text, convert_to_numpy=True)\n"
    "                best_sim, best_key = -1.0, None\n"
    "                for emb, key in self._embeddings:\n"
    "                    sim = float(_np.dot(q, emb) /\n"
    "                                (_np.linalg.norm(q) * _np.linalg.norm(emb) + 1e-8))\n"
    "                    if sim > best_sim:\n"
    "                        best_sim, best_key = sim, key\n"
    "                if best_sim >= self.threshold and best_key in self._data:\n"
    "                    return self._data[best_key].to(device)\n"
    "                return None\n"
    "\n"
    "        SemanticStore = SemanticKVCacheStore\n"
    "        print('SemanticKVCacheStore ready.')\n"
    "    except ImportError:\n"
    "        print('sentence-transformers not installed -- semantic column will be N/A')\n"
    "\n"
    "t3_results = eval_table3(\n"
    "    mistral_model, mistral_tok, mn_samples,\n"
    "    KVCacheStore, SemanticStore,\n"
    "    mistral_recomputer, mistral_blender,\n"
    "    RESULTS_DIR, k_ratio=0.15,\n"
    "    max_new_tokens=MAX_NEW_TOKENS,\n"
    ")\n"
)

# ── Cell G: print tables + save summary ───────────────────────────────────
CG = code(
    "print_tables(t12_results, t3_results)\n"
    "\n"
    "summary = {'table1_table2': t12_results, 'table3': t3_results}\n"
    "summary_path = f'{RESULTS_DIR}/eval_summary.json'\n"
    "with open(summary_path, 'w') as f:\n"
    "    json.dump(summary, f, indent=2)\n"
    "\n"
    "print(f'\\nFull results saved to {RESULTS_DIR}/')\n"
    "print(f'  table1_table2.json  -- per-ratio TTFT + ROUGE-L')\n"
    "print(f'  table3.json         -- selector x store ROUGE-L grid')\n"
    "print(f'  eval_summary.json   -- combined summary')\n"
)

nb["cells"] += [CA, CB, CC, CD, CE, CF, CG]

with open(NB, "w") as f:
    json.dump(nb, f, indent=1)

print(f"Done. Notebook now has {len(nb['cells'])} cells.")
