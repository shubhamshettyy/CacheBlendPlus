"""Insert a ROUGE-L repair cell into the Colab notebook."""
import json, uuid, pathlib

NB = pathlib.Path(
    "C:/D Drive stuff/Edu/MSCS/sem2/690AB/latest/CacheBlendPlus"
    "/colab_cacheblend_all_in_one.ipynb"
)

with open(NB) as f:
    nb = json.load(f)

def make_code(text):
    return {
        "cell_type": "code",
        "id": uuid.uuid4().hex[:8],
        "metadata": {},
        "outputs": [],
        "execution_count": None,
        "source": [text],
    }

def make_md(text):
    return {
        "cell_type": "markdown",
        "id": uuid.uuid4().hex[:8],
        "metadata": {},
        "source": [text],
    }

# ── Idempotency: remove any prior repair cells ────────────────────────────
SENTINEL = "ROUGE-L Repair"
nb["cells"] = [
    c for c in nb["cells"]
    if SENTINEL not in "".join(c.get("source", []))
]

REPAIR_MD = make_md(
    "### ROUGE-L Repair\n\n"
    "The eval ran successfully but `evaluate` wasn't importable during that session,\n"
    "so all `rougeL` fields are `null`. This cell re-generates predictions for every\n"
    "null entry (skipping TTFT re-measurement) and fills in the scores."
)

REPAIR_CODE = make_code(r"""# ROUGE-L Repair: fills null rougeL without re-measuring TTFT
import json, importlib
from pathlib import Path

# Force-reload evaluate in case it was installed mid-session
try:
    import evaluate as _ev
    importlib.reload(_ev)
    rouge = _ev.load("rouge")
    print(f"evaluate {_ev.__version__} ready")
except Exception as e:
    raise RuntimeError(
        f"evaluate not importable: {e}\n"
        "Run:  !pip install evaluate rouge_score -q\n"
        "then Restart Runtime and re-run from the top."
    )


def _generate_preds(model, tokenizer, samples, store_class, sel_fn,
                    recomputer, blender, max_new_tokens=128):
    # Re-run warm cacheblend_generate for each sample; return list of prediction strings.
    preds = []
    for i, s in enumerate(samples):
        chunks = s["chunks"]
        prompt = "Summarize the above in 2-3 sentences:"
        store    = store_class()
        selector = sel_fn()
        # Cold pass to populate cache
        cacheblend_generate(prompt, chunks, model, tokenizer,
                            store, selector, recomputer, blender,
                            max_new_tokens=1, do_sample=False)
        # Warm pass for quality
        r = cacheblend_generate(prompt, chunks, model, tokenizer,
                                store, selector, recomputer, blender,
                                max_new_tokens=max_new_tokens, do_sample=False)
        preds.append(r["text"])
        if (i + 1) % 10 == 0:
            print(f"  {i+1}/{len(samples)}")
    return preds


def repair_rouge(results_dir, model, tokenizer, samples,
                 recomputer, blender, max_new_tokens=128):
    refs = [s["summary"] for s in samples]

    # ── table1_table2.json ────────────────────────────────────────────────
    t12_path = Path(results_dir) / "table1_table2.json"
    with open(t12_path) as f:
        t12 = json.load(f)

    t12_dirty = False

    # Baseline — uses standard_cache (no selector needed)
    if t12.get("baseline", {}).get("rougeL") is None and "baseline" in t12:
        print("\n[Repair] baseline ROUGE-L ...")
        preds = _generate_preds(
            model, tokenizer, samples,
            KVCacheStore,
            lambda: TokenSelector(k_ratio=1.0),   # k=100% => standard cache behaviour
            recomputer, blender, max_new_tokens,
        )
        t12["baseline"]["rougeL"] = rouge.compute(
            predictions=preds, references=refs
        )["rougeL"]
        t12_dirty = True
        print(f"  baseline rougeL = {t12['baseline']['rougeL']:.4f}")

    for key, entry in t12.items():
        if not key.startswith("ratio_") or entry.get("rougeL") is not None:
            continue
        ratio = entry["ratio"]
        print(f"\n[Repair] ratio={ratio:.0%} ROUGE-L ...")
        preds = _generate_preds(
            model, tokenizer, samples,
            KVCacheStore,
            lambda r=ratio: TokenSelector(k_ratio=r),
            recomputer, blender, max_new_tokens,
        )
        entry["rougeL"] = rouge.compute(
            predictions=preds, references=refs
        )["rougeL"]
        t12_dirty = True
        print(f"  ratio={ratio:.0%} rougeL = {entry['rougeL']:.4f}")

    if t12_dirty:
        with open(t12_path, "w") as f:
            json.dump(t12, f, indent=2)
        print(f"\nSaved updated {t12_path}")

    # ── table3.json ────────────────────────────────────────────────────────
    t3_path = Path(results_dir) / "table3.json"
    if not t3_path.exists():
        print("table3.json not found — skipping")
        return t12, {}

    with open(t3_path) as f:
        t3 = json.load(f)

    t3_dirty = False
    sel_map = {
        "fixed":    lambda: TokenSelector(k_ratio=0.15),
        "adaptive": lambda: AdaptiveTokenSelector(
            model=model,
            low_thresh=0.05, high_thresh=0.20,
            min_k_ratio=0.075, max_k_ratio=0.30,
        ),
    }

    for key, entry in t3.items():
        if entry.get("rougeL") is not None:
            continue
        sel_type   = entry["selector"]
        store_type = entry["store"]
        print(f"\n[Repair] Table 3  {sel_type} x {store_type} ...")

        if store_type == "semantic":
            try:
                from sentence_transformers import SentenceTransformer
                import numpy as _np

                class _SemanticStore(KVCacheStore):
                    def __init__(self):
                        super().__init__()
                        self.encoder   = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")
                        self.threshold = 0.85
                        self._embeddings = []
                    def store(self, text, kv):
                        self._data[text] = kv.cpu().clone()
                        self._embeddings.append(
                            (self.encoder.encode(text, convert_to_numpy=True), text)
                        )
                    def load(self, text, device="cuda"):
                        kv = self._data.get(text)
                        if kv is not None:
                            return kv.to(device)
                        if not self._embeddings:
                            return None
                        q = self.encoder.encode(text, convert_to_numpy=True)
                        best_sim, best_key = -1.0, None
                        for emb, k in self._embeddings:
                            sim = float(_np.dot(q, emb) /
                                        (_np.linalg.norm(q)*_np.linalg.norm(emb)+1e-8))
                            if sim > best_sim:
                                best_sim, best_key = sim, k
                        if best_sim >= self.threshold and best_key in self._data:
                            return self._data[best_key].to(device)
                        return None

                store_cls = _SemanticStore
            except ImportError:
                print("  sentence-transformers not installed — skipping semantic entry")
                continue
        else:
            store_cls = KVCacheStore

        preds = _generate_preds(
            model, tokenizer, samples,
            store_cls,
            sel_map[sel_type],
            recomputer, blender, max_new_tokens,
        )
        entry["rougeL"] = rouge.compute(predictions=preds, references=refs)["rougeL"]
        t3_dirty = True
        print(f"  {key} rougeL = {entry['rougeL']:.4f}")

    if t3_dirty:
        with open(t3_path, "w") as f:
            json.dump(t3, f, indent=2)
        print(f"\nSaved updated {t3_path}")

    return t12, t3


# ── Run the repair ─────────────────────────────────────────────────────────
t12_results, t3_results = repair_rouge(
    RESULTS_DIR,
    mistral_model, mistral_tok, mn_samples,
    mistral_recomputer, mistral_blender,
    max_new_tokens=128,
)

print_tables(t12_results, t3_results)
""")

# Find the last cell (print_tables cell) and insert repair cells before it
last_idx = len(nb["cells"]) - 1
nb["cells"].insert(last_idx, REPAIR_CODE)
nb["cells"].insert(last_idx, REPAIR_MD)

with open(NB, "w") as f:
    json.dump(nb, f, indent=1)

print(f"Done. Notebook now has {len(nb['cells'])} cells.")
print(f"Repair cells inserted at positions {last_idx} and {last_idx+1}.")
