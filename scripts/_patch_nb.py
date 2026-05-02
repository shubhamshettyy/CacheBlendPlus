"""Patch colab_cacheblend_all_in_one.ipynb with three targeted fixes."""
import json, uuid, pathlib

NB = pathlib.Path(
    "C:/D Drive stuff/Edu/MSCS/sem2/690AB/latest/CacheBlendPlus"
    "/colab_cacheblend_all_in_one.ipynb"
)

with open(NB) as f:
    nb = json.load(f)

def src(cell):
    return "".join(cell.get("source", []))

def set_src(cell, text):
    cell["source"] = [text]

def make_code(text):
    return {
        "cell_type": "code",
        "id": uuid.uuid4().hex[:8],
        "metadata": {},
        "outputs": [],
        "execution_count": None,
        "source": [text],
    }

# ── Fix 1: Cell 1 — add evaluate verification after pip install ────────────
cell1 = nb["cells"][1]
assert "pip install" in src(cell1), "Cell 1 is not the pip install cell"
new_cell1 = src(cell1).rstrip()
if "evaluate.__version__" not in new_cell1:
    new_cell1 += (
        "\n\n"
        "# Verify evaluate loaded (restart kernel and re-run if this errors)\n"
        "import importlib, evaluate\n"
        "importlib.reload(evaluate)\n"
        "print(f'evaluate {evaluate.__version__} ready')\n"
    )
set_src(cell1, new_cell1)
print("Fix 1 applied: evaluate verification added to cell 1")

# ── Fix 2: Cell 2 — change max_length=512 → 1024 in cacheblend_generate ───
cell2 = nb["cells"][2]
assert "cacheblend_generate" in src(cell2), "Cell 2 is not the definitions cell"
old_src = src(cell2)
# Only the chunk tokenizer call inside cacheblend_generate uses max_length=512
# (the prompt tokenizer uses 128 — leave that alone)
patched = old_src.replace(
    "        truncation=True,\n            max_length=512,\n",
    "        truncation=True,\n            max_length=1024,\n",
)
if patched == old_src:
    # Try alternate indentation style
    patched = old_src.replace("max_length=512,", "max_length=1024,", 1)
assert patched != old_src, "Could not find max_length=512 in cell 2"
set_src(cell2, patched)
print("Fix 2 applied: max_length 512->1024 in cacheblend_generate (cell 2)")

# ── Fix 3: Insert sequence-length diagnostic cell after cell 9 ─────────────
# Cell 9 (index 9) is the "run Table 1 & 2" cell that also loads mn_samples
DIAG_CELL = make_code(
    "# Sequence-length diagnostic — run after mn_samples is loaded\n"
    "# Shows actual token counts so we can judge whether context is long enough\n"
    "# for meaningful TTFT speedups (paper uses 2k-8k token contexts).\n"
    "_lens = [\n"
    "    len(mistral_tok(' '.join(s['chunks']), truncation=True,\n"
    "                    max_length=4096)['input_ids'])\n"
    "    for s in mn_samples\n"
    "]\n"
    "_sorted = sorted(_lens)\n"
    "print(f'Sequence lengths across {len(_lens)} samples:')\n"
    "print(f'  Mean : {sum(_lens)/len(_lens):.0f} tokens')\n"
    "print(f'  Min  : {_sorted[0]} tokens')\n"
    "print(f'  p25  : {_sorted[len(_sorted)//4]} tokens')\n"
    "print(f'  p50  : {_sorted[len(_sorted)//2]} tokens')\n"
    "print(f'  p75  : {_sorted[3*len(_sorted)//4]} tokens')\n"
    "print(f'  Max  : {_sorted[-1]} tokens')\n"
    "if sum(_lens)/len(_lens) < 600:\n"
    "    print('NOTE: Mean sequence is short (<600 tokens). '\n"
    "          'Speedups will be modest. Consider using more chunks per sample.')\n"
)

# Insert after index 9
nb["cells"].insert(10, DIAG_CELL)
print("Fix 3 applied: sequence-length diagnostic inserted as new cell 10")

with open(NB, "w") as f:
    json.dump(nb, f, indent=1)

print(f"\nDone. Notebook now has {len(nb['cells'])} cells.")
