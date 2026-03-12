import torch
import torch.nn.functional as F
from typing import List, Tuple, Optional

# ─────────────────────────────────────────────────────────────────────────────
# Cache Management & Utility Views (Zero-Copy where possible)
# ─────────────────────────────────────────────────────────────────────────────

def pack_kv(past_key_values) -> torch.Tensor:
    """HuggingFace past_key_values → (L, 2, N, H, D) float16."""
    if hasattr(past_key_values, 'key_cache'):
        key_cache = past_key_values.key_cache
        val_cache = past_key_values.value_cache
        layers = []
        for k, v in zip(key_cache, val_cache):
            k = k.squeeze(0).transpose(0, 1) # Prefer transpose over permute for 2D swaps
            v = v.squeeze(0).transpose(0, 1)
            layers.append(torch.stack([k, v], dim=0))
        return torch.stack(layers, dim=0)

    layers = []
    for layer_kv in past_key_values:
        k, v = layer_kv[0], layer_kv[1]
        k = k.squeeze(0).transpose(0, 1)
        v = v.squeeze(0).transpose(0, 1)
        layers.append(torch.stack([k, v], dim=0))
    return torch.stack(layers, dim=0)

def unpack_kv(kv_tensor: torch.Tensor):
    """(L, 2, N, H_kv, D) → DynamicCache for newer transformers."""
    from transformers import DynamicCache
    cache = DynamicCache()
    for i in range(kv_tensor.shape[0]):
        k = kv_tensor[i, 0].permute(1, 0, 2).unsqueeze(0).half()  # (1, H_kv, N, D)
        v = kv_tensor[i, 1].permute(1, 0, 2).unsqueeze(0).half()
        cache.update(k, v, i)
    return cache

def get_model_layers(model) -> list:
    name = type(model).__name__.lower()
    if 'gpt2' in name:
        return list(model.transformer.h)
    if 'mistral' in name or 'llama' in name or 'mistral' in str(type(model.model)):
        return list(model.model.layers)
    raise NotImplementedError(f"Model {type(model).__name__} not supported.")

def get_embeddings(model, input_ids: torch.Tensor) -> torch.Tensor:
    name = type(model).__name__.lower()
    N = input_ids.shape[1]

    if 'gpt2' in name:
        pos_ids = torch.arange(N, device=input_ids.device).unsqueeze(0)
        tok_emb = model.transformer.wte(input_ids)
        pos_emb = model.transformer.wpe(pos_ids)
        return model.transformer.drop(tok_emb + pos_emb)

    if 'mistral' in name or 'llama' in name:
        return model.model.embed_tokens(input_ids)

    raise NotImplementedError

def apply_final_norm(model, hidden: torch.Tensor) -> torch.Tensor:
    name = type(model).__name__.lower()
    if 'gpt2' in name:
        return model.transformer.ln_f(hidden)
    if 'mistral' in name or 'llama' in name:
        return model.model.norm(hidden)
    raise NotImplementedError

# ─────────────────────────────────────────────────────────────────────────────
# Layer Execution (In-Place Mutations)
# ─────────────────────────────────────────────────────────────────────────────

def run_layer_full_gpt2_inplace(block, hidden: torch.Tensor, kv_layer_buf: torch.Tensor):
    """
    Run Layer 0 on ALL tokens and write KV directly into the pre-allocated buffer.
    """
    device = hidden.device
    N = hidden.shape[1]
    d = hidden.shape[2]
    n_heads  = block.attn.num_heads
    head_dim = d // n_heads

    ln1_out = block.ln_1(hidden)
    qkv     = block.attn.c_attn(ln1_out)
    Q, K, V = qkv.split(d, dim=-1)

    Q = Q.view(1, N, n_heads, head_dim).transpose(1, 2)
    K = K.view(1, N, n_heads, head_dim).transpose(1, 2)
    V = V.view(1, N, n_heads, head_dim).transpose(1, 2)

    # In-place write to KV buffer to avoid clones later
    kv_layer_buf[0] = K.squeeze(0).transpose(0, 1) # (N, H, D)
    kv_layer_buf[1] = V.squeeze(0).transpose(0, 1)

    mask = torch.tril(torch.ones(N, N, dtype=torch.bool, device=device)).view(1, 1, N, N)

    scale = head_dim ** -0.5
    attn_weights = torch.matmul(Q, K.transpose(-2, -1)) * scale
    attn_weights.masked_fill_(~mask, float('-inf')) # In-place mask
    attn_weights = F.softmax(attn_weights, dim=-1)
    attn_weights = torch.nan_to_num(attn_weights)
    attn_weights = block.attn.attn_dropout(attn_weights)

    attn_out = torch.matmul(attn_weights, V)
    attn_out = attn_out.transpose(1, 2).contiguous().view(1, N, d)
    attn_out = block.attn.c_proj(attn_out)
    attn_out = block.attn.resid_dropout(attn_out)

    hidden_after_attn = hidden + attn_out
    mlp_out           = block.mlp(block.ln_2(hidden_after_attn))

    # Mutate hidden in place
    hidden.copy_(hidden_after_attn + mlp_out)

    # Return fresh K, V as views of the buffer for deviation calculation
    return kv_layer_buf[0], kv_layer_buf[1]


def run_layer_selective_gpt2_inplace(
    block,
    full_hidden    : torch.Tensor,   # (1, N, d) - Mutated in place
    hkvd_indices   : torch.Tensor,   # (k,) int64
    KV_layer_buf   : torch.Tensor,   # (2, N, H, D) - Mutated in place
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Run on HKVD tokens. Mutates full_hidden and KV_layer_buf in-place to
    eliminate O(N) memory allocations.
    """
    N = full_hidden.shape[1]
    d = full_hidden.shape[2]
    k = hkvd_indices.shape[0]

    hkvd_hidden = full_hidden[:, hkvd_indices, :]

    ln1_out = block.ln_1(hkvd_hidden)
    qkv     = block.attn.c_attn(ln1_out)
    Q, K_fresh, V_fresh = qkv.split(d, dim=-1)

    n_heads  = block.attn.num_heads
    head_dim = d // n_heads

    Q = Q.view(1, k, n_heads, head_dim).transpose(1, 2)
    K_fresh_mh = K_fresh.view(k, n_heads, head_dim)
    V_fresh_mh = V_fresh.view(k, n_heads, head_dim)

    # Update KV Buffer IN-PLACE with fresh tokens
    KV_layer_buf[0, hkvd_indices] = K_fresh_mh
    KV_layer_buf[1, hkvd_indices] = V_fresh_mh

    # Zero-copy views for attention (now containing the fresh updates)
    K_full = KV_layer_buf[0].transpose(0, 1).unsqueeze(0) # (1, H, N, D)
    V_full = KV_layer_buf[1].transpose(0, 1).unsqueeze(0)

    # Vectorized causal mask generation (avoids Python loops)
    positions = hkvd_indices.unsqueeze(1) # (k, 1)
    idx_matrix = torch.arange(N, device=full_hidden.device).unsqueeze(0) # (1, N)
    mask = (idx_matrix <= positions).view(1, 1, k, N) # (1, 1, k, N)

    scale = head_dim ** -0.5
    attn_weights = torch.matmul(Q, K_full.transpose(-2, -1)) * scale
    attn_weights.masked_fill_(~mask, float('-inf'))
    attn_weights = F.softmax(attn_weights, dim=-1)
    attn_weights = torch.nan_to_num_(attn_weights) # In-place NaN clear
    attn_weights = block.attn.attn_dropout(attn_weights)

    attn_out = torch.matmul(attn_weights, V_full)
    attn_out = attn_out.transpose(1, 2).contiguous().view(1, k, d)
    attn_out = block.attn.c_proj(attn_out)
    attn_out = block.attn.resid_dropout(attn_out)

    hidden_after_attn = hkvd_hidden + attn_out
    mlp_out           = block.mlp(block.ln_2(hidden_after_attn))

    # Scatter updates back into full_hidden IN-PLACE
    full_hidden[0, hkvd_indices, :] = (hidden_after_attn + mlp_out)[0]

    return K_fresh_mh, V_fresh_mh



from transformers.models.llama.modeling_llama import apply_rotary_pos_emb, repeat_kv

def run_layer_full_llama_inplace(
    block, hidden: torch.Tensor, kv_layer_buf: torch.Tensor, position_ids: torch.Tensor,
    n_heads: int, n_kv_heads: int, head_dim: int, rotary_emb
):
    """Run Layer 0 on ALL tokens for LLaMA architectures, writing directly to buffer."""
    device = hidden.device
    N = hidden.shape[1]
    d = hidden.shape[2]

    hidden_norm = block.input_layernorm(hidden)

    Q = block.self_attn.q_proj(hidden_norm)
    K = block.self_attn.k_proj(hidden_norm)
    V = block.self_attn.v_proj(hidden_norm)

    Q = Q.view(1, N, n_heads, head_dim).transpose(1, 2)
    K = K.view(1, N, n_kv_heads, head_dim).transpose(1, 2)
    V = V.view(1, N, n_kv_heads, head_dim).transpose(1, 2)

    # Safely extract cos/sin across HuggingFace versions
    try:
        cos, sin = rotary_emb(V, position_ids)
    except TypeError:
        cos, sin = rotary_emb(V, seq_len=N)

    # Safely apply RoPE across HuggingFace versions
    try:
        Q, K = apply_rotary_pos_emb(Q, K, cos, sin, position_ids=position_ids)
    except TypeError:
        Q, K = apply_rotary_pos_emb(Q, K, cos, sin)

    kv_layer_buf[0] = K.squeeze(0).transpose(0, 1)
    kv_layer_buf[1] = V.squeeze(0).transpose(0, 1)

    num_key_value_groups = n_heads // n_kv_heads
    if num_key_value_groups > 1:
        K_full = repeat_kv(K, num_key_value_groups)
        V_full = repeat_kv(V, num_key_value_groups)
    else:
        K_full, V_full = K, V

    mask = torch.tril(torch.ones(N, N, dtype=torch.bool, device=device)).view(1, 1, N, N)

    scale = head_dim ** -0.5
    attn_weights = torch.matmul(Q, K_full.transpose(-2, -1)) * scale
    attn_weights.masked_fill_(~mask, float('-inf'))
    attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(Q.dtype)
    attn_weights = torch.nan_to_num_(attn_weights)

    attn_out = torch.matmul(attn_weights, V_full)
    attn_out = attn_out.transpose(1, 2).contiguous().view(1, N, d)
    attn_out = block.self_attn.o_proj(attn_out)

    hidden_after_attn = hidden + attn_out
    mlp_out = block.mlp(block.post_attention_layernorm(hidden_after_attn))

    hidden.copy_(hidden_after_attn + mlp_out)
    return kv_layer_buf[0], kv_layer_buf[1]


def run_layer_selective_llama_inplace(
    block, full_hidden: torch.Tensor, hkvd_indices: torch.Tensor,
    KV_layer_buf: torch.Tensor, position_ids: torch.Tensor,
    n_heads: int, n_kv_heads: int, head_dim: int, rotary_emb
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Selective execution for LLaMA/Mistral architectures."""
    N = full_hidden.shape[1]
    d = full_hidden.shape[2]
    k = hkvd_indices.shape[0]

    hkvd_hidden = full_hidden[:, hkvd_indices, :]
    hidden_norm = block.input_layernorm(hkvd_hidden)

    Q_fresh = block.self_attn.q_proj(hidden_norm)
    K_fresh = block.self_attn.k_proj(hidden_norm)
    V_fresh = block.self_attn.v_proj(hidden_norm)

    Q_fresh = Q_fresh.view(1, k, n_heads, head_dim).transpose(1, 2)
    K_fresh = K_fresh.view(1, k, n_kv_heads, head_dim).transpose(1, 2)
    V_fresh = V_fresh.view(1, k, n_kv_heads, head_dim).transpose(1, 2)

    hkvd_positions = position_ids[:, hkvd_indices]

    # Safely extract cos/sin across HuggingFace versions
    try:
      cos, sin = rotary_emb(V_fresh, hkvd_positions)
    except TypeError:
      cos, sin = rotary_emb(V_fresh, seq_len=N)
      # Manually index to get cos/sin for HKVD positions only
      if cos.dim() == 3:        # (1, N, D) or (batch, N, D)
          cos = cos[:, hkvd_indices]
          sin = sin[:, hkvd_indices]
      elif cos.dim() == 4:      # (1, 1, N, D)
          cos = cos[:, :, hkvd_indices]
          sin = sin[:, :, hkvd_indices]

    # Safely apply RoPE across HuggingFace versions
    try:
        Q_fresh, K_fresh = apply_rotary_pos_emb(Q_fresh, K_fresh, cos, sin, position_ids=hkvd_positions)
    except TypeError:
        Q_fresh, K_fresh = apply_rotary_pos_emb(Q_fresh, K_fresh, cos, sin)

    K_fresh_mh = K_fresh.squeeze(0).transpose(0, 1)
    V_fresh_mh = V_fresh.squeeze(0).transpose(0, 1)

    KV_layer_buf[0, hkvd_indices] = K_fresh_mh
    KV_layer_buf[1, hkvd_indices] = V_fresh_mh

    K_full = KV_layer_buf[0].transpose(0, 1).unsqueeze(0)
    V_full = KV_layer_buf[1].transpose(0, 1).unsqueeze(0)

    num_key_value_groups = n_heads // n_kv_heads
    if num_key_value_groups > 1:
        K_full = repeat_kv(K_full, num_key_value_groups)
        V_full = repeat_kv(V_full, num_key_value_groups)

    idx_matrix = torch.arange(N, device=full_hidden.device).unsqueeze(0)
    pos_col = hkvd_indices.unsqueeze(1)
    mask = (idx_matrix <= pos_col).view(1, 1, k, N)

    scale = head_dim ** -0.5
    attn_weights = torch.matmul(Q_fresh, K_full.transpose(-2, -1)) * scale
    attn_weights.masked_fill_(~mask, float('-inf'))
    attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(Q_fresh.dtype)
    attn_weights = torch.nan_to_num_(attn_weights)

    attn_out = torch.matmul(attn_weights, V_full)
    attn_out = attn_out.transpose(1, 2).contiguous().view(1, k, d)
    attn_out = block.self_attn.o_proj(attn_out)

    hidden_after_attn = hkvd_hidden + attn_out
    mlp_out = block.mlp(block.post_attention_layernorm(hidden_after_attn))

    full_hidden[0, hkvd_indices, :] = (hidden_after_attn + mlp_out)[0]

    return K_fresh_mh, V_fresh_mh


# ─────────────────────────────────────────────────────────────────────────────
# Systems-Level Deviation & Selection
# ─────────────────────────────────────────────────────────────────────────────

def compute_deviation_l2(
    fresh_K    : torch.Tensor,
    fresh_V    : torch.Tensor,
    cached_K   : torch.Tensor,
    cached_V   : torch.Tensor,
    miss_flags : torch.Tensor,
) -> torch.Tensor:
    """Computes deviation using squared L2 norm efficiently."""
    fK, fV = fresh_K.float(), fresh_V.float()
    cK, cV = cached_K.float(), cached_V.float()
    dev = torch.sum((fK - cK)**2 + (fV - cV)**2, dim=(-2, -1))
    dev = torch.sqrt_(dev)
    dev.masked_fill_(miss_flags, float('inf'))
    return dev

def select_topk(scores: torch.Tensor, k: int) -> torch.Tensor:
    k = min(k, scores.shape[0])
    _, idx = torch.topk(scores, k, sorted=False)
    return torch.sort(idx).values

# ─────────────────────────────────────────────────────────────────────────────
# Fusor Main Engine
# ─────────────────────────────────────────────────────────────────────────────

class CacheBlendFusor:
    def __init__(self, model, r: float = 0.15, r1_factor: float = 1.33):
        self.model    = model
        self.r        = r
        self.r1       = min(r * r1_factor, 1.0)
        self.layers   = get_model_layers(model)
        self.L        = len(self.layers)

        self.n_heads = model.config.num_attention_heads
        self.n_kv_heads = getattr(model.config, "num_key_value_heads", self.n_heads)
        self.head_dim = model.config.hidden_size // self.n_heads

        # Safely extract rotary embeddings for LLaMA architectures
        is_llama = 'llama' in type(self.model).__name__.lower() or 'mistral' in type(self.model).__name__.lower()
        if is_llama:
          if hasattr(self.model.model, 'rotary_emb'):
              self.rotary_emb = self.model.model.rotary_emb
          elif hasattr(self.layers[0].self_attn, 'rotary_emb'):
              self.rotary_emb = self.layers[0].self_attn.rotary_emb
          else:
              raise AttributeError(
                  "Cannot locate rotary_emb — check your transformers version"
              )
        else:
            self.rotary_emb = None

        self.model.eval()

    def fuse(
        self,
        full_input_ids : torch.Tensor,
        KV_new         : torch.Tensor,
        hit_mask       : torch.Tensor,
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:

        N = full_input_ids.shape[1]
        # KV_new = KV_new.float()
        KV_new = KV_new.to(self.model.dtype)
        hkvd_per_layer = []

        miss_count = int((~hit_mask).sum().item())

        k_targets = []
        for i in range(1, self.L):
            progress = i / (self.L - 1) if self.L > 1 else 1.0
            target_ratio = self.r1 + (self.r - self.r1) * progress
            k_targets.append(min(N, miss_count + max(1, int(target_ratio * N))))

        is_llama = self.rotary_emb is not None

        with torch.no_grad():
            hidden = get_embeddings(self.model, full_input_ids)

            if is_llama:
                position_ids = torch.arange(N, device=full_input_ids.device).unsqueeze(0)

            # -- LAYER 0 --
            cached_K0 = KV_new[0, 0].clone()
            cached_V0 = KV_new[0, 1].clone()

            if is_llama:
                fresh_K0, fresh_V0 = run_layer_full_llama_inplace(
                    self.layers[0], hidden, KV_new[0], position_ids,
                    self.n_heads, self.n_kv_heads, self.head_dim, self.rotary_emb
                )
            else:
                fresh_K0, fresh_V0 = run_layer_full_gpt2_inplace(self.layers[0], hidden, KV_new[0])

            dev0 = compute_deviation_l2(
                fresh_K0, fresh_V0, cached_K0, cached_V0, ~hit_mask
            )
            k0 = min(N, miss_count + max(1, int(self.r1 * N)))
            hkvd = select_topk(dev0, k0)
            hkvd_per_layer.append(hkvd)

            # -- LAYERS 1 to L-1 (Gradual Filtering) --
            for i in range(1, self.L):
                target_k = k_targets[i-1]

                cached_K_hkvd = KV_new[i, 0, hkvd].clone()
                cached_V_hkvd = KV_new[i, 1, hkvd].clone()

                if is_llama:
                    fresh_K_hkvd, fresh_V_hkvd = run_layer_selective_llama_inplace(
                        self.layers[i], hidden, hkvd, KV_new[i], position_ids,
                        self.n_heads, self.n_kv_heads, self.head_dim, self.rotary_emb
                    )
                else:
                    fresh_K_hkvd, fresh_V_hkvd = run_layer_selective_gpt2_inplace(
                        self.layers[i], hidden, hkvd, KV_new[i]
                    )

                dev_i = compute_deviation_l2(
                    fresh_K_hkvd, fresh_V_hkvd, cached_K_hkvd, cached_V_hkvd, ~hit_mask[hkvd]
                )

                local_top = select_topk(dev_i, min(len(hkvd), target_k))
                hkvd = hkvd[local_top]
                hkvd_per_layer.append(hkvd)

        return KV_new, hkvd_per_layer

    def get_stats(self, hkvd_per_layer: List[torch.Tensor], N: int) -> dict:
        counts = [len(h) for h in hkvd_per_layer]
        corrected_ops = N + sum(counts[1:])
        full_ops = N * self.L
        return {
            "N"                  : N,
            "L"                  : self.L,
            "hkvd_counts"        : counts,
            "hkvd_ratios"        : [c / N for c in counts],
            "layer0_ratio"       : counts[0] / N if counts else 0,
            "final_layer_ratio"  : counts[-1] / N if counts else 0,
            "target_r"           : self.r,
            "corrected_total_ops": corrected_ops,
            "full_prefill_ops"   : full_ops,
            "true_savings_pct"   : (1 - corrected_ops / full_ops) * 100,
        }
