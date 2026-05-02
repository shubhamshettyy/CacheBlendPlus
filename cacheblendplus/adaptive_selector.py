import torch
import torch.nn.functional as F
from typing import Dict, List


class AdaptiveTokenSelector:
    """
    Paper-aligned adaptive selector:
    1) Compute token divergence from layer-0 fresh hidden states.
    2) Adapt recompute ratio from mean divergence.
    3) Return top-k divergent token indices (sorted ascending, int64).

    This is a drop-in selector for pipeline.py:
        indices = selector.select(chunk_ids, cached_kv)
    """

    def __init__(
        self,
        model,
        base_k_ratio: float = 0.15,
        low_thresh: float = 0.05,
        high_thresh: float = 0.20,
        min_k_ratio: float = 0.05,
        max_k_ratio: float = 0.30,
        eps: float = 1e-8,
        require_cuda: bool = True,
    ):
        self.model = model
        self.base_k_ratio = float(base_k_ratio)
        self.low_thresh = float(low_thresh)
        self.high_thresh = float(high_thresh)
        self.min_k_ratio = float(min_k_ratio)
        self.max_k_ratio = float(max_k_ratio)
        self.eps = float(eps)
        self.require_cuda = bool(require_cuda)

        if not (0.0 < self.min_k_ratio <= self.max_k_ratio <= 1.0):
            raise ValueError("Expected 0 < min_k_ratio <= max_k_ratio <= 1.")
        if not (0.0 <= self.low_thresh <= self.high_thresh):
            raise ValueError("Expected 0 <= low_thresh <= high_thresh.")
        if not (0.0 < self.base_k_ratio <= 1.0):
            raise ValueError("Expected 0 < base_k_ratio <= 1.")

        self.model.eval()
        self.history: List[Dict[str, float]] = []
        self.last_selection: Dict[str, float] = {}

    def _project_cached_for_cosine(self, cached_k_mean: torch.Tensor, target_dim: int) -> torch.Tensor:
        """
        Align cached representation dim to model hidden size for cosine.
        Uses deterministic pad/truncate to avoid learnable dependencies.
        """
        cached_dim = cached_k_mean.shape[-1]
        if cached_dim == target_dim:
            return cached_k_mean
        if cached_dim > target_dim:
            return cached_k_mean[:, :target_dim]

        pad = target_dim - cached_dim
        return F.pad(cached_k_mean, (0, pad), mode="constant", value=0.0)

    def _adaptive_ratio(self, mean_divergence: float) -> float:
        if mean_divergence <= self.low_thresh:
            return self.min_k_ratio
        if mean_divergence >= self.high_thresh:
            return self.max_k_ratio

        # Linear interpolation in the transition band.
        span = max(self.high_thresh - self.low_thresh, self.eps)
        alpha = (mean_divergence - self.low_thresh) / span
        return self.min_k_ratio + alpha * (self.max_k_ratio - self.min_k_ratio)

    def compute_r(self, chunk_ids: torch.Tensor, cached_kv: torch.Tensor) -> float:
        """
        Compute adaptive recompute ratio without selecting indices.
        Use with CacheBlendFusor.fuse(r=selector.compute_r(...)) for proper Extension C:
            adaptive_r = selector.compute_r(chunk_ids, cached_kv)
            blended_kv, stats = fusor.fuse(chunk_ids, cached_kv, hit_mask, r=adaptive_r)
        """
        if chunk_ids.ndim != 2 or chunk_ids.shape[0] != 1:
            raise ValueError(f"Expected chunk_ids shape (1, N), got {tuple(chunk_ids.shape)}")
        if cached_kv.ndim != 5 or cached_kv.shape[1] != 2:
            raise ValueError(f"Expected cached_kv shape (L, 2, N, H, D), got {tuple(cached_kv.shape)}")

        with torch.no_grad():
            attn_mask = torch.ones_like(chunk_ids)
            out = self.model(
                chunk_ids,
                attention_mask=attn_mask,
                use_cache=True,
                output_hidden_states=True,
                return_dict=True,
            )
            fresh_hidden = out.hidden_states[1].squeeze(0).float()  # (N, d_model)

        cached_k = cached_kv[0, 0].float()        # (N, H, D)
        cached_hidden = cached_k.mean(dim=1)       # (N, D)
        cached_hidden = self._project_cached_for_cosine(cached_hidden, fresh_hidden.shape[-1])

        fresh_norm  = F.normalize(fresh_hidden  + self.eps, p=2, dim=-1)
        cached_norm = F.normalize(cached_hidden + self.eps, p=2, dim=-1)
        cosine    = F.cosine_similarity(fresh_norm, cached_norm, dim=-1)
        divergence = (1.0 - cosine).clamp_min(0.0)
        mean_div   = float(divergence.mean().item())
        return self._adaptive_ratio(mean_div)

    def select(self, chunk_ids: torch.Tensor, cached_kv: torch.Tensor) -> torch.Tensor:
        """
        Args:
            chunk_ids: (1, N) token IDs
            cached_kv: (L, 2, N, H, D) cache tensor
        Returns:
            Tensor[k] int64, sorted ascending
        """
        if chunk_ids.ndim != 2 or chunk_ids.shape[0] != 1:
            raise ValueError(f"Expected chunk_ids shape (1, N), got {tuple(chunk_ids.shape)}")
        if cached_kv.ndim != 5 or cached_kv.shape[1] != 2:
            raise ValueError(f"Expected cached_kv shape (L, 2, N, H, D), got {tuple(cached_kv.shape)}")
        if self.require_cuda and (chunk_ids.device.type != "cuda" or cached_kv.device.type != "cuda"):
            raise ValueError("AdaptiveTokenSelector requires CUDA tensors.")

        n_tokens = int(chunk_ids.shape[1])
        if int(cached_kv.shape[2]) != n_tokens:
            raise ValueError("chunk_ids length must match cached_kv token dimension.")

        with torch.no_grad():
            # We use an all-ones mask because chunk inputs here are unpadded.
            attn_mask = torch.ones_like(chunk_ids, device=chunk_ids.device)
            out = self.model(
                chunk_ids,
                attention_mask=attn_mask,
                use_cache=True,
                output_hidden_states=True,
                return_dict=True,
            )
            # hidden_states[0] is embedding output; hidden_states[1] is post layer-0.
            fresh_hidden = out.hidden_states[1].squeeze(0).float()  # (N, d_model)

        cached_k = cached_kv[0, 0].float()  # (N, H, D)
        cached_hidden = cached_k.mean(dim=1)  # (N, D)
        cached_hidden = self._project_cached_for_cosine(cached_hidden, fresh_hidden.shape[-1])

        fresh_norm = F.normalize(fresh_hidden + self.eps, p=2, dim=-1)
        cached_norm = F.normalize(cached_hidden + self.eps, p=2, dim=-1)

        cosine = F.cosine_similarity(fresh_norm, cached_norm, dim=-1)
        divergence = (1.0 - cosine).clamp_min(0.0)

        mean_div = float(divergence.mean().item())
        adaptive_ratio = self._adaptive_ratio(mean_div)

        k = max(1, int(adaptive_ratio * n_tokens))
        k = min(k, n_tokens)
        indices = torch.topk(divergence, k, largest=True, sorted=False).indices
        indices = torch.sort(indices).values.to(dtype=torch.int64)

        self.last_selection = {
            "sequence_length": float(n_tokens),
            "mean_divergence": mean_div,
            "selected_k_ratio": float(adaptive_ratio),
            "selected_k": float(k),
            "base_k_ratio": float(self.base_k_ratio),
        }
        self.history.append(self.last_selection.copy())

        return indices

    def get_last_selection_stats(self) -> Dict[str, float]:
        return self.last_selection.copy()

    def get_selection_history(self) -> List[Dict[str, float]]:
        return list(self.history)

    def reset_history(self) -> None:
        self.history.clear()
        self.last_selection = {}


# Alias for simpler imports in scripts/notebooks.
AdaptiveSelector = AdaptiveTokenSelector
