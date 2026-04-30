import torch
import numpy as np
import hashlib
import os
from typing import Optional, List, Tuple
from sentence_transformers import SentenceTransformer
from .kv_store import KVCacheStore

class SemanticKVCacheStore(KVCacheStore):
    """
    A KV cache store that uses semantic embeddings to retrieve caches
    for similar text chunks, rather than requiring exact matches.
    
    It also adds hashing and disk persistence for more robust operation.
    """
    def __init__(
        self, 
        model_name: str = "all-MiniLM-L6-v2", 
        threshold: float = 0.85, 
        disk_path: Optional[str] = None,
        device: str = "cpu" # Default to CPU for embedding model to save VRAM
    ):
        super().__init__()
        self.encoder = SentenceTransformer(model_name, device=device)
        self.threshold = threshold
        self.disk_path = disk_path
        if disk_path:
            os.makedirs(disk_path, exist_ok=True)
            
        self._embeddings = [] # List of (embedding, key)
        self._keys_to_text = {} # Maps hash key back to original text for debugging

        # Load existing index if available
        if disk_path:
            self._load_index()

    def _index_path(self) -> str:
        return os.path.join(self.disk_path, "semantic_index.pt")

    def _load_index(self):
        path = self._index_path()
        if os.path.exists(path):
            try:
                data = torch.load(path, weights_only=False)
                self._embeddings = data.get("embeddings", [])
                self._keys_to_text = data.get("keys_to_text", {})
                print(f"Loaded semantic index with {len(self._embeddings)} entries.")
            except Exception as e:
                print(f"Failed to load semantic index: {e}")

    def _save_index(self):
        if self.disk_path:
            try:
                torch.save({
                    "embeddings": self._embeddings,
                    "keys_to_text": self._keys_to_text
                }, self._index_path())
            except Exception as e:
                print(f"Failed to save semantic index: {e}")

    def _key(self, text: str) -> str:
        return hashlib.sha256(text.encode()).hexdigest()

    def store(self, text: str, kv_tensor: torch.Tensor):
        """Stores the KV tensor with both exact-match hashing and semantic embedding."""
        key = self._key(text)
        
        # 1. Store in memory (base class uses _data dict)
        # Store on CPU to save VRAM
        self._data[key] = kv_tensor.cpu().clone()
        self._keys_to_text[key] = text
        
        # 2. Store on disk if enabled
        if self.disk_path:
            path = os.path.join(self.disk_path, f"{key}.pt")
            torch.save(kv_tensor.cpu(), path)
            
        # 3. Compute and store embedding for semantic retrieval
        embedding = self.encoder.encode(text, convert_to_numpy=True)
        self._embeddings.append((embedding, key))
        
        # 4. Update index on disk
        if self.disk_path:
            self._save_index()

    def load_semantic(self, text: str, device: str = "cuda") -> Tuple[Optional[torch.Tensor], float]:
        """
        Returns (kv_tensor, similarity_score). 
        If no match above threshold, returns (None, 0.0).
        """
        # 1. Try exact match first (efficiency)
        key = self._key(text)
        kv = self._data.get(key)
        
        if kv is None and self.disk_path:
            path = os.path.join(self.disk_path, f"{key}.pt")
            if os.path.exists(path):
                kv = torch.load(path, weights_only=False)
                self._data[key] = kv # Cache back to memory
        
        if kv is not None:
            return kv.to(device), 1.0

        if not self._embeddings:
            return None, 0.0

        # 2. Semantic search
        query_emb = self.encoder.encode(text, convert_to_numpy=True)
        
        best_sim = -1.0
        best_key = None
        
        for emb, k in self._embeddings:
            # Cosine similarity
            sim = np.dot(query_emb, emb) / (np.linalg.norm(query_emb) * np.linalg.norm(emb))
            if sim > best_sim:
                best_sim = sim
                best_key = k

        if best_sim >= self.threshold:
            kv = self._data.get(best_key)
            if kv is None and self.disk_path:
                path = os.path.join(self.disk_path, f"{best_key}.pt")
                if os.path.exists(path):
                    kv = torch.load(path, weights_only=False)
                    self._data[best_key] = kv
            
            if kv is not None:
                return kv.to(device), best_sim

        return None, best_sim

    def load(self, text: str, device: str = "cuda") -> Optional[torch.Tensor]:
        """Override for pipeline compatibility."""
        kv, _ = self.load_semantic(text, device=device)
        return kv
