import torch
import os
import shutil
from cacheblendplus.semantic_kv_store import SemanticKVCacheStore

def verify_semantic_store():
    print("--- Verifying SemanticKVCacheStore ---")
    
    # 1. Initialization
    disk_path = "./test_semantic_cache"
    if os.path.exists(disk_path):
        shutil.rmtree(disk_path)
        
    store = SemanticKVCacheStore(threshold=0.8, disk_path=disk_path)
    
    # 2. Store original content
    text_orig = "The Eiffel Tower is a wrought-iron lattice tower on the Champ de Mars in Paris, France."
    kv_orig = torch.ones((2, 10, 8, 64)) # Dummy KV
    
    print(f"\nStoring original text: \"{text_orig}\"")
    store.store(text_orig, kv_orig)
    
    # 3. Test Exact Match
    print("\nTesting Exact Match...")
    kv_exact, sim_exact = store.load_semantic(text_orig, device="cpu")
    print(f"Similarity: {sim_exact:.4f}")
    assert sim_exact == 1.0
    assert torch.allclose(kv_orig, kv_exact)
    print("✓ Exact match verified.")
    
    # 4. Test Semantic Match (Similar Meaning)
    text_sim = "Paris's famous iron tower, the Eiffel Tower, stands in the Champ de Mars."
    print(f"\nTesting Semantic Match with: \"{text_sim}\"")
    kv_sim, sim_sim = store.load_semantic(text_sim, device="cpu")
    print(f"Similarity: {sim_sim:.4f}")
    
    if kv_sim is not None and sim_sim >= store.threshold:
        print("✓ Semantic match found above threshold.")
        assert torch.allclose(kv_orig, kv_sim)
    else:
        print("✗ Semantic match failed or below threshold.")
        
    # 5. Test Non-Match
    text_diff = "Artificial Intelligence is transforming the modern world through machine learning."
    print(f"\nTesting Non-Match with: \"{text_diff}\"")
    kv_none, sim_none = store.load_semantic(text_diff, device="cpu")
    print(f"Similarity: {sim_none:.4f}")
    
    if kv_none is None:
        print("✓ Correctly returned None for dissimilar text.")
    else:
        print("✗ Incorrectly returned a match for dissimilar text.")

    # 6. Test Disk & Index Persistence
    print("\nTesting Disk & Index Persistence...")
    # Re-initialize store with the same disk path
    new_store = SemanticKVCacheStore(threshold=0.8, disk_path=disk_path)
    
    # 6a. Exact match after reload
    kv_disk_exact = new_store.load(text_orig, device="cpu")
    if kv_disk_exact is not None:
        print("✓ KV successfully loaded from disk via exact match.")
    else:
        print("✗ KV failed to load from disk via exact match.")

    # 6b. Semantic match after reload (proves index was persisted)
    print(f"Testing Semantic Match after reload with: \"{text_sim}\"")
    kv_disk_sim, sim_disk_sim = new_store.load_semantic(text_sim, device="cpu")
    print(f"Similarity after reload: {sim_disk_sim:.4f}")
    
    if kv_disk_sim is not None and sim_disk_sim >= new_store.threshold:
        print("✓ Semantic match found after reload (Index persistence verified).")
        assert torch.allclose(kv_orig, kv_disk_sim)
    else:
        print("✗ Semantic match failed after reload (Index persistence failed).")

    # Cleanup
    if os.path.exists(disk_path):
        shutil.rmtree(disk_path)
    print("\n--- Verification Complete ---")

if __name__ == "__main__":
    verify_semantic_store()
