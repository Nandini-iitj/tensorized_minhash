"""
benchmarks.speed — Hashing throughput and random-projection baseline benchmarks.

Compares Kron, TT, and Datasketch wall-clock hashing speed, and benchmarks
the full Random Projection matrix approach that Kronecker replaces.
"""

import gc
import logging
import time

import numpy as np

__all__ = ["benchmark_speed", "benchmark_speed_real", "benchmark_random_projection"]

logger = logging.getLogger(__name__)

def benchmark_speed(
    shape: tuple[int, ...] = (50, 50, 50),
    num_hashes: int = 128,
    n_tensors: int = 500,
    seed: int = 0,
) -> dict:
    """Wall-clock hashing speed on synthetic binary tensors (~5% density)."""
    rng = np.random.default_rng(seed)
    tensors = [(rng.random(shape) < 0.05).astype(np.float32) for _ in range(n_tensors)]
    return _run_speed_benchmark(tensors, num_hashes=num_hashes)

def benchmark_speed_real(
    tensors: list,
    num_hashes: int = 128,
) -> dict:
    """Wall-clock hashing speed on real time-window tensors from Deliverable 3."""
    return _run_speed_benchmark(tensors, num_hashes=num_hashes)

def _run_speed_benchmark(tensors: list, num_hashes: int = 128) -> dict:
    """Shared speed benchmark logic for both synthetic and real tensors."""
    from core import KroneckerMinHash, TTDecomposedMinHash, TTMinHashConfig
    
    shape = tensors[0].shape
    n_tensors = len(tensors)
    results = {}
    
    for label, cls in [("Kron", KroneckerMinHash), ("TT", TTDecomposedMinHash)]:
        cfg = TTMinHashConfig(shape=shape, num_hashes=num_hashes)
        hasher = cls(cfg)
        
        # warm-up
        _ = hasher.hash_tensor(tensors[0])
        gc.collect()
        
        t0 = time.perf_counter()
        for t in tensors:
            hasher.hash_tensor(t)
        elapsed = time.perf_counter() - t0
        
        results[label] = {
            "total_sec": elapsed,
            "per_tensor_ms": elapsed / n_tensors * 1000,
            "tensors_per_sec": n_tensors / elapsed,
        }
        
    # Datasketch
    try:
        from datasketch import MinHash
        
        m = MinHash(num_perm=num_hashes)
        for idx in np.where(tensors[0].ravel() > 0)[0]:
            m.update(int(idx).to_bytes(8, byteorder="big"))
        gc.collect()
        
        t0 = time.perf_counter()
        for t in tensors:
            m = MinHash(num_perm=num_hashes)
            for idx in np.where(t.ravel() > 0)[0]:
                m.update(int(idx).to_bytes(8, byteorder="big"))
        elapsed = time.perf_counter() - t0
        
        results["Datasketch"] = {
            "total_sec": elapsed,
            "per_tensor_ms": elapsed / n_tensors * 1000,
            "tensors_per_sec": n_tensors / elapsed,
        }
    except ImportError:
        logger.warning("datasketch not installed; skipping speed comparison")
        
    return results

def benchmark_random_projection(
    tensors: list,
    num_hashes: int = 128,
) -> dict:
    """
    Standard vectorization baseline: Random Projection on flattened tensors.
    Stores full W matrix of shape (prod(shape), num_hashes).
    This is what Kron replaces with factored matrices.
    """
    shape = tensors[0].shape
    n_tensors = len(tensors)
    flat_dim = int(np.prod(shape))
    
    full_matrix_mb = flat_dim * num_hashes * 4 / 1_000_000  # float32
    
    if full_matrix_mb > 500:
        logger.warning(
            f"Full random projection matrix would need {full_matrix_mb:.1f} MB "
            f"— too large to allocate."
        )
        return {
            "feasible": False,
            "theoretical_mb": full_matrix_mb,
            "flat_dim": flat_dim,
            "num_hashes": num_hashes,
            "note": f"Requires {full_matrix_mb:.0f} MB — impractical at this scale",
        }
        
    rng = np.random.default_rng(42)
    W = rng.standard_normal((flat_dim, num_hashes)).astype(np.float32)
    W /= np.linalg.norm(W, axis=0)  # normalize columns
    
    # warm-up
    _ = tensors[0].ravel() @ W
    gc.collect()
    
    t0 = time.perf_counter()
    for t in tensors:
        _ = t.ravel() @ W
    elapsed = time.perf_counter() - t0
    
    return {
        "feasible": True,
        "theoretical_mb": full_matrix_mb,
        "flat_dim": flat_dim,
        "num_hashes": num_hashes,
        "total_sec": elapsed,
        "per_tensor_ms": elapsed / n_tensors * 1000,
        "tensors_per_sec": n_tensors / elapsed,
    }