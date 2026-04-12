"""
Benchmarks and validation for TensorizedMinHash.

Deliverables:
1. Memory profile: kron_params vs full_params for varying tensor sizes
2. Accuracy: correlation of Tensorized Jaccard vs ground-truth Jaccard
3. Speed: wall-clock hash time per tensor, TT vs Kron vs standard
4. Scalability: throughput (tensors/sec) vs batch size

Validation against datasketch (standard MinHash) on flattened tensors.
"""

import time
import numpy as np
import gc
import tracemalloc
from typing import List, Dict, Tuple
import logging

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def ground_truth_jaccard(tensor_a: np.ndarray, tensor_b: np.ndarray) -> float:
    """
    Exact Jaccard on binary tensors.
    J(A, B) = |A ∩ B| / |A ∪ B|
    """
    a = tensor_a.astype(bool).ravel()
    b = tensor_b.astype(bool).ravel()
    intersection = np.sum(a & b)
    union = np.sum(a | b)
    return float(intersection / union) if union > 0 else 0.0


def datasketch_jaccard(
    tensor_a: np.ndarray,
    tensor_b: np.ndarray,
    num_perm: int = 128,
) -> float:
    """
    Standard MinHash Jaccard using datasketch (flattened tensor).
    Falls back to exact if datasketch not installed.
    """
    try:
        from datasketch import MinHash
        m1 = MinHash(num_perm=num_perm)
        m2 = MinHash(num_perm=num_perm)
        a_flat = np.where(tensor_a.ravel() > 0)[0]
        b_flat = np.where(tensor_b.ravel() > 0)[0]
        for idx in a_flat:
            m1.update(int(idx).to_bytes(8, byteorder="big"))
        for idx in b_flat:
            m2.update(int(idx).to_bytes(8, byteorder="big"))
        return m1.jaccard(m2)
    except ImportError:
        logger.warning("datasketch not installed; falling back to exact Jaccard for baseline")
        return ground_truth_jaccard(tensor_a, tensor_b)


# ---------------------------------------------------------------------------
# Memory benchmark
# ---------------------------------------------------------------------------

def benchmark_memory(shapes: List[Tuple[int, ...]], num_hashes: int = 128) -> List[Dict]:
    """
    For each tensor shape, report:
      - kron_params: # parameters in Kronecker factor matrices
      - full_params: # parameters if full projection matrix stored
      - compression: ratio
      - kron_bytes: memory footprint of Kron approach (float32)
    """
    from core.tt_minhash import KroneckerMinHash, TTMinHashConfig

    results = []
    for shape in shapes:
        cfg = TTMinHashConfig(shape=shape, num_hashes=num_hashes)
        hasher = KroneckerMinHash(cfg)
        stats = hasher.memory_stats()
        results.append({
            "shape": shape,
            "total_cells": int(np.prod(shape)),
            **stats,
        })
        logger.info(
            f"Shape {shape}: kron={stats['kron_params']:,} params "
            f"({stats['kron_bytes'] / 1024:.1f} KB) vs "
            f"full={stats['full_params_theoretical']:,} params "
            f"({stats['full_bytes_theoretical'] / 1_000_000:.1f} MB) | "
            f"compression {stats['compression_ratio']:,}x"
        )
    return results


# ---------------------------------------------------------------------------
# Accuracy benchmark
# ---------------------------------------------------------------------------

def benchmark_accuracy(
    n_pairs: int = 200,
    shape: Tuple[int, ...] = (30, 30, 30),
    num_hashes: int = 128,
    density: float = 0.05,
    seed: int = 0,
) -> Dict:
    """
    Generate n_pairs of random binary tensors with known Jaccard similarities.
    Compare: ground truth, datasketch baseline, tensorized MinHash.

    Returns dict with MAE, RMSE, and Pearson correlation.
    """
    from core.tt_minhash import KroneckerMinHash, TTMinHashConfig

    rng = np.random.default_rng(seed)
    cfg = TTMinHashConfig(shape=shape, num_hashes=num_hashes)
    hasher = KroneckerMinHash(cfg)

    exact_j, ds_j, kron_j = [], [], []

    for _ in range(n_pairs):
        # Create two tensors with partial overlap
        base = (rng.random(shape) < density).astype(np.float32)
        noise_a = (rng.random(shape) < density * 0.5).astype(np.float32)
        noise_b = (rng.random(shape) < density * 0.5).astype(np.float32)
        a = np.clip(base + noise_a, 0, 1)
        b = np.clip(base + noise_b, 0, 1)

        exact_j.append(ground_truth_jaccard(a, b))

        sig_a = hasher.hash_tensor(a)
        sig_b = hasher.hash_tensor(b)
        kron_j.append(hasher.jaccard_from_signatures(sig_a, sig_b))

        ds_j.append(datasketch_jaccard(a, b, num_perm=num_hashes))

    exact_j = np.array(exact_j)
    kron_j = np.array(kron_j)
    ds_j = np.array(ds_j)

    def metrics(pred, true):
        mae = float(np.mean(np.abs(pred - true)))
        rmse = float(np.sqrt(np.mean((pred - true) ** 2)))
        corr = float(np.corrcoef(pred, true)[0, 1])
        return {"mae": mae, "rmse": rmse, "pearson_r": corr}

    results = {
        "tensorized_kron": metrics(kron_j, exact_j),
        "datasketch_baseline": metrics(ds_j, exact_j),
        "n_pairs": n_pairs,
        "shape": shape,
        "num_hashes": num_hashes,
    }

    logger.info(
        f"Accuracy (n={n_pairs}, shape={shape}): "
        f"Kron MAE={results['tensorized_kron']['mae']:.4f} "
        f"r={results['tensorized_kron']['pearson_r']:.4f} | "
        f"Datasketch MAE={results['datasketch_baseline']['mae']:.4f} "
        f"r={results['datasketch_baseline']['pearson_r']:.4f}"
    )
    return results


# ---------------------------------------------------------------------------
# Speed benchmark
# ---------------------------------------------------------------------------

def benchmark_speed(
    shape: Tuple[int, ...] = (50, 50, 50),
    num_hashes: int = 128,
    n_tensors: int = 500,
    seed: int = 0,
) -> Dict:
    """
    Wall-clock time to hash n_tensors with each method.
    Returns per-tensor throughput (tensors/sec).
    """
    from core.tt_minhash import (
        KroneckerMinHash, TTDecomposedMinHash, TTMinHashConfig
    )

    rng = np.random.default_rng(seed)
    tensors = [(rng.random(shape) < 0.05).astype(np.float32) for _ in range(n_tensors)]

    results = {}

    for label, cls in [("kron", KroneckerMinHash), ("tt", TTDecomposedMinHash)]:
        cfg = TTMinHashConfig(shape=shape, num_hashes=num_hashes)
        hasher = cls(cfg)

        # Warm-up
        _ = hasher.hash_tensor(tensors[0])
        gc.collect()

        t0 = time.perf_counter()
        for t in tensors:
            hasher.hash_tensor(t)
        elapsed = time.perf_counter() - t0

        throughput = n_tensors / elapsed
        results[label] = {
            "total_sec": elapsed,
            "per_tensor_ms": elapsed / n_tensors * 1000,
            "tensors_per_sec": throughput,
        }
        logger.info(
            f"{label}: {n_tensors} tensors in {elapsed:.2f}s "
            f"({throughput:.0f} tensors/sec, "
            f"{elapsed/n_tensors*1000:.2f} ms each)"
        )

    return results


# ---------------------------------------------------------------------------
# Peak RAM benchmark using tracemalloc
# ---------------------------------------------------------------------------

def benchmark_ram(shape: Tuple[int, ...] = (50, 50, 50), num_hashes: int = 128) -> Dict:
    """
    Measure peak RAM during hash-parameter allocation.
    Compares Kronecker (small) vs hypothetical full matrix (large).
    """
    from core.tt_minhash import KroneckerMinHash, TTMinHashConfig

    gc.collect()
    tracemalloc.start()
    tracemalloc.clear_traces()

    cfg = TTMinHashConfig(shape=shape, num_hashes=num_hashes)
    hasher = KroneckerMinHash(cfg)
    tensor = np.zeros(shape, dtype=np.float32)
    tensor[0, 0, 0] = 1.0
    _ = hasher.hash_tensor(tensor)

    snapshot = tracemalloc.take_snapshot()
    tracemalloc.stop()

    top_stats = snapshot.statistics("lineno")
    peak_bytes = sum(s.size for s in top_stats)

    full_theoretical_bytes = int(np.prod(shape)) * num_hashes * 4

    result = {
        "kron_peak_bytes": peak_bytes,
        "kron_peak_mb": peak_bytes / 1_000_000,
        "full_theoretical_bytes": full_theoretical_bytes,
        "full_theoretical_mb": full_theoretical_bytes / 1_000_000,
        "ram_compression": full_theoretical_bytes / max(peak_bytes, 1),
    }
    logger.info(
        f"RAM: Kron={result['kron_peak_mb']:.2f} MB peak | "
        f"Full matrix (theoretical)={result['full_theoretical_mb']:.2f} MB | "
        f"Savings: {result['ram_compression']:.1f}x"
    )
    return result


# ---------------------------------------------------------------------------
# Attack-pattern detection demo
# ---------------------------------------------------------------------------

def demo_attack_detection(
    shape: Tuple[int, ...] = (50, 50, 50),
    num_hashes: int = 128,
    similarity_threshold: float = 0.5,
) -> None:
    """
    End-to-end demo:
    1. Generate benign + attack traffic
    2. Build tensors per time window
    3. Hash with Kronecker MinHash
    4. Find similar tensor pairs → flag as repeated attack patterns
    """
    import sys, os
    sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

    from data.loader import NetworkLogGenerator, NetworkTensorBuilder
    from core.tt_minhash import KroneckerMinHash, TTMinHashConfig
    from spark.distributed_hasher import LocalTensorHashPipeline

    n = shape[0]
    gen = NetworkLogGenerator(n_src=n, n_dst=n, n_port=n, n_benign=3000, seed=99)
    df, attack_groups = gen.generate()

    builder = NetworkTensorBuilder(n_src=n, n_dst=n, n_port=n)
    tensors = builder.build_tensor_batch(df, window_size=500)
    ids = [f"window_{i}" for i in range(len(tensors))]

    cfg = TTMinHashConfig(shape=shape, num_hashes=num_hashes)
    hasher = KroneckerMinHash(cfg)
    pipeline = LocalTensorHashPipeline(hasher)

    id_sig_pairs = pipeline.hash_all(tensors, ids, parallel=False)
    similar = pipeline.find_similar_pairs(id_sig_pairs, threshold=similarity_threshold)

    print(f"\n{'='*60}")
    print(f"Attack Pattern Detection Demo  (shape={shape})")
    print(f"{'='*60}")
    print(f"Windows: {len(tensors)}  |  Threshold: {similarity_threshold}")
    print(f"Kron params: {hasher.memory_stats()['kron_params']:,}  "
          f"(vs {hasher.memory_stats()['full_params_theoretical']:,} full)")
    print(f"\nSimilar window pairs (potential repeated attacks):")
    if similar:
        for a, b, j in similar[:10]:
            print(f"  {a} ↔ {b}  Jaccard ≈ {j:.3f}")
    else:
        print("  No pairs above threshold (try lowering threshold)")
    print(f"{'='*60}\n")
