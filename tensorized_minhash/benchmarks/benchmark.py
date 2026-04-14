"""
Benchmarks and validation for TensorizedMinHash.

Deliverables:
1. Memory profile: kron_params vs full_params for varying tensor sizes
2. Accuracy: correlation of Tensorized Jaccard vs ground-truth Jaccard
3. Speed: wall-clock hash time per tensor, TT vs Kron vs datasketch
4. RAM: peak memory usage, Kron vs datasketch vs theoretical full matrix

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
    """Exact Jaccard on binary tensors. J(A,B) = |A∩B| / |A∪B|"""
    a = tensor_a.astype(bool).ravel()
    b = tensor_b.astype(bool).ravel()
    intersection = np.sum(a & b)
    union        = np.sum(a | b)
    return float(intersection / union) if union > 0 else 0.0


def datasketch_jaccard(
    tensor_a: np.ndarray,
    tensor_b: np.ndarray,
    num_perm: int = 128,
) -> float:
    """
    Standard MinHash Jaccard using datasketch (flattened tensor).
    Falls back to exact Jaccard if datasketch is not installed.
    """
    try:
        from datasketch import MinHash
        m1 = MinHash(num_perm=num_perm)
        m2 = MinHash(num_perm=num_perm)
        for idx in np.where(tensor_a.ravel() > 0)[0]:
            m1.update(int(idx).to_bytes(8, byteorder="big"))
        for idx in np.where(tensor_b.ravel() > 0)[0]:
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
    For each tensor shape report kron_params, full_params, and compression ratio.
    Covers both 3rd-order and 4th-order tensors to show how compression grows
    with tensor order.
    """
    from core.tt_minhash import KroneckerMinHash, TTMinHashConfig

    results = []
    for shape in shapes:
        cfg    = TTMinHashConfig(shape=shape, num_hashes=num_hashes)
        hasher = KroneckerMinHash(cfg)
        stats  = hasher.memory_stats()
        results.append({
            "shape": shape,
            "total_cells": int(np.prod(shape)),
            **stats,
        })
        #logger.info(
        #    f"Shape {shape}: kron={stats['kron_params']:,} params "
        #    f"({stats['kron_bytes'] / 1024:.1f} KB) vs "
        #    f"full={stats['full_params_theoretical']:,} params "
        #    f"({stats['full_bytes_theoretical'] / 1_000_000:.1f} MB) | "
        #    f"compression {stats['compression_ratio']:,}x"
        #)
    return results


# ---------------------------------------------------------------------------
# Accuracy benchmark — synthetic data
# ---------------------------------------------------------------------------

def benchmark_accuracy(
    n_pairs: int = 200,
    shape: Tuple[int, ...] = (30, 30, 30),
    num_hashes: int = 128,
    seed: int = 0,
) -> Dict:
    """
    Accuracy benchmark on synthetic tensor pairs with varying density
    so Jaccard values span a wide range (improves Pearson r measurement).
    """
    from core.tt_minhash import KroneckerMinHash, TTMinHashConfig

    rng    = np.random.default_rng(seed)
    cfg    = TTMinHashConfig(shape=shape, num_hashes=num_hashes)
    hasher = KroneckerMinHash(cfg)

    exact_j, ds_j, kron_j = [], [], []

    for _ in range(n_pairs):
        # Vary density per pair so Jaccard spans [0, 1] — critical for Pearson r
        density = rng.uniform(0.01, 0.3)
        base    = (rng.random(shape) < density).astype(np.float32)
        noise_a = (rng.random(shape) < density * 0.5).astype(np.float32)
        noise_b = (rng.random(shape) < density * 0.5).astype(np.float32)
        a = np.clip(base + noise_a, 0, 1)
        b = np.clip(base + noise_b, 0, 1)

        exact_j.append(ground_truth_jaccard(a, b))

        sig_a = hasher.hash_tensor(a)
        sig_b = hasher.hash_tensor(b)
        kron_j.append(hasher.jaccard_from_signatures(sig_a, sig_b))

        ds_j.append(datasketch_jaccard(a, b, num_perm=num_hashes))

    return _compute_accuracy_results(
        np.array(exact_j), np.array(kron_j), np.array(ds_j),
        n_pairs=n_pairs, shape=shape, num_hashes=num_hashes,
    )


# ---------------------------------------------------------------------------
# Accuracy benchmark 
# ---------------------------------------------------------------------------

def benchmark_accuracy_from_tensors(
    tensors: list,
    shape: Tuple[int, ...],
    num_hashes: int = 128,
    n_pairs: int = 200,
    seed: int = 0,
) -> Dict:
    """
    Measure Jaccard approximation accuracy on real time-window tensors
    produced by Deliverable 3. Randomly samples n_pairs from available windows.
    """
    from core.tt_minhash import KroneckerMinHash, TTMinHashConfig

    cfg    = TTMinHashConfig(shape=shape, num_hashes=num_hashes)
    hasher = KroneckerMinHash(cfg)
    rng    = np.random.default_rng(seed)

    # Build all valid pairs then sample
    n_windows = len(tensors)
    all_pairs = [(i, j)
                 for i in range(n_windows)
                 for j in range(i + 1, n_windows)]

    if len(all_pairs) > n_pairs:
        sampled = rng.choice(len(all_pairs), size=n_pairs, replace=False)
        pairs   = [all_pairs[k] for k in sampled]
    else:
        pairs = all_pairs

    exact_j, kron_j, ds_j = [], [], []

    for i, j in pairs:
        a, b = tensors[i], tensors[j]

        exact_j.append(ground_truth_jaccard(a, b))

        sig_a = hasher.hash_tensor(a)
        sig_b = hasher.hash_tensor(b)
        kron_j.append(hasher.jaccard_from_signatures(sig_a, sig_b))

        ds_j.append(datasketch_jaccard(a, b, num_perm=num_hashes))

    results = _compute_accuracy_results(
        np.array(exact_j), np.array(kron_j), np.array(ds_j),
        n_pairs=len(pairs), shape=shape, num_hashes=num_hashes,
    )
    results["n_windows"]    = n_windows
    results["jaccard_range"] = (float(np.array(exact_j).min()),
                                float(np.array(exact_j).max()))
    return results


def _compute_accuracy_results(exact_j, kron_j, ds_j, n_pairs, shape, num_hashes):
    """Shared metrics computation for both accuracy functions."""

    def metrics(pred, true):
        mae  = float(np.mean(np.abs(pred - true)))
        rmse = float(np.sqrt(np.mean((pred - true) ** 2)))
        # Guard against zero-variance arrays
        if np.std(pred) < 1e-10 or np.std(true) < 1e-10:
            corr = 0.0
        else:
            corr = float(np.corrcoef(pred, true)[0, 1])
        return {"mae": mae, "rmse": rmse, "pearson_r": corr}

    results = {
        "tensorized_kron":     metrics(kron_j, exact_j),
        "datasketch_baseline": metrics(ds_j,   exact_j),
        "n_pairs":   n_pairs,
        "shape":     shape,
        "num_hashes": num_hashes,
        "jaccard_range": (float(exact_j.min()), float(exact_j.max())),
    }

    #logger.info(
    #    f"Accuracy (n={n_pairs}, shape={shape}): "
    #    f"Jaccard range=[{exact_j.min():.3f}, {exact_j.max():.3f}] | "
    #    f"Kron MAE={results['tensorized_kron']['mae']:.4f} "
    #    f"r={results['tensorized_kron']['pearson_r']:.4f} | "
    #    f"Datasketch MAE={results['datasketch_baseline']['mae']:.4f} "
    #    f"r={results['datasketch_baseline']['pearson_r']:.4f}"
    #)
    return results


# ---------------------------------------------------------------------------
# Speed benchmark — synthetic data
# ---------------------------------------------------------------------------

def benchmark_speed(
    shape: Tuple[int, ...] = (50, 50, 50),
    num_hashes: int = 128,
    n_tensors: int = 500,
    seed: int = 0,
) -> Dict:
    """
    Wall-clock hashing speed on synthetic tensors.
    Compares Kron, TT, and Datasketch.
    """
    from core.tt_minhash import KroneckerMinHash, TTDecomposedMinHash, TTMinHashConfig

    rng     = np.random.default_rng(seed)
    tensors = [(rng.random(shape) < 0.05).astype(np.float32) for _ in range(n_tensors)]

    return _run_speed_benchmark(tensors, num_hashes=num_hashes)


# ---------------------------------------------------------------------------
# Speed benchmark
# ---------------------------------------------------------------------------

def benchmark_speed_real(
    tensors: list,
    num_hashes: int = 128,
) -> Dict:
    """
    Wall-clock hashing speed on real time-window tensors from D3.
    Compares Kron, TT, and Datasketch.
    """
    return _run_speed_benchmark(tensors, num_hashes=num_hashes)


def _run_speed_benchmark(tensors: list, num_hashes: int = 128) -> Dict:
    """Shared speed benchmark logic for both synthetic and real tensors."""
    from core.tt_minhash import KroneckerMinHash, TTDecomposedMinHash, TTMinHashConfig

    shape     = tensors[0].shape
    n_tensors = len(tensors)
    results   = {}

    # Kron and TT
    for label, cls in [("Kron", KroneckerMinHash), ("TT", TTDecomposedMinHash)]:
        cfg    = TTMinHashConfig(shape=shape, num_hashes=num_hashes)
        hasher = cls(cfg)

        _ = hasher.hash_tensor(tensors[0])  # warm-up
        gc.collect()

        t0 = time.perf_counter()
        for t in tensors:
            hasher.hash_tensor(t)
        elapsed = time.perf_counter() - t0

        results[label] = {
            "total_sec":      elapsed,
            "per_tensor_ms":  elapsed / n_tensors * 1000,
            "tensors_per_sec": n_tensors / elapsed,
        }
        #logger.info(
        #    f"{label}: {n_tensors} tensors in {elapsed:.2f}s "
        #    f"({n_tensors/elapsed:.0f} tensors/sec, "
        #    f"{elapsed/n_tensors*1000:.2f} ms each)"
        #)

    # Datasketch
    try:
        from datasketch import MinHash

        # warm-up
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
            "total_sec":      elapsed,
            "per_tensor_ms":  elapsed / n_tensors * 1000,
            "tensors_per_sec": n_tensors / elapsed,
        }
        #logger.info(
        #    f"Datasketch: {n_tensors} tensors in {elapsed:.2f}s "
        #    f"({n_tensors/elapsed:.0f} tensors/sec, "
        #    f"{elapsed/n_tensors*1000:.2f} ms each)"
        #)
    except ImportError:
        logger.warning("datasketch not installed; skipping speed comparison")

    return results


# ---------------------------------------------------------------------------
# Peak RAM benchmark
# ---------------------------------------------------------------------------

def benchmark_ram(shape: Tuple[int, ...] = (50, 50, 50), num_hashes: int = 128) -> Dict:
    from core.tt_minhash import KroneckerMinHash, TTDecomposedMinHash, TTMinHashConfig

    tensor = np.zeros(shape, dtype=np.float32)
    tensor[0, 0, 0] = 1.0

    # ── Kron RAM ──
    gc.collect()
    tracemalloc.start()
    tracemalloc.clear_traces()

    cfg    = TTMinHashConfig(shape=shape, num_hashes=num_hashes)
    hasher = KroneckerMinHash(cfg)
    _      = hasher.hash_tensor(tensor)

    snapshot   = tracemalloc.take_snapshot()
    tracemalloc.stop()
    kron_bytes = sum(s.size for s in snapshot.statistics("lineno"))

    # ── TT RAM ──
    gc.collect()
    tracemalloc.start()
    tracemalloc.clear_traces()

    tt_hasher = TTDecomposedMinHash(cfg)
    _         = tt_hasher.hash_tensor(tensor)

    tt_snapshot = tracemalloc.take_snapshot()
    tracemalloc.stop()
    tt_bytes = sum(s.size for s in tt_snapshot.statistics("lineno"))

    # ── Theoretical counts ──
    full_theoretical_bytes = int(np.prod(shape)) * num_hashes * 4  # float32

    # TT theoretical: sum of core sizes
    r = cfg.tt_rank
    ranks = [num_hashes] + [r] * (cfg.ndim - 1) + [num_hashes]
    tt_theoretical_params = sum(
        ranks[k] * shape[k] * ranks[k+1]
        for k in range(cfg.ndim)
    )
    tt_theoretical_bytes = tt_theoretical_params * 4  # float32

    kron_params = sum(s * num_hashes for s in shape)

    result = {
        "kron_peak_bytes":         kron_bytes,
        "kron_peak_mb":            kron_bytes / 1_000_000,
        "tt_peak_bytes":           tt_bytes,
        "tt_peak_mb":              tt_bytes / 1_000_000,
        "full_theoretical_bytes":  full_theoretical_bytes,
        "full_theoretical_mb":     full_theoretical_bytes / 1_000_000,
        "kron_params":             kron_params,
        "tt_params":               tt_theoretical_params,
        "full_params":             int(np.prod(shape)) * num_hashes,
        "kron_vs_full":            full_theoretical_bytes / max(kron_bytes, 1),
        "tt_vs_full":              full_theoretical_bytes / max(tt_bytes, 1),
        "kron_vs_tt":              tt_bytes / max(kron_bytes, 1),
    }

    #logger.info(
    #    f"RAM: Kron={result['kron_peak_mb']:.2f} MB | "
    #    f"TT={result['tt_peak_mb']:.2f} MB | "
    #    f"Full(theoretical)={result['full_theoretical_mb']:.2f} MB | "
    #    f"Kron vs Full: {result['kron_vs_full']:.0f}x | "
    #    f"TT vs Full: {result['tt_vs_full']:.0f}x | "
    #    f"Kron vs TT: {result['kron_vs_tt']:.1f}x"
    #)
    return result

def benchmark_random_projection(
    tensors: list,
    num_hashes: int = 128,
    ) -> Dict:
    """
    Standard vectorization baseline: Random Projection on flattened tensors.
    Stores full W matrix of shape (prod(shape), num_hashes).
    This is what Kron replaces with factored matrices.
    """
    shape     = tensors[0].shape
    n_tensors = len(tensors)
    flat_dim  = int(np.prod(shape))

    # Memory cost
    full_matrix_mb = flat_dim * num_hashes * 4 / 1_000_000  # float32

    # Check if matrix is feasible to store
    if full_matrix_mb > 500:
        logger.warning(
            f"Full random projection matrix would need {full_matrix_mb:.1f} MB "
            f"— too large to allocate."
        )
        return {
            "feasible":        False,
            "theoretical_mb":  full_matrix_mb,
            "flat_dim":        flat_dim,
            "num_hashes":      num_hashes,
            "note": f"Requires {full_matrix_mb:.0f} MB — impractical at this scale",
        }

    # If feasible, time it
    rng = np.random.default_rng(42)
    W   = rng.standard_normal((flat_dim, num_hashes)).astype(np.float32)
    W  /= np.linalg.norm(W, axis=0)  # normalise columns

    # Warm-up
    _ = tensors[0].ravel() @ W
    gc.collect()

    t0 = time.perf_counter()
    for t in tensors:
        _ = t.ravel() @ W
    elapsed = time.perf_counter() - t0

    return {
        "feasible":        True,
        "theoretical_mb":  full_matrix_mb,
        "flat_dim":        flat_dim,
        "num_hashes":      num_hashes,
        "total_sec":       elapsed,
        "per_tensor_ms":   elapsed / n_tensors * 1000,
        "tensors_per_sec": n_tensors / elapsed,
    }

# ---------------------------------------------------------------------------
# Attack-pattern detection demo (kept for reference, uses synthetic data)
# ---------------------------------------------------------------------------

def demo_attack_detection(
    shape: Tuple[int, ...] = (50, 50, 50),
    num_hashes: int = 128,
    similarity_threshold: float = 0.5,
) -> None:
    import sys, os
    sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

    from data.loader import NetworkLogGenerator, NetworkTensorBuilder
    from core.tt_minhash import KroneckerMinHash, TTMinHashConfig
    from spark.distributed_hasher import LocalTensorHashPipeline

    n   = shape[0]
    gen = NetworkLogGenerator(n_src=n, n_dst=n, n_port=n, n_benign=3000, seed=99)
    df, attack_groups = gen.generate()

    builder = NetworkTensorBuilder(n_src=n, n_dst=n, n_port=n)
    tensors = builder.build_tensor_batch(df, window_size=500)
    ids     = [f"window_{i}" for i in range(len(tensors))]

    cfg      = TTMinHashConfig(shape=shape, num_hashes=num_hashes)
    hasher   = KroneckerMinHash(cfg)
    pipeline = LocalTensorHashPipeline(hasher)

    id_sig_pairs = pipeline.hash_all(tensors, ids, parallel=False)
    similar      = pipeline.find_similar_pairs(id_sig_pairs, threshold=similarity_threshold)

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
