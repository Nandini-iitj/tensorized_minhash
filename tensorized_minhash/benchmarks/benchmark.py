"""
Benchmarks and validation for TensorizedMinHash.

Deliverables:
1. Memory profile: kron_params vs full_params for varying tensor sizes
2. Accuracy: correlation of Tensorized Jaccard vs ground-truth Jaccard
3. Speed: Wall-clock hash time per tensor, TT vs Kron vs datasketch
4. RAM: peak memory usage, Kron vs datasketch vs theoretical full matrix

Validation against datasketch (standard MinHash) on flattened tensors.
"""

import gc
import logging
import time

import numpy as np
from scipy.stats import spearmanr

__all__ = [
    "benchmark_memory",
    "benchmark_accuracy",
    "benchmark_accuracy_from_tensors",
    "benchmark_speed",
    "benchmark_speed_real",
    "benchmark_ram",
    "benchmark_random_projection",
    "ground_truth_jaccard",
    "datasketch_jaccard",
]

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------

def ground_truth_jaccard(tensor_a: np.ndarray, tensor_b: np.ndarray) -> float:
    """Exact Jaccard on binary tensors. J(A,B) = |AnB| / |AUB|"""
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


# --------------------------------------------------------------------------
# Memory benchmark
# --------------------------------------------------------------------------

def benchmark_memory(shapes: list[tuple[int, ...]], num_hashes: int = 128) -> list[dict]:
    """
    For each tensor shape report kron_params, full_params, and compression ratio.
    Covers both 3rd-order and 4th-order tensors to show how compression grows
    with tensor order.
    """
    from core.config import TTMinHashConfig
    from core.kron_minhash import KroneckerMinHash

    results = []
    for shape in shapes:
        cfg = TTMinHashConfig(shape=shape, num_hashes=num_hashes)
        hasher = KroneckerMinHash(cfg)
        stats = hasher.memory_stats()
        results.append(
            {
                "shape": shape,
                "total_cells": int(np.prod(shape)),
                **stats,
            }
        )
    return results


# --------------------------------------------------------------------------
# Accuracy benchmark — 3 similarity scenarios
# --------------------------------------------------------------------------

def _generate_high_similarity_pair(
    rng: np.random.Generator,
    shape: tuple[int, ...],
    density: float = 0.10,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Scenario 1 - Near-identical tensors (target J ≈ 0.85-0.98).
    Both tensors share the same base; only a small fraction of cells differ.
    Real-world analogy: repeated attack pattern across time windows.
    """
    all_cells = np.arange(int(np.prod(shape)))
    n_base = max(1, int(density * len(all_cells)))
    base_cells = rng.choice(all_cells, n_base, replace=False)

    # Flip a tiny fraction (<=10%) of cells independently in each tensor
    n_flip = max(1, int(0.05 * n_base))
    flip_a = rng.choice(base_cells, n_flip, replace=False)
    flip_b = rng.choice(base_cells, n_flip, replace=False)

    set_a = set(base_cells) | set(flip_a) - set(flip_b)
    set_b = set(base_cells) | set(flip_b) - set(flip_a)

    a = np.zeros(shape, dtype=np.float32)
    a.ravel()[list(set_a)] = 1.0
    b = np.zeros(shape, dtype=np.float32)
    b.ravel()[list(set_b)] = 1.0
    return a, b


def _generate_medium_similarity_pair(
    rng: np.random.Generator,
    shape: tuple[int, ...],
    density: float = 0.08,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Scenario 2 - Moderate overlap (target J ≈ 0.35-0.65).
    Both tensors share a common base but each adds substantial independent noise.
    Real-world analogy: two time windows with overlapping but distinct traffic.
    """
    all_cells = np.arange(int(np.prod(shape)))
    n_base = max(1, int(density * len(all_cells)))
    n_unique = max(1, int(density * len(all_cells)))

    base_cells = rng.choice(all_cells, n_base, replace=False)
    unique_a = rng.choice(all_cells, n_unique, replace=False)
    unique_b = rng.choice(all_cells, n_unique, replace=False)

    set_a = set(base_cells) | set(unique_a)
    set_b = set(base_cells) | set(unique_b)

    a = np.zeros(shape, dtype=np.float32)
    a.ravel()[list(set_a)] = 1.0
    b = np.zeros(shape, dtype=np.float32)
    b.ravel()[list(set_b)] = 1.0
    return a, b


def _generate_low_similarity_pair(
    rng: np.random.Generator,
    shape: tuple[int, ...],
    density: float = 0.06,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Scenario 3 - Near-disjoint tensors (target J ≈ 0.01-0.12).
    Each tensor is independently generated with no shared base.
    Real-world analogy: two completely different attack types.
    """
    all_cells = np.arange(int(np.prod(shape)))
    n_cells = max(1, int(density * len(all_cells)))

    set_a = rng.choice(all_cells, n_cells, replace=False)
    set_b = rng.choice(all_cells, n_cells, replace=False)

    a = np.zeros(shape, dtype=np.float32)
    a.ravel()[set_a] = 1.0
    b = np.zeros(shape, dtype=np.float32)
    b.ravel()[set_b] = 1.0
    return a, b


def _hash_pair(
    a: np.ndarray,
    b: np.ndarray,
    hasher,
    tt_hasher,
    num_hashes: int,
) -> tuple[float, float, float, float]:
    """Hash one pair with all three methods. Returns (exact, kron, tt, ds)."""
    exact = ground_truth_jaccard(a, b)
    kron = hasher.jaccard_from_signatures(hasher.hash_tensor(a), hasher.hash_tensor(b))
    tt = tt_hasher.jaccard_from_signatures(tt_hasher.hash_tensor(a), tt_hasher.hash_tensor(b))
    ds = datasketch_jaccard(a, b, num_perm=num_hashes)
    return exact, kron, tt, ds


def benchmark_accuracy(
    n_pairs: int = 100,
    shape: tuple[int, ...] = (30, 30, 30),
    num_hashes: int = 128,
    seed: int = 0,
) -> dict:
    """
    Three-scenario accuracy benchmark covering the full Jaccard range.

    Runs n_pairs pairs under each of three similarity regimes:
    Scenario 1 - High similarity (J ≈ 0.85-0.98): near-identical tensors
    Scenario 2 - Medium similarity (J ≈ 0.35-0.65): shared base + noise
    Scenario 3 - Low similarity    (J ≈ 0.01-0.12): independently random

    Metrics are reported per-scenario AND combined across all 3*n_pairs pairs.
    The combined Spearman ρ is the primary proposal evaluation metric because
    it spans the full [0, 1] Jaccard range, giving rank correlation real signal.
    """
    from core.config import TTMinHashConfig
    from core.kron_minhash import KroneckerMinHash
    from core.tt_minhash import TTDecomposedMinHash

    rng = np.random.default_rng(seed)
    cfg = TTMinHashConfig(shape=shape, num_hashes=num_hashes)
    hasher = KroneckerMinHash(cfg)
    tt_hasher = TTDecomposedMinHash(cfg)

    generators = [
        ("high", _generate_high_similarity_pair),
        ("medium", _generate_medium_similarity_pair),
        ("low", _generate_low_similarity_pair),
    ]

    scenario_results = {}
    all_exact, all_kron, all_tt, all_ds = [], [], [], []

    for scenario_name, gen_fn in generators:
        exact_j, kron_j, tt_j, ds_j = [], [], [], []

        for _ in range(n_pairs):
            a, b = gen_fn(rng, shape)
            exact, kron, tt, ds = _hash_pair(a, b, hasher, tt_hasher, num_hashes)
            exact_j.append(exact)
            kron_j.append(kron)
            tt_j.append(tt)
            ds_j.append(ds)

        scenario_results[scenario_name] = _compute_accuracy_results(
            np.array(exact_j),
            np.array(kron_j),
            np.array(ds_j),
            np.array(tt_j),
            n_pairs=n_pairs,
            shape=shape,
            num_hashes=num_hashes,
        )
        all_exact.extend(exact_j)
        all_kron.extend(kron_j)
        all_tt.extend(tt_j)
        all_ds.extend(ds_j)

    # Combined results over all three scenarios — primary Spearman ρ metric
    combined = _compute_accuracy_results(
        np.array(all_exact),
        np.array(all_kron),
        np.array(all_ds),
        np.array(all_tt),
        n_pairs=3 * n_pairs,
        shape=shape,
        num_hashes=num_hashes,
    )

    return {
        "scenarios": scenario_results,
        "combined": combined,
        "tensorized_kron": combined["tensorized_kron"],
        "tt_decomposition": combined["tt_decomposition"],
        "datasketch_baseline": combined["datasketch_baseline"],
        "n_pairs": combined["n_pairs"],
        "shape": combined["shape"],
        "num_hashes": combined["num_hashes"],
        "jaccard_range": combined["jaccard_range"],
    }


def benchmark_accuracy_from_tensors(
    tensors: list,
    shape: tuple[int, ...] = None,
    num_hashes: int = 128,
    n_pairs: int = 200,
    seed: int = 0,
) -> dict:
    """
    Measure Jaccard approximation accuracy on real time-window tensors
    produced by Deliverable 3. Randomly samples n_pairs from available windows.
    """
    from core.config import TTMinHashConfig
    from core.kron_minhash import KroneckerMinHash
    from core.tt_minhash import TTDecomposedMinHash

    if shape is None:
        shape = tensors[0].shape

    cfg = TTMinHashConfig(shape=shape, num_hashes=num_hashes)
    hasher = KroneckerMinHash(cfg)
    tt_hasher = TTDecomposedMinHash(cfg)
    rng = np.random.default_rng(seed)

    # Build all valid pairs then sample
    n_windows = len(tensors)
    all_pairs = [(i, j) for i in range(n_windows) for j in range(i + 1, n_windows)]

    if len(all_pairs) > n_pairs:
        sampled_indices = rng.choice(len(all_pairs), size=n_pairs, replace=False)
        pairs = [all_pairs[k] for k in sampled_indices]
    else:
        pairs = all_pairs

    exact_j, kron_j, ds_j, tt_j = [], [], [], []

    for i, j in pairs:
        a, b = tensors[i], tensors[j]
        exact_j.append(ground_truth_jaccard(a, b))

        sig_a = hasher.hash_tensor(a)
        sig_b = hasher.hash_tensor(b)
        kron_j.append(hasher.jaccard_from_signatures(sig_a, sig_b))

        sig_a_tt = tt_hasher.hash_tensor(a)
        sig_b_tt = tt_hasher.hash_tensor(b)
        tt_j.append(tt_hasher.jaccard_from_signatures(sig_a_tt, sig_b_tt))

        ds_j.append(datasketch_jaccard(a, b, num_perm=num_hashes))

    results = _compute_accuracy_results(
        np.array(exact_j),
        np.array(kron_j),
        np.array(ds_j),
        np.array(tt_j),
        n_pairs=len(pairs),
        shape=shape,
        num_hashes=num_hashes,
    )
    results["n_windows"] = n_windows
    results["jaccard_range"] = (float(np.array(exact_j).min()), float(np.array(exact_j).max()))
    return results


def _compute_accuracy_results(exact_j, kron_j, ds_j, tt_j, n_pairs, shape, num_hashes):
    """Shared metrics computation for both accuracy functions."""

    def metrics(pred, true):
        mae = float(np.mean(np.abs(pred - true)))
        rmse = float(np.sqrt(np.mean((pred - true) ** 2)))
        # Guard against zero-variance arrays (e.g. all predictions identical)
        if np.std(pred) < 1e-10 or np.std(true) < 1e-10:
            pearson = 0.0
            spearman = 0.0
        else:
            pearson = float(np.corrcoef(pred, true)[0, 1])
            spearman = float(spearmanr(pred, true).statistic)
        return {
            "avg_estimated": float(np.mean(pred)),
            "mae": mae,
            "rmse": rmse,
            "pearson_r": pearson,
            "spearman_r": spearman,
        }

    results = {
        "tensorized_kron": metrics(kron_j, exact_j),
        "tt_decomposition": metrics(tt_j, exact_j),
        "datasketch_baseline": metrics(ds_j, exact_j),
        "avg_exact": float(np.mean(exact_j)),
        "n_pairs": n_pairs,
        "shape": shape,
        "num_hashes": num_hashes,
        "jaccard_range": (float(exact_j.min()), float(exact_j.max())),
    }

    return results


# --------------------------------------------------------------------------
# Speed benchmark — synthetic data
# --------------------------------------------------------------------------

def benchmark_speed(
    shape: tuple[int, ...] = (50, 50, 50),
    num_hashes: int = 128,
    n_tensors: int = 500,
    seed: int = 0,
) -> dict:
    """
    Wall-clock hashing speed on synthetic tensors.
    Compares Kron, TT, and Datasketch.
    """
    rng = np.random.default_rng(seed)
    tensors = [(rng.random(shape) < 0.05).astype(np.float32) for _ in range(n_tensors)]

    return _run_speed_benchmark(tensors, num_hashes=num_hashes)


def benchmark_speed_real(
    tensors: list,
    num_hashes: int = 128,
) -> dict:
    """
    Wall-clock hashing speed on real time-window tensors from D3.
    Compares Kron, TT, and Datasketch.
    """
    return _run_speed_benchmark(tensors, num_hashes=num_hashes)


def _run_speed_benchmark(tensors: list, num_hashes: int = 128) -> dict:
    """Shared speed benchmark logic for both synthetic and real tensors."""
    from core.config import TTMinHashConfig
    from core.kron_minhash import KroneckerMinHash
    from core.tt_minhash import TTDecomposedMinHash

    shape = tensors[0].shape
    n_tensors = len(tensors)
    results = {}

    # Kron and TT
    for label, cls in [("Kron", KroneckerMinHash), ("TT", TTDecomposedMinHash)]:
        cfg = TTMinHashConfig(shape=shape, num_hashes=num_hashes)
        hasher = cls(cfg)

        # Warm-up
        _ = hasher.hash_tensor(tensors[0])
        gc.collect()

        t0 = time.perf_counter()
        for t in tensors:
            _ = hasher.hash_tensor(t)
        elapsed = time.perf_counter() - t0

        results[label] = {
            "total_sec": elapsed,
            "per_tensor_ms": elapsed / n_tensors * 1000,
            "tensors_per_sec": n_tensors / elapsed,
        }

    # Datasketch
    try:
        from datasketch import MinHash

        # Warm-up
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


# --------------------------------------------------------------------------
# Peak RAM benchmark
# --------------------------------------------------------------------------

def benchmark_ram(shape: tuple[int, ...] = (50, 50, 50), num_hashes: int = 128) -> dict:
    """
    Compare theoretical parameter-storage memory for Kron, TT, and full matrix.
    All numbers are computed analytically so comparisons are apples-to-apples.
    Peak working memory during hashing (temporary arrays) is excluded, because
    the full matrix cannot actually be allocated at large scales.
    """
    from core.config import TTMinHashConfig

    cfg = TTMinHashConfig(shape=shape, num_hashes=num_hashes)
    r = cfg.tt_rank

    # Kron: num_hashes * sum(shape) float64 values
    kron_params = sum(s * num_hashes for s in shape)
    kron_bytes = kron_params * 8  # float64

    # TT: num_hashes independent decompositions, each with boundary ranks 1
    ndim = len(shape)
    params_per_decomp = sum(
        (1 if k == 0 else r) * shape[k] * (1 if k == ndim - 1 else r) for k in range(ndim)
    )
    tt_params = num_hashes * params_per_decomp
    tt_bytes = tt_params * 4  # float32

    # Full matrix: prod(shape) * num_hashes float32 values
    full_params = int(np.prod(shape)) * num_hashes
    full_bytes = full_params * 4  # float32

    return {
        "kron_peak_bytes": kron_bytes,
        "kron_peak_mb": kron_bytes / 1_000_000,
        "tt_peak_bytes": tt_bytes,
        "tt_peak_mb": tt_bytes / 1_000_000,
        "full_theoretical_bytes": full_bytes,
        "full_theoretical_mb": full_bytes / 1_000_000,
        "kron_params": kron_params,
        "tt_params": tt_params,
        "full_params": full_params,
        "kron_vs_full": full_bytes / max(kron_bytes, 1),
        "tt_vs_full": full_bytes / max(tt_bytes, 1),
        "kron_vs_tt": tt_bytes / max(kron_bytes, 1),
    }


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

    # Memory cost
    full_matrix_mb = flat_dim * num_hashes * 4 / 1_000_000  # float32

    # Check if matrix is feasible to store
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
            "note": f"Requires {full_matrix_mb:.0f} MB - impractical at this scale",
        }

    # If feasible, time it
    rng = np.random.default_rng(42)
    W = rng.standard_normal((flat_dim, num_hashes)).astype(np.float32)
    W /= np.linalg.norm(W, axis=0)  # normalise columns

    # Warm-up
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
