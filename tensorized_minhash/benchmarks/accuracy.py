"""
benchmarks.accuracy - Three-scenario Jaccard estimation accuracy benchmark.

Runs n_pairs tensor pairs under three similarity regimes:
  Scenario 1 - High similarity   (J ≈ 0.85-0.98): near-identical tensors
  Scenario 2 - Medium similarity (J ≈ 0.35-0.65): shared base + noise
  Scenario 3 - Low similarity    (J ≈ 0.01-0.12): independently random

The combined Spearman ρ across all three scenarios is the primary proposal
evaluation metric (target: ρ > 0.85), because it spans the full [0, 1]
Jaccard range giving rank correlation real signal.
"""

import logging

import numpy as np
from scipy.stats import spearmanr

from .helpers import datasketch_jaccard, ground_truth_jaccard

__all__ = ["benchmark_accuracy", "benchmark_accuracy_from_tensors"]

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------
# Pair generators - one function per similarity regime
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

    n_flip = max(1, int(0.05 * n_base))
    flip_a = rng.choice(all_cells, n_flip, replace=False)
    flip_b = rng.choice(all_cells, n_flip, replace=False)

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
    a.ravel()[list(set_a)] = 1.0
    b = np.zeros(shape, dtype=np.float32)
    b.ravel()[list(set_b)] = 1.0
    return a, b


# --------------------------------------------------------------------------
# Metric computation
# --------------------------------------------------------------------------

def _compute_accuracy_results(
    exact_j: np.ndarray,
    kron_j: np.ndarray,
    ds_j: np.ndarray,
    tt_j: np.ndarray,
    n_pairs: int,
    shape: tuple[int, ...],
    num_hashes: int,
) -> dict:
    """Compute MAE, RMSE, Pearson r, and Spearman ρ for all three methods."""

    def metrics(pred: np.ndarray, true: np.ndarray) -> dict:
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

    return {
        "tensorized_kron": metrics(kron_j, exact_j),
        "tt_decomposition": metrics(tt_j, exact_j),
        "datasketch_baseline": metrics(ds_j, exact_j),
        "avg_exact": float(np.mean(exact_j)),
        "n_pairs": n_pairs,
        "shape": shape,
        "num_hashes": num_hashes,
        "jaccard_range": (float(exact_j.min()), float(exact_j.max())),
    }


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


# --------------------------------------------------------------------------
# Public benchmark functions
# --------------------------------------------------------------------------

def benchmark_accuracy(
    n_pairs: int = 100,
    shape: tuple[int, ...] = (30, 30, 30),
    num_hashes: int = 128,
    seed: int = 0,
) -> dict:
    """
    Three-scenario accuracy benchmark covering the full Jaccard range.

    Runs n_pairs pairs under each of three similarity regimes and reports
    per-scenario and combined metrics. The combined Spearman ρ is the primary
    proposal evaluation metric (target: ρ > 0.85).
    """
    from core import KroneckerMinHash, TTDecomposedMinHash, TTMinHashConfig

    rng = np.random.default_rng(seed)
    cfg = TTMinHashConfig(shape=shape, num_hashes=num_hashes, seed=seed)
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
        # Top-level keys for backward compatibility
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
    shape: tuple[int, ...],
    num_hashes: int = 128,
    n_pairs: int = 200,
    seed: int = 0,
) -> dict:
    """
    Measure Jaccard approximation accuracy on real time-window tensors
    produced by Deliverable 3. Randomly samples n_pairs from available windows.
    """
    from core import KroneckerMinHash, TTDecomposedMinHash, TTMinHashConfig

    cfg = TTMinHashConfig(shape=shape, num_hashes=num_hashes)
    hasher = KroneckerMinHash(cfg)
    tt_hasher = TTDecomposedMinHash(cfg)
    rng = np.random.default_rng(seed)

    n_windows = len(tensors)
    all_pairs = [(i, j) for i in range(n_windows) for j in range(i + 1, n_windows)]

    if len(all_pairs) > n_pairs:
        sampled = rng.choice(len(all_pairs), size=n_pairs, replace=False)
        pairs = [all_pairs[k] for k in sampled]
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