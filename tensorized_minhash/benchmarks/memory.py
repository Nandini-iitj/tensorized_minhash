"""
benchmarks.memory — Parameter-count and RAM storage benchmarks.

Compares Kronecker, Tensor Train, and full-matrix parameter counts across 
tensor shapes, all computed analytically so numbers are apples-to-apples.
"""

import logging
import numpy as np

__all__ = ["benchmark_memory", "benchmark_ram"]

logger = logging.getLogger(__name__)

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
    kron_params = sum(num_hashes * s for s in shape)
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