"""
worker.py - Standard (uncompressed) MinHash racing worker.

Implements the classic full-universe MinHash: for every tensor call the hash
family is evaluated across ALL n³ cells - active or not - and the minimum
over active cells is selected.  This is how MinHash operates without a
compressed representation of the hash universe.

Why this is slower than Kron / TT:
  Kron stores hash parameters as k vectors of length n (k·n·d params total),
  so each call is O(k × nnz × d) - it only touches active cells.
  TT does the same via a precomputed prefix table.
  Uncompressed MinHash must evaluate h_j(x) for every x in {0..n³-1},
  scaling as O(k × n³) regardless of sparsity.

Memory per tensor call: K × n³ × 8 bytes (int64 score matrix).
At n=90 this is ~1.5 GB - the worker exits with OOM at that scale.

Progress lines (read by Streamlit via docker logs):
    PROGRESS: 12/250 pairs
    DONE: algo=minhash load_level=7 elapsed=43.2s mae=0.031
"""

import json
import logging
import os
from pathlib import Path
import sys
import time

import numpy as np

sys.path.insert(0, "/app")
from log_setup import setup_logging

setup_logging()
logger = logging.getLogger(__name__)

SHARED_INPUT = Path(os.environ.get("SHARED_INPUT_DIR", "/shared_input"))
SCORES_DIR = Path(os.environ.get("SCORES_DIR", "/scores"))
NUM_HASHES = int(os.environ.get("NUM_HASHES", "256"))
SEED = int(os.environ.get("SEED", "42"))

# Large prime for the universal hash family  h(x) = (ax + b) mod p
_PRIME = (1 << 31) - 1  # Mersenne prime 2^31 − 1


def _build_hash_params(num_hashes: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Return (a, b) coefficient arrays of shape (num_hashes,) for the hash family."""
    rng = np.random.default_rng(seed)
    a = rng.integers(1, _PRIME, size=num_hashes, dtype=np.int64)
    b = rng.integers(0, _PRIME, size=num_hashes, dtype=np.int64)
    return a, b


# Pre-build once at module load so the cost is not charged per tensor
_HASH_A, _HASH_B = _build_hash_params(NUM_HASHES, SEED)


def tensor_to_signature(tensor: np.ndarray) -> np.ndarray:
    """
    Compute a NUM_HASHES-length MinHash signature using full-universe evaluation.

    Evaluates the hash family h_j(x) = (a_j*x + b_j) mod p for every x in
    {0, 1, ..., n^d - 1}, then selects the argmin over active (nonzero) cells.
    This costs O(K × n^d) per call - the uncompressed worst case.
    """
    flat = tensor.ravel()
    n_cells = flat.size
    active_mask = flat > 0

    if not np.any(active_mask):
        return np.zeros(NUM_HASHES, dtype=np.int64)

    # Evaluate hash family over the FULL universe - O(K × n^d)
    x_all = np.arange(n_cells, dtype=np.int64)
    scores = (_HASH_A[:, None] * x_all[None, :] + _HASH_B[:, None]) % _PRIME  # (K, n^d)

    # Mask inactive cells so they cannot win the argmin
    scores[:, ~active_mask] = _PRIME

    return np.argmin(scores, axis=1).astype(np.int64)


def jaccard_from_signatures(sig_a: np.ndarray, sig_b: np.ndarray) -> float:
    return float(np.mean(sig_a == sig_b))


def exact_jaccard(a: np.ndarray, b: np.ndarray) -> float:
    ab = a.astype(bool)
    bb = b.astype(bool)
    inter = int(np.sum(ab & bb))
    union = int(np.sum(ab | bb))
    return inter / union if union > 0 else 0.0


def main():
    manifest_path = SHARED_INPUT / "input_manifest.json"
    if not manifest_path.exists():
        logger.error("Manifest not found at %s - exiting", manifest_path)
        sys.exit(1)

    with open(manifest_path) as f:
        manifest = json.load(f)

    shape = tuple(manifest["shape"])
    n_pairs = manifest["n_pairs"]
    load_level = manifest["load_level"]
    labels = manifest.get("pair_labels", [])

    logger.info(
        "LOAD_LEVEL=%d  shape=%s  n_pairs=%d  num_hashes=%d", load_level, shape, n_pairs, NUM_HASHES
    )

    pairs_a = np.load(SHARED_INPUT / "pairs_a.npy")
    pairs_b = np.load(SHARED_INPUT / "pairs_b.npy")

    SCORES_DIR.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    errors = []
    pair_results = []

    n_cells = int(np.prod(shape))
    score_matrix_kb = (NUM_HASHES * n_cells * 8) // 1024
    logger.info(
        "Dense score matrix per call: %d KB  (%d × %d int64)",
        score_matrix_kb, NUM_HASHES, n_cells,
    )

    for i in range(n_pairs):
        ta = pairs_a[i]
        tb = pairs_b[i]

        try:
            sig_a = tensor_to_signature(ta)
            sig_b = tensor_to_signature(tb)
        except MemoryError:
            logger.error(
                "OOM: dense MinHash score matrix (%d KB) exceeds container memory",
                score_matrix_kb,
            )
            print(
                f"OOM: algo=minhash shape={shape} score_matrix={score_matrix_kb}KB",
                flush=True,
            )
            sys.exit(137)
        est_j = jaccard_from_signatures(sig_a, sig_b)
        exact = exact_jaccard(ta, tb)
        err = abs(est_j - exact)
        errors.append(err)

        lbl = labels[i] if i < len(labels) else {"a": "?", "b": "?"}
        pair_results.append(
            {
                "pair": f"{lbl['a']}<->{lbl['b']}",
                "exact": round(exact, 4),
                "minhash": round(est_j, 4),
                "error": round(err, 4),
            }
        )

        if (i + 1) % max(1, n_pairs // 20) == 0 or i == n_pairs - 1:
            elapsed = time.perf_counter() - start
            print(f"PROGRESS: {i + 1}/{n_pairs} pairs  elapsed={elapsed:.1f}s", flush=True)

    elapsed = time.perf_counter() - start
    mae = float(np.mean(errors))

    print(
        f"DONE: algo=minhash load_level={load_level} elapsed={elapsed:.2f}s mae={mae:.4f}",
        flush=True,
    )

    timestamp = int(time.time())
    out_path = SCORES_DIR / f"minhash_{timestamp}_level{load_level}_metrics.json"

    # Memory: dense score matrix = K × n^d × 8 bytes per tensor call
    n_cells = int(np.prod(shape))
    mem_kb = (NUM_HASHES * n_cells * 8) // 1024 or 1

    metrics = {
        "algo": "minhash",
        "load_level": load_level,
        "shape": list(shape),
        "n_pairs": n_pairs,
        "num_hashes": NUM_HASHES,
        "elapsed_s": round(elapsed, 3),
        "mae": round(mae, 4),
        "max_error": round(max(errors), 4),
        "memory_kb": mem_kb,
        "timestamp": timestamp,
        "pairs": pair_results,
    }
    with open(out_path, "w") as f:
        json.dump(metrics, f, indent=2)

    logger.info("Metrics written: %s", out_path)


if __name__ == "__main__":
    main()