"""
worker.py - Kronecker MinHash racing worker.

Reads tensor pairs from /shared_input/, computes Kron MinHash signatures,
estimates Jaccard similarity, and writes results to /scores/.

Progress lines (read by Streamlit via docker logs):
    PROGRESS: 12/250 pairs
    DONE: algo=kron load_level=7 elapsed=3.41s mae=0.021
"""

import json
import logging
import os
from pathlib import Path
import sys
import time

import numpy as np

# Engine is installed at /app/tensorized_minhash inside the container
sys.path.insert(0, "/app")

from log_setup import setup_logging
from tensorized_minhash.core import KroneckerMinHash, TTMinHashConfig

setup_logging()
logger = logging.getLogger(__name__)

SHARED_INPUT = Path(os.environ.get("SHARED_INPUT_DIR", "/shared_input"))
SCORES_DIR = Path(os.environ.get("SCORES_DIR", "/scores"))
NUM_HASHES = int(os.environ.get("NUM_HASHES", "256"))
SEED = int(os.environ.get("SEED", "42"))

def exact_jaccard(a: np.ndarray, b: np.ndarray) -> float:
    ab = a.astype(bool)
    bb = b.astype(bool)
    inter = int(np.sum(ab & bb))
    union = int(np.sum(ab | bb))
    return inter / union if union > 0 else 0.0

def main():
    # Load manifest
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

    logger.info("LOAD_LEVEL=%d shape=%s n_pairs=%d", load_level, shape, n_pairs)

    # Load tensor pairs
    pairs_a = np.load(SHARED_INPUT / "pairs_a.npy") # (n_pairs, *shape)
    pairs_b = np.load(SHARED_INPUT / "pairs_b.npy")

    # Build hasher
    cfg = TTMinHashConfig(shape=shape, num_hashes=NUM_HASHES, seed=SEED)
    hasher = KroneckerMinHash(cfg)
    mem = hasher.memory_stats()
    logger.info(
        "Hasher built: %d KB compression=%dx", mem["kron_bytes"] // 1024, mem["compression_ratio"]
    )

    # Process pairs
    SCORES_DIR.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    errors = []
    pair_results = []

    for i in range(n_pairs):
        ta = pairs_a[i]
        tb = pairs_b[i]

        sig_a = hasher.hash_tensor(ta)
        sig_b = hasher.hash_tensor(tb)
        kron_j = hasher.jaccard_from_signatures(sig_a, sig_b)

        exact = exact_jaccard(ta, tb)
        err = abs(kron_j - exact)
        errors.append(err)

        lbl = labels[i] if i < len(labels) else {"a": "?", "b": "?"}
        pair_results.append(
            {
                "pair": f"{lbl['a']} <-> {lbl['b']}",
                "exact": round(exact, 4),
                "kron": round(kron_j, 4),
                "error": round(err, 4),
            }
        )

        # Progress line - Streamlit reads this via container.logs()
        if (i + 1) % max(1, n_pairs // 20) == 0 or i == n_pairs - 1:
            elapsed = time.perf_counter() - start
            print(f"PROGRESS: {i + 1}/{n_pairs} pairs elapsed={elapsed:.1f}s", flush=True)

    elapsed = time.perf_counter() - start
    mae = float(np.mean(errors))

    print(
        f"DONE: algo=kron load_level={load_level} elapsed={elapsed:.2f}s mae={mae:.4f}", flush=True
    )

    # Write metrics file
    timestamp = int(time.time())
    out_path = SCORES_DIR / f"kron_{timestamp}_level{load_level}_metrics.json"
    metrics = {
        "algo": "kron",
        "load_level": load_level,
        "shape": list(shape),
        "n_pairs": n_pairs,
        "num_hashes": NUM_HASHES,
        "elapsed_s": round(elapsed, 3),
        "mae": round(mae, 4),
        "max_error": round(max(errors), 4),
        "memory_kb": mem["kron_bytes"] // 1024,
        "timestamp": timestamp,
        "pairs": pair_results,
    }

    with open(out_path, "w") as f:
        json.dump(metrics, f, indent=2)

    logger.info("Metrics written: %s", out_path)

if __name__ == "__main__":
    main()