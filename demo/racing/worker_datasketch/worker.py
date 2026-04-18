"""
worker.py - Datasketch (standard MinHash) racing worker.

Hashes the flat index positions of nonzero cells in each tensor.
This gives a fair comparison against the tensorized approaches on
identical input.

Progress lines (read by Streamlit via Docker logs):
    PROGRESS: 12/250 pairs
    DONE: algo=datasketch load_level=7 elapsed=43.2s mae=0.031
"""

import json
import logging
import os
from pathlib import Path
import sys
import time

from datasketch import MinHash
import numpy as np

sys.path.insert(0, "/app")
from log_setup import setup_logging

setup_logging()
logger = logging.getLogger(__name__)

SHARED_INPUT = Path(os.environ.get("SHARED_INPUT_DIR", "/shared_input"))
SCORES_DIR = Path(os.environ.get("SCORES_DIR", "/scores"))
NUM_HASHES = int(os.environ.get("NUM_HASHES", "256"))

def exact_jaccard(a: np.ndarray, b: np.ndarray) -> float:
    ab = a.astype(bool)
    bb = b.astype(bool)
    inter = int(np.sum(ab & bb))
    union = int(np.sum(ab | bb))
    return float(inter / union) if union > 0 else 0.0

def tensor_to_minhash(tensor: np.ndarray) -> MinHash:
    """Represent each nonzero cell as its flat index (bytes)."""
    m = MinHash(num_perm=NUM_HASHES)
    for idx in np.where(tensor.ravel() > 0)[0]:
        m.update(int(idx).to_bytes(8, "big"))
    return m

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
        "LOAD_LEVEL=%d  shape=%s  n_pairs=%d  num_perm=%d", 
        load_level, shape, n_pairs, NUM_HASHES
    )

    pairs_a = np.load(SHARED_INPUT / "pairs_a.npy")
    pairs_b = np.load(SHARED_INPUT / "pairs_b.npy")

    SCORES_DIR.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    errors = []
    pair_results = []

    for i in range(n_pairs):
        ta = pairs_a[i]
        tb = pairs_b[i]

        m1 = tensor_to_minhash(ta)
        m2 = tensor_to_minhash(tb)
        ds_j = m1.jaccard(m2)

        exact = exact_jaccard(ta, tb)
        err = abs(ds_j - exact)
        errors.append(err)

        lbl = labels[i] if i < len(labels) else {"a": "?", "b": "?"}
        pair_results.append(
            {
                "pair": f"{lbl['a']} <-> {lbl['b']}",
                "exact": round(exact, 4),
                "datasketch": round(ds_j, 4),
                "error": round(err, 4),
            }
        )

        if (i + 1) % max(1, n_pairs // 20) == 0 or i == n_pairs - 1:
            elapsed = time.perf_counter() - start
            print(f"PROGRESS: {i + 1}/{n_pairs} pairs  elapsed={elapsed:.1f}s", flush=True)

    elapsed = time.perf_counter() - start
    mae = float(np.mean(errors))

    print(
        f"DONE: algo=datasketch load_level={load_level} elapsed={elapsed:.2f}s mae={mae:.4f}",
        flush=True,
    )

    timestamp = int(time.time())
    out_path = SCORES_DIR / f"datasketch_{timestamp}_level{load_level}_metrics.json"

    # datasketch memory: num_perm * 8 bytes per signature
    mem_kb = (NUM_HASHES * 8) // 1024 or 1

    metrics = {
        "algo": "datasketch",
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