"""
generate_input.py - Converts FASTA genomes into tensor pairs for the racing track.

Reads FASTA files from demo/genome/data/, builds 3D k-mer tensors, then
generates n_pairs (tensor_A, tensor_B) arrays at the LOAD_LEVEL shape and
saves them to the shared_input volume so all workers load the same data.

Usage (run before starting workers):
    python generate_input.py --load-level 7
    # or via Docker Compose pre-start hook
"""

import argparse
import json
import logging
from pathlib import Path
import sys

import numpy as np

# Add repo root so tensorized_minhash is importable as a package.
# Add demo/genome/ explicitly for kmer_builder.
_DEMO_ROOT = Path(__file__).resolve().parents[2] # demo/
_GENOME = _DEMO_ROOT / "genome"
_REPO_ROOT = _DEMO_ROOT.parent # repo root
sys.path.insert(0, str(_REPO_ROOT)) # -> from tensorized_minhash.X import
sys.path.insert(0, str(_GENOME))    # -> from kmer_builder import

from kmer_builder import kmers_to_tensor, read_fasta, sequence_to_kmers # noqa: E402

from log_setup import setup_logging # noqa: E402

setup_logging()
logger = logging.getLogger(__name__)

# -----------------------------------------------------------------------------
# LOAD_LEVEL -> (tensor shape, n_pairs)
# -----------------------------------------------------------------------------
LEVEL_MAP = {
    1: ((15, 15, 15), 30),
    2: ((20, 20, 20), 50),
    3: ((28, 28, 28), 70),
    4: ((35, 35, 35), 100),
    5: ((50, 50, 50), 150),
    6: ((60, 60, 60), 200),
    7: ((70, 70, 70), 250),
    8: ((80, 80, 80), 320),
    9: ((90, 90, 90), 400),
    10: ((100, 100, 100), 500),
}

K_MER_SIZE = 12
GEN_SEED = 42

DATA_DIR = _GENOME / "data"
SPECIES = {
    "human": DATA_DIR / "human.fasta",
    "chimp": DATA_DIR / "chimp.fasta",
    "mouse": DATA_DIR / "mouse.fasta",
    "yeast": DATA_DIR / "yeast.fasta",
}


def build_genome_tensors(shape: tuple) -> dict:
    """Load all FASTA files and return tensors at the given shape."""
    tensors = {}
    fasta_missing = []
    for name, path in SPECIES.items():
        if not path.exists():
            logger.warning("FASTA not found: %s - will use synthetic tensor", path)
            fasta_missing.append(name)
            continue
        header, seq = read_fasta(str(path))
        kmers = sequence_to_kmers(seq, k=K_MER_SIZE)
        tensors[name] = kmers_to_tensor(kmers, shape=shape)
        logger.info(
            "Built tensor for %s: %d k-mers -> shape %s density=%.3f",
            name,
            len(kmers),
            shape, 
            tensors[name].mean(),
        )

    # Fill missing species with synthetic random tensors at ~0.15 density
    rng = np.random.default_rng(GEN_SEED)
    for name in fasta_missing:
        tensors[name] = (rng.random(shape) < 0.15).astype(np.float32)
        logger.warning("Synthetic tensor created for %s", name)

    return tensors


def generate_pairs(tensors: dict, n_pairs: int, seed: int) -> list:
    """
    Create n_pairs (a, b) pairs by sampling with replacement from genome pairs.
    Extra pairs beyond the 6 real pairs are created by adding small random noise
    to preserve realistic density while increasing workload.
    """
    rng = np.random.default_rng(seed)
    names = list(tensors.keys())
    base_pairs = [(a, b) for i, a in enumerate(names) for b in names[i + 1 :]]

    pairs = []
    for i in range(n_pairs):
        a_name, b_name = base_pairs[i % len(base_pairs)]
        ta = tensors[a_name].copy().astype(np.float32)
        tb = tensors[b_name].copy().astype(np.float32)    
        # Jitter for repeated pairs to create distinct workload
        if i >= len(base_pairs):
            noise_a = (rng.random(ta.shape) < 0.03).astype(np.float32)
            noise_b = (rng.random(tb.shape) < 0.03).astype(np.float32)
            ta = np.clip(ta + noise_a - (rng.random(ta.shape) < 0.01).astype(np.float32), 0, 1)
            tb = np.clip(tb + noise_b - (rng.random(tb.shape) < 0.01).astype(np.float32), 0, 1)
        pairs.append((ta, tb, a_name, b_name))
    
    return pairs


def save_input(out_dir: Path, pairs: list, shape: tuple, level: int, seed: int):
    """Serialize tensor pairs and manifest to out_dir."""
    out_dir.mkdir(parents=True, exist_ok=True)

    # Clear old data
    for f in out_dir.glob("pairs_*.npy"):
        f.unlink()

    a_stack = np.stack([p[0] for p in pairs]) # (n_pairs, *shape)
    b_stack = np.stack([p[1] for p in pairs])
    labels = [{"a": p[2], "b": p[3]} for p in pairs]

    np.save(out_dir / "pairs_a.npy", a_stack)
    np.save(out_dir / "pairs_b.npy", b_stack)

    manifest = {
        "load_level": level,
        "shape": list(shape),
        "n_pairs": len(pairs),
        "seed": seed,
        "pair_labels": labels,
    }
    with open(out_dir / "input_manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    size_mb = (a_stack.nbytes + b_stack.nbytes) / 1024 / 1024
    logger.info("Saved %d pairs -> %s (%.1f MB total)", len(pairs), out_dir, size_mb)


def main():
    parser = argparse.ArgumentParser(description="Generate racing-track tensor input")
    parser.add_argument(
        "--load-level",
        type=int,
        default=7,
        choices=range(1, 11),
        metavar="1-10",
        help="Workload intensity (1=tiny, 10=max)",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default=str(Path(__file__).parent),
        help="Output directory (default: shared_input/)",
    )
    parser.add_argument("--seed", type=int, default=GEN_SEED)
    args = parser.parse_args()

    shape, n_pairs = LEVEL_MAP[args.load_level]
    logger.info("LOAD_LEVEL=%d -> shape=%s, n_pairs=%d", args.load_level, shape, n_pairs)

    tensors = build_genome_tensors(shape)
    pairs = generate_pairs(tensors, n_pairs, args.seed)
    save_input(Path(args.out_dir), pairs, shape, args.load_level, args.seed)

    print(f"\n Input ready: {n_pairs} pairs, shape {shape}, level {args.load_level}\n")

if __name__ == "__main__":
    main()