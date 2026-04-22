"""
similarity_report.py - Human vs Evolution: genomic similarity via TensorizedMinHash.

Anchors on the human mitochondrial genome and compares against species at
increasing evolutionary distances, producing a visual similarity gradient.

Missing FASTA files are downloaded automatically from NCBI on first run.

Methods:
    - Exact (ground truth)
    - Kronecker MinHash
    - Tensor Train MinHash
    - Datasketch (standard MinHash baseline)

Usage:
    cd demo/genome
    python similarity_report.py

Expected similarity order (highest -> lowest):
    Human ↔ Chimp      - great ape  - ~6 Ma divergence
    Human ↔ Gorilla    - great ape  - ~10 Ma divergence
    Human ↔ Orangutan  - great ape  - ~14 Ma divergence
    Human ↔ Mouse      - mammal     - ~80 Ma divergence
    Human ↔ Chicken    - bird       - ~310 Ma divergence
    Human ↔ Zebrafish  - fish       - ~450 Ma divergence
    Chimp ↔ Mouse      : exact ~0.28  (distant mammals)
"""

import logging
from pathlib import Path
import sys
import urllib.request

import numpy as np

# Add repo root so tensorized_minhash is importable as a package.
# Add this script's own directory explicitly for kmer_builder.
_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT))  # -> from tensorized_minhash.X import
sys.path.insert(0, str(Path(__file__).parent))  # -> from kmer_builder import

from kmer_builder import kmers_to_tensor, read_fasta, sequence_to_kmers  # noqa: E402

from log_setup import setup_logging  # noqa: E402
from tensorized_minhash.core import (  # noqa: E402
    KroneckerMinHash,
    TTDecomposedMinHash,
    TTMinHashConfig,
)

setup_logging()
logger = logging.getLogger(__name__)

# Suppress verbose per-call INFO lines that clutter the report output
logging.getLogger("kmer_builder").setLevel(logging.WARNING)
logging.getLogger("tensorized_minhash.core.config").setLevel(logging.WARNING)
logging.getLogger("core.config").setLevel(logging.WARNING)

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------

DATA_DIR = Path(__file__).parent / "data"
K_MER_SIZE = 12
TENSOR_SHAPE = (64, 64, 64)
NUM_HASHES = 256
SEED = 42

# Ordered from evolutionarily closest -> most distant from Human.
# Each value: (filename, NCBI_accession, one-line description)
SPECIES_META: dict[str, tuple[str, str, str]] = {
    "Human":     ("human.fasta",     "NC_012920.1", "Reference     - Homo sapiens"),
    "Chimp":     ("chimp.fasta",     "NC_001643.1", "Great ape     - diverged ~6 Ma"),
    "Gorilla":   ("gorilla.fasta",   "NC_011120.1", "Great ape     - diverged ~10 Ma"),
    "Orangutan": ("orangutan.fasta", "NC_002083.1", "Great ape     - diverged ~14 Ma"),
    "Mouse":     ("mouse.fasta",     "NC_005089.1", "Mammal        - diverged ~80 Ma"),
    "Chicken":   ("chicken.fasta",   "NC_001323.1", "Bird          - diverged ~310 Ma"),
    "Zebrafish": ("zebrafish.fasta", "NC_002333.2", "Fish          - diverged ~450 Ma"),
}

ANCHOR = "Human"
SPECIES = {name: DATA_DIR / meta[0] for name, meta in SPECIES_META.items()}

PAIR_NOTES = {
    ("Human", "Chimp"):     "Great ape - expect VERY HIGH similarity",
    ("Human", "Gorilla"):   "Great ape - expect VERY HIGH similarity",
    ("Human", "Orangutan"): "Great ape - expect HIGH similarity",
    ("Human", "Mouse"):     "Mammal    - expect MEDIUM similarity",
    ("Human", "Chicken"):   "Bird      - expect LOW similarity",
    ("Human", "Zebrafish"): "Fish      - expect LOW similarity",
}


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------


def exact_jaccard(a: np.ndarray, b: np.ndarray) -> float:
    """Binary tensor Jaccard: |A∩B| / |A∪B|."""
    ab = a.astype(bool)
    bb = b.astype(bool)
    inter = np.sum(ab & bb)
    union = np.sum(ab | bb)
    return float(inter / union) if union > 0 else 0.0


def _fmt(j: float) -> str:
    bar_len = int(j * 30)
    bar = "█" * bar_len + "░" * (30 - bar_len)
    label = "HIGH" if j > 0.5 else ("MED " if j > 0.2 else "LOW ")
    return f"{j:.4f}  [{bar}]  {label}"


def _col(w: int, s: str) -> str:
    return str(s).ljust(w)


def _download_fasta(name: str, dest: Path, accession: str) -> None:
    """Download a mitochondrial FASTA from NCBI RefSeq."""
    url = (
        "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
        f"?db=nuccore&id={accession}&rettype=fasta&retmode=text"
    )
    print(f" Fetching {name:10s} ({accession})...", end=" ", flush=True)
    urllib.request.urlretrieve(url, dest)
    print("✓")


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------


def main():

    # Load + build tensors
    print("\n" + "=" * 72)
    print(" Human vs Animals - MinHash on Mitochondrial Genomes")
    print("=" * 72)
    print(f"\n  k={K_MER_SIZE} k-mers  •  {TENSOR_SHAPE} tensor  •  {NUM_HASHES} hash planes\n")

    to_download = [
        (name, DATA_DIR / meta[0], meta[1])
        for name, meta in SPECIES_META.items()
        if not (DATA_DIR / meta[0]).exists()
    ]
    if to_download:
        print(f"\n Downloading {len(to_download)} missing FASTA file(s) from NCBI...")
        for _name, _dest, _acc in to_download:
            _download_fasta(_name, _dest, _acc)
        print()

    tensors: dict[str, np.ndarray] = {}
    kmer_counts: dict[str, int] = {}
    for name, fasta_path in SPECIES.items():
        header, seq = read_fasta(str(fasta_path))
        kmers = sequence_to_kmers(seq, k=K_MER_SIZE)
        kmer_counts[name] = len(kmers)
        tensors[name] = kmers_to_tensor(kmers, shape=TENSOR_SHAPE)
        print(
            f"  {name:10s}: ({len(seq):>7,}) bp  ->  {len(kmers):>6,} unique {K_MER_SIZE}-mers  "
            f"-> tensor density {tensors[name].mean():.3f}"
        )

    # Set up hashers
    
    cfg = TTMinHashConfig(shape=TENSOR_SHAPE, num_hashes=NUM_HASHES, seed=SEED)
    kron_h = KroneckerMinHash(cfg)
    tt_h = TTDecomposedMinHash(cfg)

    mem_kron = kron_h.memory_stats()
    mem_tt = tt_h.memory_stats()
    print("\n Hash function memory:")
    print(f"  Kronecker : {mem_kron['kron_bytes'] / 1024:.1f} KB")
    print(f"  TT        : {mem_tt['tt_bytes'] / 1024:.1f} KB")

    print("\n Computing signatures...", end=" ", flush=True)
    sigs_kron: dict[str, np.ndarray] = {}
    sigs_tt: dict[str, np.ndarray] = {}
    for name, tensor in tensors.items():
        sigs_kron[name] = kron_h.hash_tensor(tensor)
        sigs_tt[name] = tt_h.hash_tensor(tensor)
    print("✓")

    print("\n" + "=" * 72)
    print(" Jaccard Similarity (Kronecker & TT vs exact ground truth)")
    print("=" * 72)
    header_row = f" {'Pair':22s} {'Exact':>8} {'Kron':>8} {'TT':>8} {'Datasketch':>10}"
    print(header_row)
    print(" " + "-" * 68)

    try:
        from datasketch import MinHash as DsMinHash

        USE_DS = True
    except ImportError:
        USE_DS = False
        logger.warning("datasketch not installed - skipping baseline column")

    pairs = [(ANCHOR, name) for name in SPECIES_META if name != ANCHOR]
    results = []
    for a_name, b_name in pairs:
        ta = tensors[a_name]
        tb = tensors[b_name]

        exact = exact_jaccard(ta, tb)
        kron = kron_h.jaccard_from_signatures(sigs_kron[a_name], sigs_kron[b_name])
        tt = tt_h.jaccard_from_signatures(sigs_tt[a_name], sigs_tt[b_name])

        if USE_DS:
            m1 = DsMinHash(num_perm=NUM_HASHES)
            m2 = DsMinHash(num_perm=NUM_HASHES)
            for idx in np.where(ta.ravel() > 0)[0]:
                m1.update(int(idx).to_bytes(8, "big"))
            for idx in np.where(tb.ravel() > 0)[0]:
                m2.update(int(idx).to_bytes(8, "big"))
            ds = m1.jaccard(m2)
        else:
            ds = exact  # fallback

        pair_label = f"({a_name}) ↔ ({b_name})"
        print(f" {pair_label:22s} {exact:8.4f} {kron:8.4f} {tt:8.4f} {ds:10.4f}")
        results.append(
            {
                "pair": pair_label,
                "exact": exact,
                "kron": kron,
                "tt": tt,
                "datasketch": ds,
            }
        )

    #
    # 4. Accuracy summary
    #
    kron_errors = [abs(r["kron"] - r["exact"]) for r in results]
    tt_errors = [abs(r["tt"] - r["exact"]) for r in results]
    ds_errors = [abs(r["datasketch"] - r["exact"]) for r in results]

    print("\n" + "=" * 72)
    print("  Estimation accuracy (vs exact Jaccard)")
    print("=" * 72)
    print(f" {'Method':20s}  {'MAE':>8}  {'Max error':>10}")
    print(" " + "-" * 44)
    print(f" {'Kronecker MinHash':20s}  {np.mean(kron_errors):8.4f}  {max(kron_errors):10.4f}")
    print(f" {'Tensor Train (TT)':20s}  {np.mean(tt_errors):8.4f}  {max(tt_errors):10.4f}")
    if USE_DS:
        print(f" {'Datasketch':20s}  {np.mean(ds_errors):8.4f}  {max(ds_errors):10.4f}")


if __name__ == "__main__":
    main()