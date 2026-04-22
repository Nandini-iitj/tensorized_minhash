"""
kmer_builder.py - FASTA sequence -> k-mer set -> 3D binary tensor.

Reads a FASTA file, extracts all overlapping k-mers from the sequence,
then maps each k-mer into a 3D binary tensor of shape
(hash_bucket_0, hash_bucket_1, hash_bucket_2) using three independent
polynomial hash functions.

This is the same tensor representation used in the MinHash engine -
so KroneckerMinHash and TTDecomposedMinHash work on it without modification.
"""

import logging
from pathlib import Path

import numpy as np

__all__ = ["read_fasta", "sequence_to_kmers", "kmers_to_tensor"]

logger = logging.getLogger(__name__)

# Fixed polynomial hash bases (prime) for each tensor mode
_BASES = (31, 37, 41)
_MOD = (2**31) - 1 # Mersenne prime


def read_fasta(path: str) -> tuple[str, str]:
    """
    Parse a single-record FASTA file.
    Returns (header, sequence) - sequence is upper-cased, whitespace stripped.
    """
    fasta_path = Path(path)
    if not fasta_path.exists():
        raise FileNotFoundError(f"FASTA file not found: {fasta_path.resolve()}")

    lines = fasta_path.read_text().splitlines()
    header = ""
    seq_parts = []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        if line.startswith(">"):
            header = line[1:].strip()
        else:
            seq_parts.append(line.upper())

    sequence = "".join(seq_parts)
    logger.info(f"Read {fasta_path.name}: {len(sequence):,} bp  header='{header[:60]}'")
    return header, sequence


def sequence_to_kmers(sequence: str, k: int = 12) -> set[str]:
    """
    Extract all overlapping k-mers from a DNA sequence.
    Non-ACGT characters (gaps, ambiguity codes) are skipped.
    """
    valid = set("ACGT")
    kmers: set[str] = set()
    n = len(sequence)
    for i in range(n - k + 1):
        kmer = sequence[i : i + k]
        if all(c in valid for c in kmer):
            kmers.add(kmer)
    logger.info(f"Extracted {len(kmers):,} unique {k}-mers from {n:,} bp sequence")
    return kmers


def _poly_hash(kmer: str, base: int, mod: int) -> int:
    """Polynomial rolling hash of a k-mer string."""
    h = 0
    for c in kmer:
        h = (h * base + ord(c)) % mod
    return h


def kmers_to_tensor(
    kmers: set[str],
    shape: tuple[int, int, int] = (64, 64, 64),
) -> np.ndarray:
    """
    Map a set of k-mers into a 3D binary tensor.

    Each k-mer is hashed three times (with different bases) to produce an
    (i, j, k) index. T[i, j, k] = 1 if any k-mer maps to that bucket.

    Shape is configurable - larger shapes reduce collision rate but increase
    tensor sparsity. (64, 64, 64) gives ~262K cells with ~1-5% density for
    typical mitochondrial genomes (~16K bp, k=12).
    """
    tensor = np.zeros(shape, dtype=np.float32)
    d0, d1, d2 = shape

    for kmer in kmers:
        i = _poly_hash(kmer, _BASES[0], _MOD) % d0
        j = _poly_hash(kmer, _BASES[1], _MOD) % d1
        k = _poly_hash(kmer, _BASES[2], _MOD) % d2
        tensor[i, j, k] = 1.0

    nonzero = int(tensor.sum())
    density = nonzero / tensor.size
    logger.info(f"Tensor shape={shape}, nonzeros={nonzero:,}, density={density:.4f}")
    return tensor