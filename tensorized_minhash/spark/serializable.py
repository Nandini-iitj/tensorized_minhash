"""
spark.serializable — Pickle-safe KroneckerMinHash wrapper for Spark workers.

We store only factor matrices (numpy arrays) and config primitives —
no Python closures or unpicklable objects — so the class distributes
safely across Spark executors via Kryo serialisation.
"""

import logging
import numpy as np

__all__ = ["SerializableKronHasher"]

logger = logging.getLogger(__name__)


class SerializableKronHasher:
    """
    Lightweight wrapper around KroneckerMinHash that is safely picklable
    for distribution across Spark workers.
    """

    def __init__(self, factors: list[np.ndarray], shape: tuple[int, ...], num_hashes: int):
        self.factors = factors
        self.shape = shape
        self.num_hashes = num_hashes

    @classmethod
    def from_kron_hasher(cls, hasher) -> "SerializableKronHasher":
        """Construct from a KroneckerMinHash instance by copying its factor arrays."""
        return cls(
            factors=[f.copy() for f in hasher.factors],
            shape=hasher.cfg.shape,
            num_hashes=hasher.cfg.num_hashes,
        )

    def hash_tensor(self, tensor: np.ndarray) -> np.ndarray:
        """Compute MinHash signature using Kronecker additive-exponential argmin."""
        nonzero_idx = np.argwhere(tensor > 0)
        if len(nonzero_idx) == 0:
            return np.zeros(self.num_hashes, dtype=np.int32)

        nnz = len(nonzero_idx)
        ranks = np.zeros((self.num_hashes, nnz), dtype=np.float64)
        for mode in range(len(self.shape)):
            ranks += self.factors[mode][:, nonzero_idx[:, mode]]  # (num_hashes, nnz)

        min_cells = np.argmin(ranks, axis=1)  # (num_hashes,)
        multi_indices = nonzero_idx[min_cells]  # (num_hashes, ndim)
        return np.ravel_multi_index(multi_indices.T, self.shape).astype(np.int32)