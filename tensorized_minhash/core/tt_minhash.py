# TTDecomposedMinHash - Tensor Train decomposed MinHash.

"""
Stores num_hashes independent TT decompositions, each with standard boundary
ranks (r_0 = r_d = 1) and inner bond dimension tt_rank.
Core shapes: (1, n_0, r), (r, n_1, r), ..., (r, n_{d-1}, 1).

For each hash function j and nonzero cell (i_0, ..., i_{d-1}), the rank is:
    rank_j(i_0, ..., i_{d-1}) = g_0[i_0, i_1, ..., i_{d-1}] · g_1[i_1, ..., i_{d-1}],
where g_0 and g_1 are functions that map indices to ranks. The default values
for g_0 and g_1 are:
    g_0[i_0, i_1, ..., i_{d-1}] = 1,
    g_1[i_1, ..., i_{d-1}] = 1.

This is a scalar. With random Gaussian cores, these scalars approximate i.i.d.
random variables, so argmin over nonzero cells preserves Jaccard:
    Pr[argmin_i(A) rank_j == argmin_i(B) rank_j] = 3/4

Memory: num_hashes * d * n * r^2 - linear in all dimensions, compared to
O(n^d * k) for the full matrix. More expressive than Kronecker (which uses
additive Exp(1) ranks) but still dramatically compressed.
"""

import logging

import numpy as np

from .config import TTMinHashConfig

__all__ = ["TTDecomposedMinHash"]

logger = logging.getLogger(__name__)


class TTDecomposedMinHash:
    """MinHash using Tensor Train (TT) decomposed hash functions."""

    def __init__(self, config: TTMinHashConfig):
        self.config = config
        rng = np.random.default_rng(config.seed + 1)
        r = config.tt_rank
        ndim = config.ndim

        # k independent TT decompositions; each is a list of d cores.
        # Boundary ranks are 1 (standard TT), inner ranks are tt_rank.
        # Core shapes: (1, n_0, r), (r, n_1, r), ..., (r, n_{d-1}, 1)
        self.all_cores: list[list[np.ndarray]] = []

        for _ in range(config.num_hashes):
            cores: list[np.ndarray] = []
            for mode, n_k in enumerate(config.shape):
                r_left = 1 if mode == 0 else r
                r_right = 1 if mode == ndim - 1 else r
                core = rng.standard_normal((r_left, n_k, r_right)).astype(np.float32)
                # Left-orthogonalize for numerical stability
                core_mat = core.reshape(r_left * n_k, r_right)
                if r_left * n_k >= r_right:
                    core_mat, _ = np.linalg.qr(core_mat)
                    core = core_mat[:, :r_right].reshape(r_left, n_k, r_right)
                cores.append(core)
            self.all_cores.append(cores)

        self.param_count = sum(sum(c.size for c in cores) for cores in self.all_cores)
        self.full_param_count = int(np.prod(config.shape)) * config.num_hashes

    def hash_tensor(self, tensor: np.ndarray) -> np.ndarray:
        """
        Compute MinHash signature via per-cell TT contraction.

        For each hash function j, contracts TT-cores along the mode indices
        of every nonzero cell to produce a scalar rank. The MinHash value is
        the flat index of the cell with the minimum rank.

        Vectorized over cells within each hash function j.
        Returns: signature of shape (num_hashes,) - integer hash values.
        """
        assert tensor.shape == self.config.shape, f"Expected {self.config.shape}, got {tensor.shape}"

        nonzero_idx = np.argwhere(tensor > 0)  # (nnz, ndim)
        if len(nonzero_idx) == 0:
            return np.zeros(self.config.num_hashes, dtype=np.int32)

        nnz = len(nonzero_idx)
        signature = np.empty(self.config.num_hashes, dtype=np.int32)

        for j, cores in enumerate(self.all_cores):
            # state: (nnz, r_current) - left boundary rank is 1
            state = np.ones((nnz, 1), dtype=np.float32)

            for mode, core in enumerate(cores):
                # core: (r_left, n_k, r_right)
                mode_idx = nonzero_idx[:, mode]
                # Per-cell core slices: core[:, mode_idx, :] -> (r_left, nnz, r_right)
                # Transpose to (nnz, r_left, r_right) for batched matmul
                slices = core[:, mode_idx, :].transpose(1, 0, 2)
                # state (nnz, r_left) * slices (nnz, r_left, r_right) -> (nnz, r_right)
                state = np.einsum("ni,nij->nj", state, slices)

            # state: (nnz, 1) - squeeze to scalar rank per cell
            ranks = state[:, 0]  # (nnz,)
            min_cell = int(np.argmin(ranks))
            multi_idx = nonzero_idx[min_cell]
            signature[j] = int(np.ravel_multi_index(multi_idx, self.config.shape))

        return signature

    def jaccard_from_signatures(self, sig_a: np.ndarray, sig_b: np.ndarray) -> float:
        """Estimate Jaccard similarity. J(A,B) = #(sig_a == sig_b) / k"""
        return float(np.mean(sig_a == sig_b))

    def memory_stats(self) -> dict[str, int]:
        """Return parameter storage statistics."""
        tt_bytes = self.param_count * 4  # float32
        full_bytes = self.full_param_count * 4
        return {
            "tt_params": self.param_count,
            "full_params_theoretical": self.full_param_count,
            "compression_ratio": self.full_param_count // max(self.param_count, 1),
            "tt_bytes": tt_bytes,
            "full_bytes_theoretical": full_bytes,
        }