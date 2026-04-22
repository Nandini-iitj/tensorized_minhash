"""
TTDecomposedMinHash - Tensor Train decomposed MinHash.

Stores num_hashes independent TT decompositions, each with standard
boundary ranks (r_0 = r_d = 1) and inner bond dimension tt_rank.
Core shapes: (1, n_0, r), (r, n_1, r), ..., (r, n_{d-1}, 1).

For each hash function j and nonzero cell (i_0,...,i_{d-1}), the rank is:
    rank_j(i_0,...,i_{d-1}) = G_0[0, i_0, :] @ G_1[:, i_1, :] @ ... @ G_{d-1}[:, i_{d-1}, 0]

This is a scalar. With random Gaussian cores, these scalars approximate
i.i.d. random variables, so argmin over nonzero cells preserves Jaccard:
    Pr[argmin_{A} rank_j == argmin_{B} rank_j] ≈ J(A, B)

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
        self.cfg = config
        rng = np.random.default_rng(config.seed + 1)
        r = config.tt_rank
        ndim = config.ndim

        # k independent TT decompositions; each is a list of d cores.
        # Boundary ranks are 1 (standard TT), inner ranks are tt_rank.
        # Core shapes: (1, n_0, r), (r, n_1, r), ..., (r, n_{d-1}, 1)
        self.all_cores: list[list[np.ndarray]] = []

        for _ in range(config.num_hashes):
            cores: list[np.ndarray] = []
            for k_mode, n_k in enumerate(config.shape):
                r_left = 1 if k_mode == 0 else r
                r_right = 1 if k_mode == ndim - 1 else r
                core = rng.standard_normal((r_left, n_k, r_right)).astype(np.float32)
                # Left-orthogonalise for numerical stability
                core_mat = core.reshape(r_left * n_k, r_right)
                if r_left * n_k >= r_right:
                    core_mat, _ = np.linalg.qr(core_mat)
                    core = core_mat[:, :r_right].reshape(r_left, n_k, r_right)
                cores.append(core)
            self.all_cores.append(cores)

        self.param_count = sum(sum(c.size for c in cores) for cores in self.all_cores)
        self.full_param_count = int(np.prod(config.shape)) * config.num_hashes

        # Precompute stacked cores: _stacked_cores[m] shape (num_hashes, r_left, n_m, r_right)
        self._stacked_cores: list[np.ndarray] = [
            np.stack([self.all_cores[j][m] for j in range(config.num_hashes)], axis=0)
            for m in range(config.ndim)
        ]

        # Precompute prefix product of the first (ndim-1) cores contracted over all
        # (i0, i1, ..., i_{d-2}) index combinations.  This is a one-time O(K·n^(d-1)·r²)
        # cost that reduces each hash_tensor call to two gathers + one elementwise dot.
        #
        # _prefix_flat shape: (K, n0*...*n_{d-2}, r)  - contiguous for fast fancy indexing
        # _last_core_T  shape: (K, n_{d-1}, r)         - last core transposed for gather
        K_ = config.num_hashes
        if config.ndim >= 2:
            prefix = self._stacked_cores[0][:, 0, :, :]  # (K, n0, r)  - squeeze r_left=1
            for m in range(1, config.ndim - 1):
                sc = self._stacked_cores[m]               # (K, r_left, n_m, r_right)
                n_prev = prefix.size // (K_ * sc.shape[1])
                pre2d = prefix.reshape(K_, n_prev, sc.shape[1])
                # result[k,p,i,s] = Σ_r pre2d[k,p,r] * sc[k,r,i,s]
                result = np.einsum("kpr,kris->kpis", pre2d, sc, optimize=True)
                prefix = result.reshape((K_,) + config.shape[: m + 1] + (sc.shape[3],))

            n_spatial = int(np.prod(config.shape[:-1]))
            self._prefix_flat = np.ascontiguousarray(
                prefix.reshape(K_, n_spatial, prefix.shape[-1])
            )  # (K, n_spa, r)
            self._last_core_T = np.ascontiguousarray(
                self._stacked_cores[-1][:, :, :, 0].transpose(0, 2, 1)
            )  # (K, n_last, r)  - squeeze r_right=1, then swap axes

    def hash_tensor(self, tensor: np.ndarray) -> np.ndarray:
        """
        Compute MinHash signature using precomputed prefix cores.

        For ndim >= 2 the score of cell (i0, ..., i_{d-1}) under plane k is:
            score_k = prefix_flat[k, flat(i0,...,i_{d-2}), :] · last_core_T[k, i_{d-1}, :]
        Both operands are gathered with a single fancy-index step; no Python loops
        over hash planes are needed.

        Returns: signature of shape (num_hashes,) - integer hash values.
        """
        assert tensor.shape == self.cfg.shape, f"Expected {self.cfg.shape}, got {tensor.shape}"

        nonzero_idx = np.argwhere(tensor > 0)  # (nnz, ndim)
        if len(nonzero_idx) == 0:
            return np.zeros(self.cfg.num_hashes, dtype=np.int32)

        if self.cfg.ndim == 1:
            # Degenerate case: score is just the single core value.
            scores = self._stacked_cores[0][:, 0, nonzero_idx[:, 0], 0]  # (K, nnz)
        else:
            # Flatten first (ndim-1) mode indices -> single 1-D index for prefix gather.
            flat_idx = np.ravel_multi_index(
                nonzero_idx[:, :-1].T.astype(np.intp), self.cfg.shape[:-1]
            )  # (nnz,)
            pref  = self._prefix_flat[:, flat_idx, :]           # (K, nnz, r)
            last  = self._last_core_T[:, nonzero_idx[:, -1], :] # (K, nnz, r)
            scores = (pref * last).sum(-1)                      # (K, nnz)

        min_cells = np.argmin(scores, axis=1)          # (K,)
        multi_indices = nonzero_idx[min_cells]          # (K, ndim)
        return np.ravel_multi_index(multi_indices.T, self.cfg.shape).astype(np.int32)

    def jaccard_from_signatures(self, sig_a: np.ndarray, sig_b: np.ndarray) -> float:
        """Estimate Jaccard similarity. J(A,B) ≈ #(sig_a == sig_b) / k"""
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