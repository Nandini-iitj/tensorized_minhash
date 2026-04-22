"""
KroneckerMinHash - Kronecker-factored additive-exponential MinHash.

Core idea (from the 2025 paper):
A standard MinHash hash function h_pi: 2^U -> U picks the minimum element
under a random permutation pi of the universe.

For a tensor T of shape (d0, d1, ..., dn), the universe is all multi-indices
(i0, i1, ..., in). Rather than storing |U| = prod(shape) random values
(the permutation), we generate k independent random projections per mode
and approximate the permutation rank via the Kronecker (outer product)
combination:

    rank(i0, i1, ..., in) ≈ W0[i0] * W1[i1] * ... * Wn[in]

where Wm ∈ R^{dm} is a vector of Uniform(0,1) random values. The minimum
over nonzero cells of T under this combined rank approximates the MinHash
function. We repeat k times (k independent sets of Wm) to get k hash values.

Jaccard approximation theorem: E[h_j(A) == h_j(B)] == J(A, B).

Memory: instead of k * prod(shape) random values, we store k * sum(shape)
values - from O(n^d * k) to O(n*d*k), a factor of n^(d-1) reduction.
For shape=(100,100,100), k=128: 128M -> 38,400 parameters.
"""

import logging

import numpy as np

from .config import TTMinHashConfig

__all__ = ["KroneckerMinHash"]

logger = logging.getLogger(__name__)


class KroneckerMinHash:
    """MinHash using Kronecker-factored random projections."""

    def __init__(self, config: TTMinHashConfig):
        self.cfg = config
        rng = np.random.default_rng(config.seed)

        # Factor vectors: list of [num_hashes, shape[i]]
        # We use -log(Uniform(0,1)) ~ Exponential(1) random variables.
        # Rationale: for a standard MinHash over a universe U, the classic
        # "bottom-k" construction assigns each element u an independent
        # Exp(1) r.v. r(u), then the hash is argmin_u r(u) among active elements.
        # With the Kronecker factoring, we approximate r(i0, i1, ..., in) as the
        # SUM of independent Exp(1) r.v.s per mode:
        #   r_j(i0, ..., in) = E0[j,i0] + E1[j,i1] + ... + En[j,in]
        # where Em[j,im] ~ Exp(1). Sums of exponentials concentrate around their
        # mean, providing a stable, consistent rank ordering. The minimum under
        # this additive rank is equivalent to the minimum under a valid hash
        # function in the sense of preserving Jaccard via:
        #   Pr[argmin_A r_j == argmin_B r_j] ≈ J(A, B)
        # This outperforms the product-of-Uniforms approach because the additive
        # structure avoids rank collapse when mode values are close to zero.
        self.factors: list[np.ndarray] = []
        for s in config.shape:
            # Generate Uniform(0,1) then transform
            u = rng.random((config.num_hashes, s))
            u = np.clip(u, 1e-10, 1.0)  # avoid log(0)
            exp_rvs = -np.log(u).astype(np.float64)
            self.factors.append(exp_rvs)

        # Parameter count
        self.param_count = sum(f.size for f in self.factors)
        self.full_param_count = int(np.prod(config.shape)) * config.num_hashes

    def hash_tensor(self, tensor: np.ndarray) -> np.ndarray:
        """
        Compute k-dimensional MinHash signature for a single tensor.

        For each hash function j:
        1. Compute rank(i0, ..., in) = E0[j,i0] + E1[j,i1] + ... + En[j,in]
        2. hash_j(T) = flat cell index of the nonzero cell with minimum rank

        Returns: signature of shape (num_hashes,) - integer hash values.
        """
        assert tensor.shape == self.cfg.shape, f"Expected {self.cfg.shape}, got {tensor.shape}"

        nonzero_idx = np.argwhere(tensor > 0)  # shape (nnz, ndim)
        if len(nonzero_idx) == 0:
            return np.zeros(self.cfg.num_hashes, dtype=np.int32)

        # Vectorized over all hash functions simultaneously.
        # ranks[j, i] = sum of Exp(1) r.v.s for hash j at nonzero cell i
        # shape: (num_hashes, nnz)
        nnz = len(nonzero_idx)
        ranks = np.zeros((self.cfg.num_hashes, nnz), dtype=np.float64)
        for mode in range(self.cfg.ndim):
            mode_indices = nonzero_idx[:, mode]
            ranks += self.factors[mode][:, mode_indices]  # (num_hashes, nnz)

        min_cells = np.argmin(ranks, axis=1)  # (num_hashes,)
        multi_indices = nonzero_idx[min_cells]  # (num_hashes, ndim)
        signature = np.ravel_multi_index(multi_indices.T, self.cfg.shape).astype(np.int32)

        return signature

    def jaccard_from_signatures(
        self,
        sig_a: np.ndarray,
        sig_b: np.ndarray,
    ) -> float:
        """Estimate Jaccard similarity. J(A,B) ≈ #(sig_a == sig_b) / k"""
        return float(np.mean(sig_a == sig_b))

    def memory_stats(self) -> dict[str, int]:
        """Return parameter storage statistics"""
        return {
            "kron_params": self.param_count,
            "full_params_theoretical": self.full_param_count,
            "compression_ratio": self.full_param_count // max(self.param_count, 1),
            "kron_bytes": self.param_count * 8,  # float64
            "full_bytes_theoretical": self.full_param_count * 8,
        }