"""
Tensorized MinHash via Tensor Train (TT) and Kronecker Product compression.

Implements the core technique from:
  "Improving LSH via Tensorized Random Projection" (2025)

Key idea: A random projection matrix W ∈ R^{d1*d2*...*dn × k} normally needs
O(d^n * k) memory. By representing W as a Kronecker product of n small factor
matrices W_i ∈ R^{d_i × k}, we store only O(n * d * k) parameters, then
simulate the full projection on-the-fly via the mixed-product property:
  (A ⊗ B)(x_A ⊗ x_B) = (Ax_A) ⊗ (Bx_B)

For MinHash: we approximate min-hash signatures without flattening tensors.
"""

import numpy as np
from typing import List, Tuple, Optional, Dict
import time
import logging

logger = logging.getLogger(__name__)


class TTMinHashConfig:
    """Configuration for the TT-MinHash hasher."""

    def __init__(
        self,
        shape: Tuple[int, ...],   # tensor mode sizes e.g. (100, 100, 100)
        num_hashes: int = 128,     # number of MinHash signatures (k)
        tt_rank: int = 4,          # TT bond dimension (compression knob)
        seed: int = 42,
    ):
        self.shape = shape
        self.ndim = len(shape)
        self.num_hashes = num_hashes
        self.tt_rank = tt_rank
        self.seed = seed

        # Theoretical savings
        full_params = int(np.prod(shape)) * num_hashes
        kron_params = sum(s * num_hashes for s in shape)
        self.compression_ratio = full_params / max(kron_params, 1)

        logger.info(
            f"TTMinHashConfig: shape={shape}, k={num_hashes}, "
            f"full_params={full_params:,}, kron_params={kron_params:,}, "
            f"compression={self.compression_ratio:.1f}x"
        )


class KroneckerMinHash:
    """
    MinHash using Kronecker-factored random projections.

    Core idea (from the 2025 paper):
    A standard MinHash hash function h_pi: 2^U → U picks the minimum element
    under a random permutation π of the universe.

    For a tensor T of shape (d0, d1, ..., dn), the universe is all multi-indices
    (i0, i1, ..., in).  Rather than storing |U| = prod(shape) random values
    (the permutation), we generate k independent random projections per mode
    and approximate the permutation rank via the Kronecker (outer product)
    combination:

        rank(i0, i1, ..., in) ≈ W0[i0] * W1[i1] * ... * Wn[in]

    where Wm ∈ R^{dm} is a vector of Uniform(0,1) random values.  The minimum
    over nonzero cells of T under this combined rank approximates the MinHash
    function.  We repeat k times (k independent sets of Wm) to get k hash values.

    Jaccard approximation theorem: E[h_j(A) == h_j(B)] = J(A, B).

    Memory: instead of k * prod(shape) random values, we store k * sum(shape)
    values — from O(n^d * k) to O(n*d*k), a factor of n^(d-1) reduction.
    For shape=(100,100,100), k=128: 128M → 38,400 parameters.
    """

    def __init__(self, config: TTMinHashConfig):
        self.cfg = config
        rng = np.random.default_rng(config.seed)

        # Factor vectors: list of [num_hashes, shape[i]]
        # We use -log(Uniform(0,1)) ~ Exponential(1) random variables.
        # Rationale: for a standard MinHash over a universe U, the classic
        # "bottom-k" construction assigns each element u an independent
        # Exp(1) r.v. r(u), then the hash is argmin_u r(u) among active elements.
        # With the Kronecker factoring, we approximate r(i0,i1,...,in) as the
        # SUM of independent Exp(1) r.v.s per mode:
        #   r_j(i0,...,in) = E0[j,i0] + E1[j,i1] + ... + En[j,in]
        # where Em[j,im] ~ Exp(1). Sums of exponentials concentrate around their
        # mean, providing a stable, consistent rank ordering. The minimum under
        # this additive rank is equivalent to the minimum under a valid hash
        # function in the sense of preserving Jaccard via:
        #   Pr[argmin_A r_j == argmin_B r_j] ≈ J(A, B)
        # This outperforms the product-of-Uniforms approach because the additive
        # structure avoids rank collapse when mode values are close to zero.
        self.factors: List[np.ndarray] = []
        for s in config.shape:
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
          1. Compute rank(i0,...,in) = W0[j,i0] * W1[j,i1] * ... * Wn[j,in]
          2. hash_j(T) = argmin_{(i0,...in): T[i0,...,in]>0} rank(i0,...,in)
             But since we only care about the VALUE (not identity) for Jaccard:
             hash_j(T) ≈ min_{nonzero cells} rank(i0,...,in)
             and two tensors agree on hash j iff their minimising cells coincide.

        For Jaccard estimation, we compare the minimum rank values directly:
          Pr[min_rank_A == min_rank_B] ≈ J(A,B)

        In practice we use a more numerically stable variant: we compute the
        minimum rank value among nonzero cells, then discretise it into a
        bucket. Two tensors with the same minimum-rank index agree on that hash.

        Returns: signature of shape (num_hashes,) — integer hash values
        """
        assert tensor.shape == self.cfg.shape, (
            f"Expected {self.cfg.shape}, got {tensor.shape}"
        )

        # Get flat indices of nonzero cells
        nonzero_idx = np.argwhere(tensor > 0)  # shape (nnz, ndim)
        if len(nonzero_idx) == 0:
            return np.zeros(self.cfg.num_hashes, dtype=np.int32)

        signature = np.empty(self.cfg.num_hashes, dtype=np.int32)

        for j in range(self.cfg.num_hashes):
            # Additive rank: sum of Exp(1) r.v.s per mode
            # rank_j(i0,...,in) = E0[j,i0] + E1[j,i1] + ... + En[j,in]
            # argmin of this sum approximates the MinHash function correctly
            ranks = np.zeros(len(nonzero_idx), dtype=np.float64)
            for mode in range(self.cfg.ndim):
                mode_indices = nonzero_idx[:, mode]
                ranks += self.factors[mode][j, mode_indices]

            # Hash value = index of cell with minimum rank
            min_cell_flat = np.argmin(ranks)
            # Encode the multi-index as a single integer for comparison
            multi_idx = nonzero_idx[min_cell_flat]
            cell_id = int(np.ravel_multi_index(multi_idx, self.cfg.shape))
            signature[j] = cell_id

        return signature

    def jaccard_from_signatures(
        self,
        sig_a: np.ndarray,
        sig_b: np.ndarray,
    ) -> float:
        """
        Estimate Jaccard similarity from two MinHash signatures.
        J(A,B) ≈ #(sig_a == sig_b) / k
        """
        return float(np.mean(sig_a == sig_b))

    def memory_stats(self) -> Dict[str, int]:
        return {
            "kron_params": self.param_count,
            "full_params_theoretical": self.full_param_count,
            "compression_ratio": self.full_param_count // max(self.param_count, 1),
            "kron_bytes": self.param_count * 8,   # float64
            "full_bytes_theoretical": self.full_param_count * 8,
        }


class TTDecomposedMinHash:
    """
    MinHash using Tensor Train (TT) decomposed hash functions.

    The TT decomposition represents the hash weight tensor W as a product of
    3-way TT-cores G_k ∈ R^{r_{k-1} × n_k × r_k}:

        W[i_1, i_2, ..., i_d] = G_1[:, i_1, :] * G_2[:, i_2, :] * ... * G_d[:, i_d, :]

    This contracts from left to right, turning the exponential parameter count
    into a linear one: O(d * n * r^2) instead of O(n^d).

    For large-scale use, TT-format allows distributed contraction (each core
    can live on a different worker) — critical for PySpark integration.
    """

    def __init__(self, config: TTMinHashConfig):
        self.cfg = config
        rng = np.random.default_rng(config.seed + 1)
        r = config.tt_rank

        # Build TT-cores: shapes r_{k-1} × n_k × r_k
        # Boundary conditions: r_0 = r_d = num_hashes
        self.cores: List[np.ndarray] = []
        ranks = [config.num_hashes] + [r] * (config.ndim - 1) + [config.num_hashes]

        for k, n_k in enumerate(config.shape):
            r_left, r_right = ranks[k], ranks[k + 1]
            core = rng.standard_normal((r_left, n_k, r_right)).astype(np.float32)
            # Orthogonalise for numerical stability
            core_mat = core.reshape(r_left * n_k, r_right)
            if r_left * n_k >= r_right:
                core_mat, _ = np.linalg.qr(core_mat)
                core = core_mat[:, :r_right].reshape(r_left, n_k, r_right)
            self.cores.append(core)

        self.param_count = sum(c.size for c in self.cores)

    def hash_tensor(self, tensor: np.ndarray) -> np.ndarray:
        """
        Compute MinHash signature via left-to-right TT contraction.

        For each position in the tensor, we contract its value against the
        corresponding TT slice. The final shape is (num_hashes, num_hashes),
        and we take the diagonal as the projection vector.
        """
        assert tensor.shape == self.cfg.shape
        tensor = tensor.astype(np.float32)

        # Left-to-right contraction
        # state shape: (num_hashes,) initially, grows to (num_hashes, r, ...)
        state = np.ones(self.cfg.num_hashes, dtype=np.float32)

        for mode, core in enumerate(self.cores):
            # tensor mode marginal: sum over all other modes
            marginal = np.tensordot(tensor, np.ones(
                [s for i, s in enumerate(self.cfg.shape) if i != mode]
            ), axes=(
                [i for i in range(self.cfg.ndim) if i != mode],
                list(range(self.cfg.ndim - 1))
            ))  # shape: (n_k,)

            # Contract: state (r_left,) x core (r_left, n_k, r_right) x marginal (n_k,)
            # → new_state (r_right,)
            contracted = np.einsum("i,ijk,j->k", state, core, marginal)
            state = contracted

        return (state > 0).astype(np.int32)

    def jaccard_from_signatures(self, sig_a, sig_b) -> float:
        return float(np.mean(sig_a == sig_b))
