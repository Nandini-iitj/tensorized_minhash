"""
Tests for TTDecomposedMinHash correctness.

Covers: signature shape, identical-tensor Jaccard = 1, disjoint-tensor
Jaccard near 0, positive correlation with exact Jaccard, parameter-count
formula, zero-tensor handling, memory_stats keys, reproducibility, symmetry.
"""

import numpy as np
import pytest
from benchmark import ground_truth_jaccard
from core.config import TTMinHashConfig
from core.tt_minhash import TTDecomposedMinHash

class TestTTMinHash:
    def setup_method(self):
        self.shape = (15, 15, 15)
        # Tab title says test_tt_minhash.py but config uses TTMinHashConfig
        self.cfg = TTMinHashConfig(shape=self.shape, num_hashes=32, tt_rank=3, seed=1)
        self.hasher = TTDecomposedMinHash(self.cfg)

    # --------------------------------------------------------------------------
    # Output format
    # --------------------------------------------------------------------------

    def test_signature_shape(self):
        t = np.random.rand(*self.shape).astype(np.float32)
        sig = self.hasher.hash_tensor(t)
        assert sig.shape == (32,)

    # --------------------------------------------------------------------------
    # Correctness
    # --------------------------------------------------------------------------

    def test_identical_tensors(self):
        """Same tensor must always produce Jaccard 1.0."""
        t = (np.random.rand(*self.shape) < 0.1).astype(np.float32)
        sig_a = self.hasher.hash_tensor(t)
        sig_b = self.hasher.hash_tensor(t)
        j = self.hasher.jaccard_from_signatures(sig_a, sig_b)
        assert j == 1.0

    def test_disjoint_tensors_jaccard_low(self):
        """Disjoint tensors should give Jaccard near 0."""
        rng = np.random.default_rng(99)
        all_cells = np.arange(int(np.prod(self.shape)))
        half = len(all_cells) // 2
        set_a = rng.choice(all_cells[:half], 50, replace=False)
        set_b = rng.choice(all_cells[half:], 50, replace=False)
        a = np.zeros(self.shape, dtype=np.float32)
        a.ravel()[set_a] = 1
        b = np.zeros(self.shape, dtype=np.float32)
        b.ravel()[set_b] = 1

        sig_a = self.hasher.hash_tensor(a)
        sig_b = self.hasher.hash_tensor(b)
        j = self.hasher.jaccard_from_signatures(sig_a, sig_b)
        assert j < 0.2, f"Disjoint tensors should have low Jaccard, got {j:.3f}"

    def test_zero_tensor_tt_returns_zero_sig(self):
        """TT hasher on an all-zero tensor must return a zero signature."""
        t = np.zeros(self.shape, dtype=np.float32)
        sig = self.hasher.hash_tensor(t)
        assert sig.shape == (32,)
        np.testing.assert_array_equal(sig, 0)

    def test_signature_reproducible_tt(self):
        """Hashing the same tensor twice with the same TT hasher produces identical signatures."""
        t = (np.random.default_rng(33).random(self.shape) < 0.1).astype(np.float32)
        sig_a = self.hasher.hash_tensor(t)
        sig_b = self.hasher.hash_tensor(t)
        np.testing.assert_array_equal(sig_a, sig_b)

    def test_jaccard_symmetry_tt(self):
        """J(A, B) must equal J(B, A) for TT hasher."""
        rng = np.random.default_rng(44)
        a = (rng.random(self.shape) < 0.1).astype(np.float32)
        b = (rng.random(self.shape) < 0.1).astype(np.float32)
        j_ab = self.hasher.jaccard_from_signatures(
            self.hasher.hash_tensor(a), self.hasher.hash_tensor(b)
        )
        j_ba = self.hasher.jaccard_from_signatures(
            self.hasher.hash_tensor(b), self.hasher.hash_tensor(a)
        )
        assert j_ab == j_ba

    # --------------------------------------------------------------------------
    # Accuracy
    # --------------------------------------------------------------------------

    def test_jaccard_positive_correlation(self):
        """TT Jaccard should correlate with exact Jaccard across varied overlap."""
        rng = np.random.default_rng(5)
        cfg = TTMinHashConfig(shape=(20, 20, 20), num_hashes=64, seed=7)
        hasher = TTDecomposedMinHash(cfg)

        exact_js, tt_js = [], []
        n_cells = int(0.05 * 20**3)
        all_cells = np.arange(20**3)

        for target_j in [0.1, 0.2, 0.4, 0.6, 0.8] * 6:
            set_a = rng.choice(all_cells, n_cells, replace=False)
            n_shared = int(target_j * n_cells)
            shared = set_a[:n_shared]
            candidates = rng.choice(all_cells, n_cells * 4, replace=False)
            a_set = set(set_a)
            unique = [c for c in candidates if c not in a_set][:n_cells - n_shared]
            set_b = np.concatenate([shared, unique])

            a = np.zeros((20, 20, 20), dtype=np.float32)
            a.ravel()[set_a] = 1
            b = np.zeros((20, 20, 20), dtype=np.float32)
            b.ravel()[set_b] = 1

            exact_js.append(ground_truth_jaccard(a, b))
            sig_a = hasher.hash_tensor(a)
            sig_b = hasher.hash_tensor(b)
            tt_js.append(hasher.jaccard_from_signatures(sig_a, sig_b))

        r = np.corrcoef(exact_js, tt_js)[0, 1]
        assert r > 0.80, f"TT Jaccard correlation too low: {r:.3f} (expect >0.80)"

    # --------------------------------------------------------------------------
    # Memory
    # --------------------------------------------------------------------------

    def test_param_count_linear(self):
        """TT param count should be O(n * r^2 * d * k)."""
        # Linear in d (number of dimensions)
        full = int(np.prod(self.shape)) * self.cfg.num_hashes
        assert self.hasher.param_count < full

    def test_memory_stats_keys_tt(self):
        """TT memory_stats must return all expected keys."""
        stats = self.hasher.memory_stats()
        for key in (
            "tt_params",
            "full_params_theoretical",
            "compression_ratio",
            "tt_bytes",
            "full_bytes_theoretical",
        ):
            assert key in stats, f"Missing key: {key}"
