"""
Unit tests for TensorizedMinHash correctness.
Run with: python -m pytest tests/ -v
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import numpy as np
import pytest
from core.tt_minhash import KroneckerMinHash, TTDecomposedMinHash, TTMinHashConfig
from benchmarks.benchmark import ground_truth_jaccard


class TestKronMinHash:

    def setup_method(self):
        self.shape = (20, 20, 20)
        self.cfg = TTMinHashConfig(shape=self.shape, num_hashes=64, seed=0)
        self.hasher = KroneckerMinHash(self.cfg)

    def test_signature_shape(self):
        t = np.random.rand(*self.shape).astype(np.float32)
        sig = self.hasher.hash_tensor(t)
        assert sig.shape == (64,)

    def test_signature_dtype(self):
        t = np.random.rand(*self.shape).astype(np.float32)
        sig = self.hasher.hash_tensor(t)
        assert sig.dtype == np.int32

    def test_identical_tensors_jaccard_one(self):
        t = (np.random.rand(*self.shape) < 0.1).astype(np.float32)
        sig_a = self.hasher.hash_tensor(t)
        sig_b = self.hasher.hash_tensor(t)
        j = self.hasher.jaccard_from_signatures(sig_a, sig_b)
        assert j == 1.0, f"Expected 1.0 for identical tensors, got {j}"

    def test_zero_tensor(self):
        t = np.zeros(self.shape, dtype=np.float32)
        sig = self.hasher.hash_tensor(t)
        assert sig.shape == (64,)

    def test_jaccard_positive_correlation(self):
        """
        Tensorized Jaccard should track exact Jaccard across a wide range of
        overlap levels. We explicitly construct pairs with known target overlaps
        so the Jaccard values have high variance, making correlation meaningful.
        """
        rng = np.random.default_rng(0)
        cfg = TTMinHashConfig(shape=(30, 30, 30), num_hashes=128, seed=42)
        hasher = KroneckerMinHash(cfg)

        exact_js, kron_js = [], []
        n_cells = int(0.06 * 30**3)  # ~6% density
        all_cells = np.arange(30**3)

        for target_j in [0.1, 0.2, 0.3, 0.5, 0.7, 0.9] * 8:
            set_a = rng.choice(all_cells, size=n_cells, replace=False)
            n_shared = int(target_j * n_cells)
            shared = set_a[:n_shared]
            candidates = rng.choice(all_cells, n_cells * 4, replace=False)
            a_set = set(set_a)
            unique = [c for c in candidates if c not in a_set][:n_cells - n_shared]
            set_b = np.concatenate([shared, unique])

            a = np.zeros((30, 30, 30), dtype=np.float32)
            b = np.zeros((30, 30, 30), dtype=np.float32)
            a.ravel()[set_a] = 1
            b.ravel()[set_b[:n_cells]] = 1

            exact_js.append(ground_truth_jaccard(a, b))
            sig_a = hasher.hash_tensor(a)
            sig_b = hasher.hash_tensor(b)
            kron_js.append(hasher.jaccard_from_signatures(sig_a, sig_b))

        r = np.corrcoef(exact_js, kron_js)[0, 1]
        assert r > 0.85, f"Correlation too low: {r:.3f} (expect >0.85 with varied Jaccard range)"

    def test_jaccard_low_bias(self):
        """Mean absolute error between Kronecker and exact Jaccard should be small."""
        rng = np.random.default_rng(7)
        cfg = TTMinHashConfig(shape=(25, 25, 25), num_hashes=128, seed=1)
        hasher = KroneckerMinHash(cfg)

        errors = []
        n_cells = int(0.08 * 25**3)
        all_cells = np.arange(25**3)

        for _ in range(50):
            set_a = rng.choice(all_cells, n_cells, replace=False)
            set_b = rng.choice(all_cells, n_cells, replace=False)
            a = np.zeros((25, 25, 25), np.float32); a.ravel()[set_a] = 1
            b = np.zeros((25, 25, 25), np.float32); b.ravel()[set_b] = 1
            exact = ground_truth_jaccard(a, b)
            sig_a = hasher.hash_tensor(a)
            sig_b = hasher.hash_tensor(b)
            kron = hasher.jaccard_from_signatures(sig_a, sig_b)
            errors.append(abs(exact - kron))

        mae = float(np.mean(errors))
        assert mae < 0.08, f"MAE too high: {mae:.4f}"

    def test_memory_compression(self):
        stats = self.hasher.memory_stats()
        assert stats["compression_ratio"] > 10, "Should achieve >10× compression"

    def test_param_count_formula(self):
        """kron_params == sum(shape[i] * num_hashes)."""
        expected = sum(s * self.cfg.num_hashes for s in self.shape)
        assert self.hasher.param_count == expected


class TestTTMinHash:

    def setup_method(self):
        self.shape = (15, 15, 15)
        self.cfg = TTMinHashConfig(shape=self.shape, num_hashes=32, tt_rank=3, seed=1)
        self.hasher = TTDecomposedMinHash(self.cfg)

    def test_signature_shape(self):
        t = np.random.rand(*self.shape).astype(np.float32)
        sig = self.hasher.hash_tensor(t)
        assert sig.shape == (32,)

    def test_identical_tensors(self):
        t = (np.random.rand(*self.shape) < 0.1).astype(np.float32)
        sig_a = self.hasher.hash_tensor(t)
        sig_b = self.hasher.hash_tensor(t)
        j = self.hasher.jaccard_from_signatures(sig_a, sig_b)
        assert j == 1.0

    def test_param_count_linear(self):
        """TT param count should be O(n * r^2 * d)."""
        # Each core: (r, n, r) except boundaries (k, n, r) and (r, n, k)
        expected = sum(c.size for c in self.hasher.cores)
        assert self.hasher.param_count == expected
        # Sanity: must be << n^d * k
        full = int(np.prod(self.shape)) * self.cfg.num_hashes
        assert self.hasher.param_count < full


class TestDataLoader:

    def test_synthetic_generation(self):
        from data.loader import NetworkLogGenerator, NetworkTensorBuilder
        gen = NetworkLogGenerator(n_src=20, n_dst=20, n_port=20, n_benign=500, seed=0)
        df, attacks = gen.generate()
        assert len(df) > 0
        assert "src_ip" in df.columns
        assert "portscan" in attacks

    def test_tensor_builder(self):
        from data.loader import NetworkLogGenerator, NetworkTensorBuilder
        gen = NetworkLogGenerator(n_src=20, n_dst=20, n_port=20, seed=0)
        df, _ = gen.generate()
        builder = NetworkTensorBuilder(n_src=20, n_dst=20, n_port=20)
        t = builder.build_tensor(df)
        assert t.shape == (20, 20, 20)
        assert t.max() <= 1.0
        assert t.min() >= 0.0
        assert t.sum() > 0
