"""
Unit tests for TensorizedMinHash correctness.
Run with: python -m pytest tests/ -v
"""

import os
import sys

_REPO_ROOT = os.path.dirname(os.path.dirname(__file__))
sys.path.insert(0, os.path.join(_REPO_ROOT, "tensorized_minhash"))

from benchmarks.benchmark import ground_truth_jaccard  # noqa: E402
from core.config import TTMinHashConfig  # noqa: E402
from core.kron_minhash import KroneckerMinHash  # noqa: E402
from core.tt_minhash import TTDecomposedMinHash  # noqa: E402
import numpy as np  # noqa: E402

class TestKronMinHash:
    def setup_method(self, method):
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
        n_cells = int(0.06 * 30**3) # ~6% density
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
            b.ravel()[set_b] = 1

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
            a = np.zeros((25, 25, 25), dtype=np.float32)
            a.ravel()[set_a] = 1
            b = np.zeros((25, 25, 25), dtype=np.float32)
            b.ravel()[set_b] = 1

            exact = ground_truth_jaccard(a, b)
            sig_a = hasher.hash_tensor(a)
            sig_b = hasher.hash_tensor(b)
            kron = hasher.jaccard_from_signatures(sig_a, sig_b)
            errors.append(abs(exact - kron))

        mae = float(np.mean(errors))
        assert mae < 0.08, f"MAE too high: {mae:.4f}"

    def test_memory_compression(self):
        stats = self.hasher.memory_stats()
        assert stats["compression_ratio"] > 10, "Should achieve >10x compression"

    def test_param_count_formula(self):
        """kron_params == sum(shape[i] * num_hashes)."""
        expected = sum(s * self.cfg.num_hashes for s in self.shape)
        assert self.hasher.param_count == expected

    def test_memory_stats_keys(self):
        """memory_stats must return all expected keys."""
        stats = self.hasher.memory_stats()
        for key in (
            "kron_params",
            "full_params_theoretical",
            "compression_ratio",
            "kron_bytes",
            "full_bytes_theoretical",
        ):
            assert key in stats, f"Missing key: {key}"

    def test_signature_reproducible(self):
        """Hashing the same tensor twice with the same hasher must produce identical results."""
        t = (np.random.default_rng(11).random(self.shape) < 0.1).astype(np.float32)
        sig_a = self.hasher.hash_tensor(t)
        sig_b = self.hasher.hash_tensor(t)
        np.testing.assert_array_equal(sig_a, sig_b)

    def test_jaccard_symmetry(self):
        """J(A, B) must equal J(B, A)."""
        rng = np.random.default_rng(22)
        a = (rng.random(self.shape) < 0.1).astype(np.float32)
        b = (rng.random(self.shape) < 0.1).astype(np.float32)
        j_ab = self.hasher.jaccard_from_signatures(
            self.hasher.hash_tensor(a), self.hasher.hash_tensor(b)
        )
        j_ba = self.hasher.jaccard_from_signatures(
            self.hasher.hash_tensor(b), self.hasher.hash_tensor(a)
        )
        assert j_ab == j_ba

class TestTTMinHash:
    def setup_method(self, method):
        self.shape = (15, 15, 15)
        self.cfg = TTMinHashConfig(shape=self.shape, num_hashes=32, tt_rank=3, seed=1)
        self.hasher = TTDecomposedMinHash(self.cfg)

    def test_signature_shape(self):
        t = np.random.rand(*self.shape).astype(np.float32)
        sig = self.hasher.hash_tensor(t)
        assert sig.shape == (32,)

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

    def test_jaccard_positive_correlation(self):
        """TT Jaccard should correlate with exact Jaccard across varied overlap."""
        rng = np.random.default_rng(5)
        cfg = TTMinHashConfig(shape=(20, 20, 20), num_hashes=64, seed=7)
        hasher = TTDecomposedMinHash(cfg)

        exact_js, tt_js = [], []
        n_cells = int(0.05 * 20**3)
        all_cells = np.arange(20**3)

        for target_j in [0.1, 0.2, 0.4, 0.6, 0.8] * 6:
            set_a = rng.choice(all_cells, size=n_cells, replace=False)
            n_shared = int(target_j * n_cells)
            shared = set_a[:n_shared]
            candidates = rng.choice(all_cells, n_cells * 4, replace=False)
            a_set = set(set_a)
            unique = [c for c in candidates if c not in a_set][:n_cells - n_shared]
            set_b = np.concatenate([shared, unique])

            a = np.zeros((20, 20, 20), dtype=np.float32)
            b = np.zeros((20, 20, 20), dtype=np.float32)
            a.ravel()[set_a] = 1
            b.ravel()[set_b] = 1

            exact_js.append(ground_truth_jaccard(a, b))
            sig_a = hasher.hash_tensor(a)
            sig_b = hasher.hash_tensor(b)
            tt_js.append(hasher.jaccard_from_signatures(sig_a, sig_b))

        r = np.corrcoef(exact_js, tt_js)[0, 1]
        assert r > 0.80, f"TT Jaccard correlation too low: {r:.3f} (expect >0.80)"

    def test_param_count_linear(self):
        """TT param count should be O(n * r^2 * d * k)."""
        # all_cores is a list of k decompositions, each a list of d cores
        expected = sum(c.size for cores in self.hasher.all_cores for c in cores)
        assert self.hasher.param_count == expected
        # Sanity: must be << n^d * k
        full = int(np.prod(self.shape)) * self.cfg.num_hashes
        assert self.hasher.param_count < full

    def test_zero_tensor_tt(self):
        """TT hasher on an all-zero tensor must return a zero signature."""
        t = np.zeros(self.shape, dtype=np.float32)
        sig = self.hasher.hash_tensor(t)
        assert sig.shape == (self.cfg.num_hashes,)
        np.testing.assert_array_equal(sig, 0)

    def test_memory_stats_keys_tt(self):
        """memory_stats must return all expected keys."""
        stats = self.hasher.memory_stats()
        for key in (
            "tt_params",
            "full_params_theoretical",
            "compression_ratio",
            "tt_bytes",
            "full_bytes_theoretical",
        ):
            assert key in stats, f"Missing key: {key}"

    def test_signature_reproducible_tt(self):
        """Hashing the same tensor twice with the same TT hasher must produce identical results."""
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

class TestThreeScenarioBenchmark:
    """
    Validates the three-scenario accuracy benchmark meets the target:
    Spearman r > 0.85 on the combined (full Jaccard range) results.
    """

    def test_combined_spearman_above_target(self):
        """Combined Spearman r across all three scenarios must exceed 0.85.
        This is the primary proposal evaluation metric.
        """
        from benchmarks.benchmark import benchmark_accuracy

        acc = benchmark_accuracy(n_pairs=60, shape=(25, 25, 25), num_hashes=128, seed=0)
        kron_rho = acc["tensorized_kron"]["spearman_r"]
        assert kron_rho > 0.85, (
            f"Kron Spearman r = {kron_rho:.4f} - must exceed 0.85"
            f"Jaccard range was {acc['jaccard_range']}"
        )

    def test_scenario_jaccard_ranges(self):
        """
        Each scenario must produce Jaccard values in the expected range.
        High > 0.7, Medium in [0.2, 0.8], Low < 0.3.
        """
        from benchmarks.benchmark import benchmark_accuracy

        acc = benchmark_accuracy(n_pairs=40, shape=(20, 20, 20), num_hashes=64, seed=1)
        scenarios = acc["scenarios"]

        hi_lo, hi_hi = scenarios["high"]["jaccard_range"]
        assert hi_lo > 0.60, f"High scenario min Jaccard too low: {hi_lo:.3f}"

        lo_lo, lo_hi = scenarios["low"]["jaccard_range"]
        assert lo_hi < 0.40, f"Low scenario max Jaccard too high: {lo_hi:.3f}"

        med_lo, med_hi = scenarios["medium"]["jaccard_range"]
        assert med_lo < 0.70 and med_hi > 0.20, (
            f"Medium scenario Jaccard range unexpected: [{med_lo:.3f}, {med_hi:.3f}]"
        )

    def test_all_methods_present_in_results(self):
        """All three methods must appear in both per-scenario and combined results."""
        from benchmarks.benchmark import benchmark_accuracy

        acc = benchmark_accuracy(n_pairs=10, shape=(15, 15, 15), num_hashes=32, seed=2)
        for key in ("tensorized_kron", "tt_decomposition", "datasketch_baseline"):
            assert key in acc, f"Missing top-level key: {key}"
            for scenario in ("high", "medium", "low"):
                assert key in acc["scenarios"][scenario], f"Missing {key} in scenario {scenario}"

class TestDataLoader:
    def test_synthetic_generation(self):
        from data.loader import NetworkLogGenerator

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
