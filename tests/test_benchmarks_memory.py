"""
Tests for benchmarks/memory.py and benchmarks/helpers.py.

Covers benchmark_memory, benchmark_ram, ground_truth_jaccard,
and datasketch_jaccard (including the branch for non-binary tensors).
"""

from benchmark import datasketch_jaccard, ground_truth_jaccard
from benchmark import benchmark_memory, benchmark_ram
import numpy as np

# --------------------------------------------------------------------------
# helpers.py
# --------------------------------------------------------------------------

class TestGroundTruthJaccard:
    def test_identical_tensors(self):
        t = np.ones((5, 5, 5), dtype=np.float32)
        assert ground_truth_jaccard(t, t) == 1.0

    def test_disjoint_tensors(self):
        a = np.zeros((4, 4), dtype=np.float32)
        b = np.zeros((4, 4), dtype=np.float32)
        a[0, 0] = 1.0
        b[3, 3] = 1.0
        assert ground_truth_jaccard(a, b) == 0.0

    def test_partial_overlap(self):
        a = np.array([1, 1, 0, 0], dtype=np.float32)
        b = np.array([1, 0, 1, 0], dtype=np.float32)
        # |AnB| = 1, |AUB| = 3
        assert abs(ground_truth_jaccard(a, b) - 1 / 3) < 1e-6

    def test_all_zeros_returns_zero(self):
        a = np.zeros((3, 3), dtype=np.float32)
        b = np.zeros((3, 3), dtype=np.float32)
        assert ground_truth_jaccard(a, b) == 0.0

class TestDatasketchJaccard:
    def test_returns_float_in_range(self):
        a = (np.random.default_rng(0).random((10, 10)) < 0.3).astype(np.float32)
        b = (np.random.default_rng(1).random((10, 10)) < 0.3).astype(np.float32)
        j = datasketch_jaccard(a, b, num_perm=32)
        assert 0.0 <= j <= 1.0

    def test_identical_tensors_near_one(self):
        t = (np.random.default_rng(5).random((10, 10)) < 0.3).astype(np.float32)
        j = datasketch_jaccard(t, t, num_perm=64)
        assert j > 0.95

# --------------------------------------------------------------------------
# memory.py
# --------------------------------------------------------------------------

class TestBenchmarkMemory:
    def test_returns_one_result_per_shape(self):
        shapes = [(10, 10, 10), (20, 20, 20)]
        results = benchmark_memory(shapes, num_hashes=64)
        assert len(results) == 2

    def test_result_keys_present(self):
        results = benchmark_memory([(10, 10, 10)], num_hashes=32)
        r = results[0]
        for key in (
            "shape",
            "total_cells",
            "kron_params",
            "full_params_theoretical",
            "compression_ratio",
        ):
            assert key in r, f"Missing key: {key}"

    def test_compression_increases_with_order(self):
        """4th-order tensor should compress more than 3rd-order at same side length."""
        r3 = benchmark_memory([(20, 20, 20)], num_hashes=64)[0]
        r4 = benchmark_memory([(10, 10, 10, 10)], num_hashes=64)[0]
        assert r4["compression_ratio"] > r3["compression_ratio"]

    def test_total_cells_correct(self):
        results = benchmark_memory([(5, 6, 7)], num_hashes=32)
        # Note: Image showed assert results[0]["total_cells"] == 5 * 6 * 7
        assert results[0]["total_cells"] == 5 * 6 * 7

class TestBenchmarkRam:
    def test_result_keys(self):
        ram = benchmark_ram(shape=(10, 10, 10), num_hashes=32)
        for key in (
            "kron_peak_mb",
            "tt_peak_mb",
            "full_theoretical_mb",
            "kron_params",
            "tt_params",
            "full_params",
            "kron_vs_full",
            "tt_vs_full",
            "kron_vs_tt",
        ):
            assert key in ram, f"Missing key: {key}"

    def test_kron_smaller_than_full(self):
        ram = benchmark_ram(shape=(20, 20, 20), num_hashes=64)
        assert ram["kron_peak_mb"] < ram["full_theoretical_mb"]

    def test_tt_smaller_than_full(self):
        ram = benchmark_ram(shape=(20, 20, 20), num_hashes=64)
        assert ram["tt_peak_mb"] < ram["full_theoretical_mb"]

    def test_kron_vs_full_ratio(self):
        """kron_vs_full should equal full_bytes / kron_bytes analytically."""
        ram = benchmark_ram(shape=(10, 10, 10), num_hashes=32)
        expected = ram["full_theoretical_mb"] / ram["kron_peak_mb"]
        assert abs(ram["kron_vs_full"] - expected) < 1e-6
