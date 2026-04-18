"""
Tests for benchmarks/accuracy.py - benchmark_accuracy_from_tensors path.

The three-scenario benchmark_accuracy is covered in test_benchmark_accuracy.py.
This module focuses on the real-tensor variant and internal helpers.
"""

from benchmarks.accuracy import (
    _generate_high_similarity_pair,
    _generate_low_similarity_pair,
    _generate_medium_similarity_pair,
    benchmark_accuracy_from_tensors,
)
import numpy as np

def _make_tensors(shape=(15, 15, 15), n=10, seed=7):
    rng = np.random.default_rng(seed)
    return [(rng.random(shape) < 0.08).astype(np.float32) for _ in range(n)]

class TestBenchmarkAccuracyFromTensors:
    def test_result_keys_present(self):
        tensors = _make_tensors()
        result = benchmark_accuracy_from_tensors(
            tensors, shape=(15, 15, 15), num_hashes=32, n_pairs=10
        )
        for key in (
            "tensorized_kron",
            "tt_decomposition",
            "datasketch_baseline",
            "n_windows",
            "jaccard_range",
        ):
            assert key in result, f"Missing key: {key}"

    def test_metrics_in_range(self):
        tensors = _make_tensors()
        result = benchmark_accuracy_from_tensors(
            tensors, shape=(15, 15, 15), num_hashes=32, n_pairs=10
        )
        for method in ("tensorized_kron", "tt_decomposition", "datasketch_baseline"):
            m = result[method]
            assert 0.0 <= m["mae"] <= 1.0
            assert 0.0 <= m["rmse"] <= 1.0
            assert 0.0 <= m["avg_estimated"] <= 1.0

    def test_n_windows_correct(self):
        tensors = _make_tensors(n=8)
        result = benchmark_accuracy_from_tensors(
            tensors, shape=(15, 15, 15), num_hashes=32, n_pairs=5
        )
        assert result["n_windows"] == 8

    def test_fewer_pairs_than_available(self):
        """When n_pairs > available, all pairs are used without error."""
        tensors = _make_tensors(n=4) # C(4,2)=6 pairs
        result = benchmark_accuracy_from_tensors(
            tensors, shape=(15, 15, 15), num_hashes=32, n_pairs=100
        )
        assert result["n_pairs"] <= 6

class TestPairGenerators:
    """Unit tests for the internal similarity-pair generators."""

    shape = (10, 10, 10)

    def test_high_similarity_jaccard_range(self):
        import numpy as np
        rng = np.random.default_rng(0)
        jaccards = []
        from benchmarks.benchmark import ground_truth_jaccard

        for _ in range(20):
            a, b = _generate_high_similarity_pair(rng, self.shape)
            jaccards.append(ground_truth_jaccard(a, b))
        assert min(jaccards) > 0.5, f"High similarity min too low: {min(jaccards):.3f}"

    def test_low_similarity_jaccard_range(self):
        import numpy as np
        rng = np.random.default_rng(1)
        jaccards = []
        from benchmarks.benchmark import ground_truth_jaccard

        for _ in range(20):
            a, b = _generate_low_similarity_pair(rng, self.shape)
            jaccards.append(ground_truth_jaccard(a, b))
        assert max(jaccards) < 0.5, f"Low similarity max too high: {max(jaccards):.3f}"

    def test_medium_similarity_shape(self):
        import numpy as np
        rng = np.random.default_rng(2)
        a, b = _generate_medium_similarity_pair(rng, self.shape)
        assert a.shape == self.shape
        assert b.shape == self.shape
        assert a.dtype == np.float32
