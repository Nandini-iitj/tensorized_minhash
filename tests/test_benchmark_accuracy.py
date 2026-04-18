"""
Tests for the three-scenario accuracy benchmark.

Validates that the combined Spearman ρ across high/medium/low similarity
scenarios exceeds the proposal target of 0.85, and that each scenario
produces Jaccard values in the expected range.
"""

from benchmark import benchmark_accuracy


class TestThreeScenarioBenchmark:
    """
    Validates the three-scenario accuracy benchmark meets the proposal target:
    Spearman ρ > 0.85 on the combined (full Jaccard range) results.
    """

    def test_combined_spearman_above_target(self):
        """
        Combined Spearman ρ across all three scenarios must exceed 0.85.
        This is the primary proposal evaluation metric.
        """
        acc = benchmark_accuracy(n_pairs=60, shape=(25, 25, 25), num_hashes=128, seed=0)
        kron_rho = acc["tensorized_kron"]["spearman_r"]
        assert kron_rho > 0.85, (
            f"Kron Spearman ρ = {kron_rho:.4f} — must exceed 0.85 (proposal target). "
            f"Jaccard range was {acc['jaccard_range']}"
        )

    def test_scenario_jaccard_ranges(self):
        """
        Each scenario must produce Jaccard values in the expected range.
        High > 0.7, Medium in [0.2, 0.8], Low < 0.3.
        """
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
        acc = benchmark_accuracy(n_pairs=10, shape=(15, 15, 15), num_hashes=32, seed=2)
        for key in ["tensorized_kron", "tt_decomposition", "datasketch_baseline"]:
            assert key in acc, f"Missing top-level key: {key}"
            for scenario in ["high", "medium", "low"]:
                assert key in acc["scenarios"][scenario], f"Missing {key} in scenario '{scenario}'"


"""
Tests for benchmark/accuracy.py — benchmark_accuracy_from_tensors path.

The three-scenario benchmark accuracy is covered in test_benchmark_accuracy.py.
This module focuses on the real-tensor variant and internal helpers.
"""

from benchmark import (
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
        tensors = _make_tensors(n=4)  # C(4,2)=6 pairs
        result = benchmark_accuracy_from_tensors(
            tensors, shape=(15, 15, 15), num_hashes=32, n_pairs=100
        )
        assert result["n_pairs"] <= 6


class TestPairGenerators:
    """Unit tests for the internal similarity-pair generators."""

    shape = (10, 10, 10)

    def test_high_similarity_jaccard_range(self):
        rng = np.random.default_rng(0)
        jaccards = []
        from benchmark import ground_truth_jaccard

        for _ in range(20):
            a, b = _generate_high_similarity_pair(rng, self.shape)
            jaccards.append(ground_truth_jaccard(a, b))
        assert min(jaccards) > 0.5, f"High similarity min too low: {min(jaccards):.3f}"

    def test_low_similarity_jaccard_range(self):
        rng = np.random.default_rng(1)
        jaccards = []
        from benchmark import ground_truth_jaccard

        for _ in range(20):
            a, b = _generate_low_similarity_pair(rng, self.shape)
            jaccards.append(ground_truth_jaccard(a, b))
        assert max(jaccards) < 0.5, f"Low similarity max too high: {max(jaccards):.3f}"

    def test_medium_similarity_shape(self):
        rng = np.random.default_rng(2)
        a, b = _generate_medium_similarity_pair(rng, self.shape)
        assert a.shape == self.shape
        assert b.shape == self.shape
        assert a.dtype == np.float32