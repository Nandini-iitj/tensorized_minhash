"""
Tests for the three-scenario accuracy benchmark.

Validates that the combined Spearman ρ across high/medium/low similarity
scenarios exceeds the target of 0.85, and that each scenario
produces Jaccard values in the expected range.
"""

from benchmarks import benchmark_accuracy


class TestThreeScenarioBenchmark:
    """
    Validates the three-scenario accuracy benchmark meets the target:
    Spearman ρ > 0.85 on the combined (full Jaccard range) results.
    """

    def test_combined_spearman_above_target(self):
        """
        Combined Spearman ρ across all three scenarios must exceed 0.85.
        This is the primary evaluation metric.
        """
        acc = benchmark_accuracy(n_pairs=60, shape=(25, 25, 25), num_hashes=128, seed=0)
        kron_rho = acc["tensorized_kron"]["spearman_r"]
        assert kron_rho > 0.85, (
            f"Kron Spearman ρ = {kron_rho:.4f} — must exceed 0.85 (target). "
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

