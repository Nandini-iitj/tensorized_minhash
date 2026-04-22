"""
Tests for benchmarks/speed.py.

Covers benchmark_speed, benchmark_speed_real, and benchmark_random_projection
including both feasible and infeasible (too-large) matrix paths.
"""

from benchmarks import (
    benchmark_random_projection,
    benchmark_speed,
    benchmark_speed_real,
)
import numpy as np

def _make_tensors(shape=(10, 10, 10), n=5, seed=0):
    """Helper: generate small binary tensors for fast testing."""
    rng = np.random.default_rng(seed)
    return [(rng.random(shape) < 0.1).astype(np.float32) for _ in range(n)]

class TestBenchmarkSpeed:
    def test_result_keys_kron_tt(self):
        results = benchmark_speed(shape=(10, 10, 10), num_hashes=16, n_tensors=5)
        for method in ("Kron", "TT"):
            assert method in results
            for key in ("total_sec", "per_tensor_ms", "tensors_per_sec"):
                assert key in results[method], f"Missing key {key} in {method}"

    def test_datasketch_key_present(self):
        """Datasketch should be present (it is installed in this environment)."""
        results = benchmark_speed(shape=(10, 10, 10), num_hashes=16, n_tensors=5)
        assert "Datasketch" in results

    def test_per_tensor_ms_positive(self):
        results = benchmark_speed(shape=(10, 10, 10), num_hashes=16, n_tensors=5)
        for method in ("Kron", "TT"):
            assert results[method]["per_tensor_ms"] > 0

    def test_tensors_per_sec_consistent(self):
        results = benchmark_speed(shape=(10, 10, 10), num_hashes=16, n_tensors=5)
        kron = results["Kron"]
        expected = 1000.0 / kron["per_tensor_ms"]
        actual = kron["tensors_per_sec"]
        assert abs(expected - actual) / expected < 0.01  # within 1%

class TestBenchmarkSpeedReal:
    def test_uses_provided_tensors(self):
        tensors = _make_tensors(shape=(8, 8, 8), n=4)
        results = benchmark_speed_real(tensors, num_hashes=16)
        assert "Kron" in results
        assert "TT" in results

class TestBenchmarkRandomProjection:
    def test_feasible_small_tensor(self):
        """Small tensor -> matrix fits -> feasible=True with timing."""
        tensors = _make_tensors(shape=(5, 5, 5), n=4)
        result = benchmark_random_projection(tensors, num_hashes=16)
        assert result["feasible"] is True
        assert result["tensors_per_sec"] > 0
        assert result["per_tensor_ms"] > 0

    def test_feasible_result_keys(self):
        tensors = _make_tensors(shape=(5, 5, 5), n=4)
        result = benchmark_random_projection(tensors, num_hashes=16)
        for key in (
            "feasible",
            "theoretical_mb",
            "flat_dim",
            "num_hashes",
            "total_sec",
            "per_tensor_ms",
            "tensors_per_sec",
        ):
            assert key in result, f"Missing key: {key}"

    def test_infeasible_large_tensor(self):
        """
        A shape where flat_dim * num_hashes * 4 > 500 MB triggers infeasible path.
        flat_dim = 200^3 = 8M, 8M * 128 * 4 = 4 GB >> 500 MB.
        We mock large tensors with a very small list so we don't allocate them.
        """
        # We only need tensors[0].shape - we can use tiny actual tensors
        # but fake a huge shape by using a list with one element whose shape
        # reports as large. Use a simple array but rely on shape inspection.
        # Simplest: use a large shape that triggers the >500MB check.
        big_tensor = np.zeros((200, 200, 200), dtype=np.float32)
        result = benchmark_random_projection([big_tensor], num_hashes=128)
        assert result["feasible"] is False
        assert "note" in result
        assert result["theoretical_mb"] > 500
