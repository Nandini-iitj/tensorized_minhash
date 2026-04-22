"""
Benchmarks for Memory, accuracy, speed, and RAM benchmarks.

Modules:
    helpers = ground_truth_jaccard, datasketch_jaccard
    memory = benchmark_memory, benchmark_ram
    accuracy = benchmark_accuracy, benchmark_accuracy_from_tensors
    speed = benchmark_speed, benchmark_speed_real, benchmark_random_projection
"""

from .accuracy import benchmark_accuracy, benchmark_accuracy_from_tensors
from .helpers import datasketch_jaccard, ground_truth_jaccard
from .memory import benchmark_memory, benchmark_ram
from .speed import benchmark_random_projection, benchmark_speed, benchmark_speed_real

__all__ = [
    "benchmark_memory",
    "benchmark_accuracy",
    "benchmark_accuracy_from_tensors",
    "benchmark_speed",
    "benchmark_speed_real",
    "benchmark_ram",
    "benchmark_random_projection",
    "ground_truth_jaccard",
    "datasketch_jaccard",
]