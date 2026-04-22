"""
Shared pytest configuration for TensorizedMinHash tests.

Adds the package root to sys.path so all test modules can import
from core, data, benchmarks, and spark without installation.
"""

import os
import sys

# Ensure the tensorized_minhash/ package directory is on the path
_REPO_ROOT = os.path.dirname(os.path.dirname(__file__))
sys.path.insert(0, os.path.join(_REPO_ROOT, "tensorized_minhash"))