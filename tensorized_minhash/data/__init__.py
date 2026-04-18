"""
data — Network-log data pipeline.

Modules:
    generator — NetworkLogGenerator (synthetic CIC-IDS2017-style log generator)
    builder   — NetworkTensorBuilder (DataFrame -> 3D binary tensor converter)
"""

from .builder import NetworkTensorBuilder
from .generator import NetworkLogGenerator

__all__ = ["NetworkLogGenerator", "NetworkTensorBuilder"]