"""
spark - PySpark distributed hashing pipeline.

Modules:
    session  - SparkSession factory (_get_on_create_spark_) and SPARK_AVAILABLE flag
    serializable - SerializableKronHasher (pickle-safe Kronecker hasher for workers)
    pipeline - SparkTensorHasher (distributed LSH) and LocalTensorHashPipeline (multiprocessing)
"""

from .pipeline import LocalTensorHashPipeline, SparkTensorHasher
from .serializable import SerializableKronHasher

__all__ = [
    "SerializableKronHasher",
    "SparkTensorHasher",
    "LocalTensorHashPipeline",
]