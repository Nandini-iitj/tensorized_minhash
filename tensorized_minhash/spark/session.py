"""
spark.session — SparkSession factory and Spark availability flag.

Reads cluster configuration from environment variables:
  SPARK_MASTER_URL    — Spark master URL (default: local[*] or local[N] via SLURM)
  SPARK_DRIVER_MEMORY  — driver heap (default: 4g)
  SPARK_EXECUTOR_MEMORY — executor heap (default: 4g)
  SLURM_CPUS_PER_TASK  — SLURM core count, used for local[N] mode
"""

import logging
import os

__all__ = ["SPARK_AVAILABLE", "_get_or_create_spark"]

logger = logging.getLogger(__name__)

try:
    from pyspark.sql import SparkSession

    SPARK_AVAILABLE = True
except ImportError:
    SPARK_AVAILABLE = False
    logger.warning(
        "PySpark not installed. SparkTensorHasher will raise if used. "
        "Install with: pip install pyspark"
    )


def _get_or_create_spark(app_name: str = "TensorizedMinHash") -> "SparkSession":
    """
    Create a Spark session for local or cluster use.
    Override via env vars: SPARK_MASTER_URL, SPARK_DRIVER_MEMORY,
    SPARK_EXECUTOR_MEMORY, SLURM_CPUS_PER_TASK.
    """
    if not SPARK_AVAILABLE:
        raise ImportError("PySpark is required. pip install pyspark")

    master = os.environ.get("SPARK_MASTER_URL", None)
    if master is None:
        num_cores = os.environ.get("SLURM_CPUS_PER_TASK", "*")
        master = f"local[{num_cores}]"

    driver_mem = os.environ.get("SPARK_DRIVER_MEMORY", "4g")
    executor_mem = os.environ.get("SPARK_EXECUTOR_MEMORY", "4g")

    spark = (
        SparkSession.builder.appName(app_name)
        .master(master)
        .config("spark.driver.memory", driver_mem)
        .config("spark.executor.memory", executor_mem)
        .config("spark.serializer", "org.apache.spark.serializer.KryoSerializer")
        .config("spark.ui.showConsoleProgress", "false")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("WARN")
    return spark
