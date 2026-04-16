"""
PySpark distributed MinHash pipeline.

Design:
  - Each Spark worker holds a COPY of the (small) Kronecker factor matrices.
    Because these are O(n*k) = e.g. 400 parameters, broadcasting is trivial.
  - The tensor dataset is an RDD of (id, tensor_bytes) pairs.
  - map() applies hash_tensor on each worker without any shuffle.
  - Jaccard estimation is a reduce/join step, not a full cross-product.

Scalability argument:
  Standard MinHash on 100^3 tensors: each hash function needs a 100^3=1M
  param vector → 128 hash functions = 128M floats = 512 MB just for params.
  With Kronecker: 3 * 100 * 128 = 38,400 params = 150 KB. This fits in the
  Spark broadcast variable (recommended < 10 MB). Workers never OOM from
  hash-function parameters regardless of cluster size.
"""
import os
import numpy as np
import pickle
import logging
from typing import List, Tuple, Optional, Iterator

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Attempt PySpark import; fall back gracefully for environments without Spark
# ---------------------------------------------------------------------------
try:
    from pyspark.sql import SparkSession
    from pyspark.broadcast import Broadcast
    import pyspark.sql.functions as F
    SPARK_AVAILABLE = True
except ImportError:
    SPARK_AVAILABLE = False
    logger.warning(
        "PySpark not installed. SparkTensorHasher will raise if used. "
        "Install with: pip install pyspark"
    )


def _get_or_create_spark(app_name: str = "TensorizedMinHash") -> "SparkSession":
    """Create a local Spark session suitable for single-machine prototyping."""
    if not SPARK_AVAILABLE:
        raise ImportError("PySpark is required. pip install pyspark")

    #spark = (
    #    SparkSession.builder
    #    .appName(app_name)
    #    .master("local[*]")
    #    .config("spark.driver.memory", "4g")
    #    .config("spark.executor.memory", "4g")
        # Reduce log noise for demos
    #    .config("spark.ui.showConsoleProgress", "false")
    #    .getOrCreate()
    #)
    num_cores = os.environ.get("SLURM_CPUS_PER_TASK", "4")
    spark = (
        SparkSession.builder
        .appName(app_name)
        .master(f"local[{num_cores}]")
        .config("spark.driver.memory", "4g")
        .config("spark.executor.memory", "24g")
        # Reduce log noise for demos
        .config("spark.ui.showConsoleProgress", "false")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("WARN")
    return spark

def _get_or_create_spark1(app_name: str = "TensorizedMinHash") -> "SparkSession":
    """Create a Spark session — uses SPARK_MASTER env var if set, else local[*]."""
    if not SPARK_AVAILABLE:
        raise ImportError("PySpark is required. pip install pyspark")

    master = os.environ.get("SPARK_MASTER_URL", "local[*]")
    driver_mem = os.environ.get("SPARK_DRIVER_MEMORY", "4g")
    executor_mem = os.environ.get("SPARK_EXECUTOR_MEMORY", "4g")

    spark = (
        SparkSession.builder
        .appName(app_name)
        .master(master)
        .config("spark.driver.memory", driver_mem)
        .config("spark.executor.memory", executor_mem)
        .config("spark.serializer", "org.apache.spark.serializer.KryoSerializer")
        .config("spark.ui.showConsoleProgress", "false")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("WARN")
    return spark


# ---------------------------------------------------------------------------
# Serialisable hasher wrapper (must be pickle-friendly for Spark)
# ---------------------------------------------------------------------------

class SerializableKronHasher:
    """
    Lightweight wrapper around KroneckerMinHash that is safely picklable
    for distribution across Spark workers.

    We store only the factor matrices (numpy arrays) and config primitives —
    no Python closures or unpicklable objects.
    """

    def __init__(self, factors: List[np.ndarray], shape: Tuple[int, ...], num_hashes: int):
        self.factors = factors
        self.shape = shape
        self.num_hashes = num_hashes

    @classmethod
    def from_kron_hasher(cls, hasher) -> "SerializableKronHasher":
        from core.tt_minhash import KroneckerMinHash
        return cls(
            factors=[f.copy() for f in hasher.factors],
            shape=hasher.cfg.shape,
            num_hashes=hasher.cfg.num_hashes,
        )

    def hash_tensor(self, tensor: np.ndarray) -> np.ndarray:
        """Compute MinHash signature using Kronecker additive-exponential argmin."""
        nonzero_idx = np.argwhere(tensor > 0)
        if len(nonzero_idx) == 0:
            return np.zeros(self.num_hashes, dtype=np.int32)

        signature = np.empty(self.num_hashes, dtype=np.int32)
        for j in range(self.num_hashes):
            # Additive rank: sum of -log(Uniform) per mode
            ranks = np.zeros(len(nonzero_idx), dtype=np.float64)
            for mode in range(len(self.shape)):
                mode_indices = nonzero_idx[:, mode]
                ranks += self.factors[mode][j, mode_indices]
            min_cell_flat = np.argmin(ranks)
            multi_idx = nonzero_idx[min_cell_flat]
            cell_id = int(np.ravel_multi_index(multi_idx, self.shape))
            signature[j] = cell_id
        return signature


# ---------------------------------------------------------------------------
# Spark pipeline
# ---------------------------------------------------------------------------

class SparkTensorHasher:
    """
    Distributes tensorized MinHash across a Spark cluster.

    Usage:
        hasher = SparkTensorHasher(kron_hasher, spark=spark)
        signatures_rdd = hasher.hash_rdd(tensor_rdd)   # (id, sig) pairs
        similar_pairs  = hasher.find_similar_pairs(signatures_rdd, threshold=0.7)
    """

    def __init__(
        self,
        kron_hasher,           # KroneckerMinHash instance
        spark=None,
        similarity_bands: int = 16,  # LSH band count (num_hashes must be divisible)
    ):
        if not SPARK_AVAILABLE:
            raise ImportError("PySpark is required.")

        self.spark = spark or _get_or_create_spark()
        self.sc = self.spark.sparkContext
        self.bands = similarity_bands

        # Serialise and broadcast the (tiny) hash parameters
        serializable = SerializableKronHasher.from_kron_hasher(kron_hasher)
        self.bc_hasher: Broadcast = self.sc.broadcast(serializable)

        num_hashes = kron_hasher.cfg.num_hashes
        assert num_hashes % similarity_bands == 0, (
            f"num_hashes={num_hashes} must be divisible by similarity_bands={similarity_bands}"
        )
        self.rows_per_band = num_hashes // similarity_bands

        param_bytes = sum(f.nbytes for f in kron_hasher.factors)
        logger.info(
            f"Broadcast {param_bytes} bytes of hash parameters "
            f"({param_bytes / 1024:.1f} KB) across cluster"
        )

    def hash_rdd(self, tensor_rdd) -> "pyspark.RDD":
        """
        Map over an RDD of (doc_id: str, tensor: np.ndarray) pairs.
        Returns RDD of (doc_id, signature: np.ndarray).

        The lambda captures bc_hasher by value — Spark handles serialisation.
        """
        bc = self.bc_hasher

        def hash_one(record):
            doc_id, tensor = record
            hasher = bc.value
            sig = hasher.hash_tensor(tensor)
            return (doc_id, sig)

        return tensor_rdd.map(hash_one)

    def find_similar_pairs(
        self,
        sig_rdd,
        threshold: float = 0.7,
    ) -> "pyspark.RDD":
        """
        LSH banding: group signatures into bands, find documents sharing a band.
        Pairs sharing ≥1 band are candidate duplicates; we then verify with
        exact Jaccard estimate.

        Returns RDD of (id_a, id_b, jaccard_estimate) where estimate >= threshold.
        """
        bands = self.bands
        rows = self.rows_per_band

        def emit_band_keys(record):
            doc_id, sig = record
            for b in range(bands):
                band_sig = sig[b * rows : (b + 1) * rows]
                band_key = (b, band_sig.tobytes())
                yield (band_key, doc_id)

        # Group by band key → candidate pairs
        band_rdd = sig_rdd.flatMap(emit_band_keys).groupByKey()

        def make_pairs(record):
            _key, doc_ids = record
            doc_ids = list(doc_ids)
            for i in range(len(doc_ids)):
                for j in range(i + 1, len(doc_ids)):
                    yield (doc_ids[i], doc_ids[j])

        candidate_pairs = band_rdd.flatMap(make_pairs).distinct()

        # Join back signatures for exact Jaccard estimate
        sig_lookup = sig_rdd.collectAsMap()  # fits in driver for prototype
        bc_sigs = self.sc.broadcast(sig_lookup)

        def estimate_jaccard(pair):
            a, b = pair
            sigs = bc_sigs.value
            if a not in sigs or b not in sigs:
                return None
            j = float(np.mean(sigs[a] == sigs[b]))
            return (a, b, j)

        result = (
            candidate_pairs
            .map(estimate_jaccard)
            .filter(lambda x: x is not None and x[2] >= threshold)
        )
        return result

    @staticmethod
    def tensors_to_rdd(tensors: List[np.ndarray], sc, ids: Optional[List[str]] = None):
        """Helper: convert a local list of tensors to a Spark RDD."""
        if ids is None:
            ids = [str(i) for i in range(len(tensors))]
        records = list(zip(ids, tensors))
        return sc.parallelize(records, numSlices=min(len(records), 100))


# ---------------------------------------------------------------------------
# Pure-Python fallback (no Spark) for local prototyping
# ---------------------------------------------------------------------------

class LocalTensorHashPipeline:
    """
    Drop-in replacement for SparkTensorHasher when Spark is unavailable.
    Uses multiprocessing instead of Spark workers for local parallelism.
    """

    def __init__(self, kron_hasher):
        self.hasher = kron_hasher

    def hash_all(
        self,
        tensors: List[np.ndarray],
        ids: Optional[List[str]] = None,
        parallel: bool = True,
    ) -> List[Tuple[str, np.ndarray]]:
        if ids is None:
            ids = [str(i) for i in range(len(tensors))]

        if parallel:
            from multiprocessing import Pool
            import os
            n_workers = min(os.cpu_count() or 1, len(tensors))
            with Pool(n_workers) as pool:
                sigs = pool.map(self.hasher.hash_tensor, tensors)
        else:
            sigs = [self.hasher.hash_tensor(t) for t in tensors]

        return list(zip(ids, sigs))

    def find_similar_pairs(
        self,
        id_sig_pairs: List[Tuple[str, np.ndarray]],
        threshold: float = 0.7,
    ) -> List[Tuple[str, str, float]]:
        results = []
        n = len(id_sig_pairs)
        for i in range(n):
            for j in range(i + 1, n):
                id_a, sig_a = id_sig_pairs[i]
                id_b, sig_b = id_sig_pairs[j]
                j_est = float(np.mean(sig_a == sig_b))
                if j_est >= threshold:
                    results.append((id_a, id_b, j_est))
        return sorted(results, key=lambda x: -x[2])
