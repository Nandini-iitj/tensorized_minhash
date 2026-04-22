"""
Tests for spark/serializable.py, spark/session.py, and spark/pipeline.py.

Only LocalTensorHashPipeline and SerializableKronHasher are tested here -
SparkTensorHasher is excluded because it requires a running PySpark cluster.
"""

from core.config import TTMinHashConfig
from core.kron_minhash import KroneckerMinHash
import numpy as np
from spark.pipeline import LocalTensorHashPipeline
from spark.serializable import SerializableKronHasher
from spark.session import SPARK_AVAILABLE

def _kron_hasher(shape=(10, 10, 10), num_hashes=16):
    cfg = TTMinHashConfig(shape=shape, num_hashes=num_hashes, seed=0)
    return KroneckerMinHash(cfg)

def _tensors(shape=(10, 10, 10), n=6, seed=3):
    rng = np.random.default_rng(seed)
    return [(rng.random(shape) < 0.1).astype(np.float32) for _ in range(n)]

# --------------------------------------------------------------------------
# SerializableKronHasher
# --------------------------------------------------------------------------

class TestSerializableKronHasher:
    def setup_method(self):
        self.hasher = _kron_hasher()
        self.ser = SerializableKronHasher.from_kron_hasher(self.hasher)

    def test_from_kron_hasher_copies_factors(self):
        assert len(self.ser.factors) == len(self.hasher.factors)
        for f_orig, f_copy in zip(self.hasher.factors, self.ser.factors):
            # Ensure it is a copy, not a view
            np.testing.assert_array_equal(f_orig, f_copy)
            
    def test_factors_are_independent(self):
        # Modify copy to check if independent
        orig_val = self.hasher.factors[0][0, 0]
        self.ser.factors[0][0, 0] += 999.0
        assert self.hasher.factors[0][0, 0] == orig_val
        assert self.ser.factors[0][0, 0] != orig_val

    def test_hash_tensor_shape(self):
        t = (np.random.default_rng(0).random((10, 10, 10)) < 0.1).astype(np.float32)
        sig = self.ser.hash_tensor(t)
        assert sig.shape == (self.hasher.cfg.num_hashes,)
        assert sig.dtype == np.int32

    def test_hash_tensor_empty(self):
        t = np.zeros((10, 10, 10), dtype=np.float32)
        sig = self.ser.hash_tensor(t)
        np.testing.assert_array_equal(sig, 0)

    def test_matches_original_hasher(self):
        """SerializableKronHasher must produce identical results to KroneckerMinHash."""
        t = (np.random.default_rng(7).random((10, 10, 10)) < 0.1).astype(np.float32)
        sig_orig = self.hasher.hash_tensor(t)
        sig_ser = self.ser.hash_tensor(t)
        np.testing.assert_array_equal(sig_orig, sig_ser)

# --------------------------------------------------------------------------
# spark.session
# --------------------------------------------------------------------------

class TestSparkSession:
    def test_spark_available_is_bool(self):
        assert isinstance(SPARK_AVAILABLE, bool)

# --------------------------------------------------------------------------
# LocalTensorHashPipeline
# --------------------------------------------------------------------------

class TestLocalTensorHashPipeline:
    def setup_method(self):
        self.shape = (10, 10, 10)
        self.hasher = _kron_hasher(self.shape)
        self.pipeline = LocalTensorHashPipeline(self.hasher)
        self.tensors = _tensors(self.shape, n=6)
        self.ids = [f"t{i}" for i in range(6)]

    def test_hash_all_serial(self):
        results = self.pipeline.hash_all(self.tensors, self.ids, parallel=False)
        assert len(results) == 6
        for _doc_id, sig in results:
            assert sig.shape == (self.hasher.cfg.num_hashes,)

    def test_hash_all_auto_ids(self):
        """When ids=None, integer string IDs are assigned automatically."""
        results = self.pipeline.hash_all(self.tensors, parallel=False)
        assert [doc_id for doc_id, _ in results] == [str(i) for i in range(6)]

    def test_hash_all_parallel(self):
        results = self.pipeline.hash_all(self.tensors, self.ids, parallel=True)
        assert len(results) == 6

    def test_find_similar_pairs_threshold(self):
        id_sigs = self.pipeline.hash_all(self.tensors, self.ids, parallel=False)
        # With threshold=0 every pair should be returned (Jaccard >= 0 always)
        pairs = self.pipeline.find_similar_pairs(id_sigs, threshold=0.0)
        expected_n = len(self.tensors) * (len(self.tensors) - 1) // 2
        assert len(pairs) == expected_n

    def test_find_similar_pairs_high_threshold(self):
        """With threshold=1.0 only identical-signature pairs qualify."""
        # Use identical tensors so at least one pair matches
        t = self.tensors[0]
        ids = ["a", "b"]
        sigs = self.pipeline.hash_all([t, t], ids, parallel=False)
        pairs = self.pipeline.find_similar_pairs(sigs, threshold=1.0)
        assert len(pairs) == 1
        assert pairs[0][2] == 1.0

    def test_pairs_sorted_descending(self):
        id_sigs = self.pipeline.hash_all(self.tensors, self.ids, parallel=False)
        pairs = self.pipeline.find_similar_pairs(id_sigs, threshold=0.0)
        scores = [p[2] for p in pairs]
        assert scores == sorted(scores, reverse=True)
