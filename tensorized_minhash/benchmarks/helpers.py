"""
benchmarks.helpers — Shared helpers for all benchmark modules.

Provides exact Jaccard (ground truth) and the datasketch standard MinHash
baseline used across accuracy and speed benchmarks.
"""

import logging
import numpy as np

__all__ = ["ground_truth_jaccard", "datasketch_jaccard"]

logger = logging.getLogger(__name__)

def ground_truth_jaccard(tensor_a: np.ndarray, tensor_b: np.ndarray) -> float:
    """Exact Jaccard on binary tensors. J(A,B) = |A∩B| / |A∪B|"""
    a = tensor_a.astype(bool).ravel()
    b = tensor_b.astype(bool).ravel()
    intersection = np.sum(a & b)
    union = np.sum(a | b)
    return float(intersection / union) if union > 0 else 0.0

def datasketch_jaccard(
    tensor_a: np.ndarray,
    tensor_b: np.ndarray,
    num_perm: int = 128,
) -> float:
    """
    Standard MinHash Jaccard using datasketch (flattened tensor).
    Falls back to exact Jaccard if datasketch is not installed.
    """
    try:
        from datasketch import MinHash
        
        m1 = MinHash(num_perm=num_perm)
        m2 = MinHash(num_perm=num_perm)
        
        for idx in np.where(tensor_a.ravel() > 0)[0]:
            m1.update(int(idx).to_bytes(8, byteorder="big"))
            
        for idx in np.where(tensor_b.ravel() > 0)[0]:
            m2.update(int(idx).to_bytes(8, byteorder="big"))
            
        return m1.jaccard(m2)
        
    except ImportError:
        logger.warning("datasketch not installed; falling back to exact Jaccard for baseline")
        return ground_truth_jaccard(tensor_a, tensor_b)