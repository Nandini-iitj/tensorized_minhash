import numpy as np
from core.config import TTMinHashConfig

class KroneckerMinHash:
    """
    Kronecker-factored Additive-Exponential MinHash.
    
    Replaces a massive (D, L) random projection matrix with d small 
    factor matrices (n_i, L). This reduces storage from O(D*L) to O(sum(n_i)*L).
    """

    def __init__(self, config: TTMinHashConfig):
        self.config = config
        self.shape = config.shape
        self.num_hashes = config.num_hashes
        self.seed = config.seed
        
        # Initialize factors analytically using the provided seed
        rng = np.random.default_rng(self.seed)
        self.factors = [
            rng.standard_normal((s, self.num_hashes))
            for s in self.shape
        ]

    def hash_tensor(self, tensor: np.ndarray) -> np.ndarray:
        """
        Compute MinHash fingerprint via Kronecker-factored projection.
        Approximates the projection of the flattened tensor onto a 
        full random matrix.
        """
        if tensor.shape != self.shape:
            raise ValueError(f"Expected shape {self.shape}, got {tensor.shape}")

        # Initial projection using the first factor
        # We work through the dimensions to avoid forming the full Kronecker product
        res = None
        
        # The logic below performs the "Kronecker Product Property" for 
        # matrix-vector multiplication: (A ⊗ B)vec(X) = vec(B X A^T)
        # Optimized for d-order tensors:
        current_view = tensor.astype(np.float32)
        
        for i, factor in enumerate(self.factors):
            # Roll the axis to be processed to the front
            current_view = np.moveaxis(current_view, i, 0)
            s_i = self.shape[i]
            
            # Reshape to (s_i, -1) to multiply against the factor
            remaining_shape = current_view.shape[1:]
            flat_view = current_view.reshape(s_i, -1)
            
            # Factored projection step
            if res is None:
                res = factor.T @ flat_view
            else:
                # Interleave hashes with the existing projection
                res = res.reshape(self.num_hashes, -1)
                # This approximates the additive-exponential property
                res = factor.T @ flat_view + res 
            
            # Move axis back (optional depending on implementation style, 
            # but usually required for multi-factor consistency)
            current_view = current_view.reshape(s_i, *remaining_shape)
            current_view = np.moveaxis(current_view, 0, i)

        # Final fingerprint is the argmin over the projected values 
        # (MinHash characteristic)
        return np.argmin(res.reshape(self.num_hashes, -1), axis=1)

    def memory_stats(self) -> dict:
        """Report compression metrics compared to a full matrix."""
        full_params = int(np.prod(self.shape)) * self.num_hashes
        kron_params = sum(s * self.num_hashes for s in self.shape)
        
        return {
            "full_params": full_params,
            "kron_params": kron_params,
            "compression_ratio": full_params / max(kron_params, 1)
        }