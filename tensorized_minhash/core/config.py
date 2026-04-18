# "TTMinHashConfig" - shared configuration for Kronecker and TT-MinHash hashers.

import logging

import numpy as np

__all__ = ["TTMinHashConfig"]

logger = logging.getLogger(__name__)


class TTMinHashConfig:
    """Configuration for the TT-MinHash hasher."""

    def __init__(
        self,
        shape: tuple[int, ...],
        num_hashes: int = 128,
        tt_rank: int = 4,
        seed: int = 42,
    ):
        self.shape = shape
        self.ndim = len(shape)
        self.num_hashes = num_hashes
        self.tt_rank = tt_rank
        self.seed = seed

        # Theoretical parameter counts (for reference / logging)
        full_params = int(np.prod(shape)) * num_hashes
        kron_params = sum(s * num_hashes for s in shape)
        # TT_params: num_hashes independent decompositions with boundary ranks 1
        r = tt_rank
        ndim = len(shape)
        tt_params = num_hashes * sum(
            (1 if k == 0 else r) * shape[k] * (1 if k == ndim - 1 else r)
            for k in range(ndim)
        )

        self.kron_compression_ratio = full_params / max(kron_params, 1)
        self.tt_compression_ratio = full_params / max(tt_params, 1)

        logger.info(
            f"TTMinHashConfig: shape={shape}, k={num_hashes}, "
            f"full={full_params}, kron={kron_params}, "
            f"(kron: {self.kron_compression_ratio:.1f}x), "
            f"tt={tt_params}, (tt: {self.tt_compression_ratio:.1f}x)"
        )