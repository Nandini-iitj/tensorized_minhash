import numpy as np
import pandas as pd
from typing import Tuple

class NetworkTensorBuilder:
    """
    Transforms tabular network log DataFrames into 3D binary tensors.
    
    The tensor dimensions typically represent:
    (Source IP, Destination IP, Port/Protocol characteristics)
    """

    def __init__(self, shape: Tuple[int, int, int] = (50, 50, 50)):
        self.shape = shape

    def build_tensor(self, df: pd.DataFrame) -> np.ndarray:
        """
        Converts a DataFrame of network flows into a single 3D binary tensor.
        Uses hashing to map IP addresses and ports into the tensor grid.
        """
        tensor = np.zeros(self.shape, dtype=np.float32)
        
        if df.empty:
            return tensor

        # Map features to tensor indices using modulo hashing
        # Dimension 0: Source IPs
        src_indices = pd.util.hash_pandas_object(df['Source IP'], index=False) % self.shape[0]
        
        # Dimension 1: Destination IPs
        dst_indices = pd.util.hash_pandas_object(df['Destination IP'], index=False) % self.shape[1]
        
        # Dimension 2: Ports (Destination Port is a common pivot)
        port_indices = df['Destination Port'].astype(int) % self.shape[2]

        # Fill the tensor: 1.0 indicates the presence of a flow (binary)
        # We use a loop or advanced indexing to set the bits
        for s, d, p in zip(src_indices, dst_indices, port_indices):
            tensor[s, d, p] = 1.0
            
        return tensor

    def build_batch(self, df: pd.DataFrame, window_size: str = '1min') -> list:
        """
        Splits a large log DataFrame into time-windows and builds a tensor for each.
        """
        if 'Timestamp' not in df.columns:
            # If no timestamp, return as a single batch
            return [self.build_tensor(df)]
            
        df['Timestamp'] = pd.to_datetime(df['Timestamp'])
        df = df.sort_values('Timestamp')
        
        tensors = []
        for _, window_df in df.groupby(pd.Grouper(key='Timestamp', freq=window_size)):
            if not window_df.empty:
                tensors.append(self.build_tensor(window_df))
                
        return tensors