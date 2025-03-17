import math
import torch

def _index_to_bitstring(vector:torch.Tensor, index: int) -> str:
        """
        Convert an integer index into its corresponding bitstring representation.
        """
        nqubits = int(math.log2(vector.reshape(-1).shape[0]))
        return format(index, f"0{nqubits}b")

