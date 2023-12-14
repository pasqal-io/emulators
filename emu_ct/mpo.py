from mps import MPS
from operations import contract
import cupy as cp

class MPO:    
    """
    Matrix Product Operator    
    """

    def __init__(self, factors, max_virtual_extent):
        self.factors = factors
        self.num_sites = len(factors)
        assert self.num_sites > 1 #otherwise, do state vector
        
        untruncated_max_extent = 2 ** (self.num_sites // 2)
        if max_virtual_extent == 0:
            self.max_virtual_extent = untruncated_max_extent
        else:
            self.max_virtual_extent = min(max_virtual_extent, untruncated_max_extent)
        
    def __repr__(self) -> str:
        result = "["
        for fac in self.factors:
            result += repr(fac)
            result += ", "
        result +="]"
        return result
    
    def __mul__(self, state:MPS):
        out_factors = []
        #to keep track of the bond dimensions of the output mps
        #so we can do the logic in a single sweep from either left-to-right or vice-versa
        left_bond_size = 1
        if state.orth_center == 0:
            for i in range(self.num_sites):
                factor, left_bond_size = self._contract_factors(self.factors[i], state.factors[i], left_bond_size)
                out_factors.append(factor)
            mps = MPS(out_factors, state.max_virtual_extent)
            mps.truncate()
            return mps
        elif state.orth_center == self.num_sites - 1:
            raise NotImplementedError()
        else:
            raise NotImplementedError()
    
    @staticmethod
    def _contract_factors(mpo_factor, mps_factor, left_bond_size):
        factor = contract("ijkl,mko->imjlo", mpo_factor, mps_factor)
        factor = cp.reshape(factor, (left_bond_size, 2, -1))
        left_bond_size = cp.shape(factor)[-1]
        return factor, left_bond_size

    def __rmul__(self, state:MPS):
        self*state