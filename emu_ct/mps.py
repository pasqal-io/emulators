import itertools
from .operations import contract, qr, svd
import cupy as cp
import numpy as np
from typing import Union, List

class MPS:    
    """
    Matrix Product State    
    """

    def __init__(self, sites:Union[int, List], max_virtual_extent):
        if isinstance(sites, int):
            self.num_sites = sites
            self.factors = []
            for i in range(self.num_sites):
                tensor = cp.zeros((1,2,1), dtype=np.complex128, order="F")
                tensor[0,0,0] = 1.0
                self.factors.append(tensor)
        elif isinstance(sites, List):
            self.factors = sites
            self.num_sites = len(sites)
            self.truncate()
        else:
            raise NotImplementedError()
        assert self.num_sites > 1 #otherwise, do state vector
                
        untruncated_max_extent = 2 ** (self.num_sites // 2)
        if max_virtual_extent == 0:
            self.max_virtual_extent = untruncated_max_extent
        else:
            self.max_virtual_extent = min(max_virtual_extent, untruncated_max_extent)
        self.orth_center = 0

    
    def __repr__(self) -> str:
        result = "["
        for fac in self.factors:
            result += repr(fac)
            result += ", "
        result +="]"
        return result
    
    def truncate(self):
        for i in range(self.num_sites - 1):
            q,r = qr("ijk->ijl,lk", self.factors[i])
            self.factors[i] = q
            self.factors[i+1] = contract("ij, jkl->ikl",r,self.factors[i+1])
        for i in range(self.num_sites -1, 0, -1):
            u,s,vh = svd("ijk->il,ljk", self.factors[i])
            self.factors[i] = vh
            factor = contract("ij, j->ij", u, s.astype(dtype=np.complex128))
            factor = contract("ijk, kl->ijl",self.factors[i-1], factor)
            self.factors[i-1] = factor