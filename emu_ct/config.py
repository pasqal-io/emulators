from cuquantum import cutensornet as cutn
import cupy as cp
import numpy as np
import cuquantum as cq

class ConfigPrivate:
    def __init__(self):
        self.data_type =    cq.cudaDataType.CUDA_C_64F
        self.compute_type = cq.ComputeType.COMPUTE_64F
        self.ct_handle = cutn.create()
        self.work_desc = cutn.create_workspace_descriptor(self.ct_handle)
        self.work = None
        self.stream = cp.cuda.Stream()
        #have cupy use default mempool with async allocations
        cp.cuda.set_allocator(cp.cuda.MemoryAsyncPool(pool_handles='default').malloc)
        #Now we configure cutensornet to also use the default mempool with async allocations
        #This way, we don't have to manually allocate workspaces for everything
        #bind mallocAsync and freeAsync to cutensornet
        cutn.set_device_mem_handler(self.ct_handle, [cp.cuda.runtime.mallocAsync, lambda ptr,size,str: cp.cuda.runtime.freeAsync(ptr, str),"default"])
        #tell cutensornet to use the bound memory management
        cutn.workspace_set_memory(self.ct_handle, self.work_desc, cutn.Memspace.DEVICE, cutn.WorkspaceKind.SCRATCH, 0, 0)
        cutn.workspace_set_memory(self.ct_handle, self.work_desc, cutn.Memspace.DEVICE, cutn.WorkspaceKind.CACHE, 0, 0)

    def __del__(self):
        """Free all resources owned by the object."""
        cutn.destroy(self.ct_handle)
        cutn.destroy_workspace_descriptor(self.work_desc)

    def set_gate_algorithm(self, gate_algo):    
        """Set the algorithm to use for all gate split operations.
        
        Args:
            gate_algo (cuquantum.cutensornet.GateSplitAlgo): The gate splitting algorithm to use.
        """

        self.gate_algo = gate_algo

    def set_svd_config(self, abs_cutoff, rel_cutoff, renorm, partition):
        """Update the SVD truncation setting.
        
        Args:
            abs_cutoff: The cutoff value for absolute singular value truncation.
            rel_cutoff: The cutoff value for relative singular value truncation.
            renorm (cuquantum.cutensornet.TensorSVDNormalization): The option for renormalization of the truncated singular values.
            partition (cuquantum.cutensornet.TensorSVDPartition): The option for partitioning of the singular values.
        """        
        
        if partition != cutn.TensorSVDPartition.UV_EQUAL:
            raise NotImplementedError("this basic example expects partition to be cutensornet.TensorSVDPartition.UV_EQUAL")

        svd_config_attributes = [cutn.TensorSVDConfigAttribute.ABS_CUTOFF, 
                                 cutn.TensorSVDConfigAttribute.REL_CUTOFF, 
                                 cutn.TensorSVDConfigAttribute.S_NORMALIZATION,
                                 cutn.TensorSVDConfigAttribute.S_PARTITION]
            
        for (attr, value) in zip(svd_config_attributes, [abs_cutoff, rel_cutoff, renorm, partition]):
            dtype = cutn.tensor_svd_config_get_attribute_dtype(attr)
            value = np.array([value], dtype=dtype)
            cutn.tensor_svd_config_set_attribute(self.ct_handle, 
                self.svd_config, attr, value.ctypes.data, value.dtype.itemsize)
            
g_config = ConfigPrivate()

class Config:
    def __init__(self):
        pass
    
    def set_compute_type(self, compute_type):
        g_config.compute_type = compute_type

    def set_data_type(self, data_type):
        g_config.data_type = data_type