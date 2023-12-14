from cuquantum import cutensornet as cutn
import cupy as cp
import cuquantum as cq


class ConfigPrivate:
    def __init__(self) -> None:
        self.data_type = cq.cudaDataType.CUDA_C_64F
        self.compute_type = cq.ComputeType.COMPUTE_64F
        self.ct_handle = cutn.create()
        self.work_desc = cutn.create_workspace_descriptor(self.ct_handle)
        self.work = None
        self.stream = cp.cuda.Stream()
        # have cupy use default mempool with async allocations
        cp.cuda.set_allocator(cp.cuda.MemoryAsyncPool(pool_handles="default").malloc)
        # Now we configure cutensornet to also use the default mempool with async allocations
        # This way, we don't have to manually allocate workspaces for everything
        # bind mallocAsync and freeAsync to cutensornet
        cutn.set_device_mem_handler(
            self.ct_handle,
            [
                cp.cuda.runtime.mallocAsync,
                lambda ptr, size, str: cp.cuda.runtime.freeAsync(ptr, str),
                "default",
            ],
        )
        # tell cutensornet to use the bound memory management
        cutn.workspace_set_memory(
            self.ct_handle,
            self.work_desc,
            cutn.Memspace.DEVICE,
            cutn.WorkspaceKind.SCRATCH,
            0,
            0,
        )
        cutn.workspace_set_memory(
            self.ct_handle,
            self.work_desc,
            cutn.Memspace.DEVICE,
            cutn.WorkspaceKind.CACHE,
            0,
            0,
        )

    def __del__(self) -> None:
        """Free all resources owned by the object."""
        cutn.destroy(self.ct_handle)
        cutn.destroy_workspace_descriptor(self.work_desc)

    def set_gate_algorithm(self, gate_algo: cutn.GateSplitAlgo) -> None:
        """Set the algorithm to use for all gate split operations.

        Args:
            gate_algo (cuquantum.cutensornet.GateSplitAlgo): The gate splitting algorithm to use.
        """

        self.gate_algo = gate_algo


g_config = ConfigPrivate()


class Config:
    def __init__(self) -> None:
        pass

    def set_compute_type(self, compute_type: cq.ComputeType) -> None:
        g_config.compute_type = compute_type

    def set_data_type(self, data_type: cq.cudaDataType) -> None:
        g_config.data_type = data_type
