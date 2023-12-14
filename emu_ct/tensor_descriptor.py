from cuquantum import cutensornet as cutn

class TensorDescriptor:
    def __init__(self, handle, extents, modes, data_type):
        self.descriptor = cutn.create_tensor_descriptor(handle, len(modes), extents, 0, modes, data_type)
    
    def __del__(self):
        cutn.destroy_tensor_descriptor(self.descriptor)