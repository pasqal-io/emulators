from cuquantum import cutensornet as cutn
from typing import Any


class TensorDescriptor:
    def __init__(self, handle: Any, extents: list, modes: list, data_type: Any) -> None:
        self.descriptor = cutn.create_tensor_descriptor(
            handle, len(modes), extents, 0, modes, data_type
        )

    def __del__(self) -> None:
        cutn.destroy_tensor_descriptor(self.descriptor)
