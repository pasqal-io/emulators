import cuquantum as cq
from cuquantum import cutensornet as cutn
from .config import g_config

def contract(expression, *operands):
    return cq.contract(expression, *operands, options=cq.NetworkOptions(handle=g_config.ct_handle))

def svd(expression, operand):
    return cutn.tensor.decompose(expression, operand, options={"handle":g_config.ct_handle}, method=cutn.tensor.SVDMethod())

def qr(expression, operand):
    return cutn.tensor.decompose(expression, operand, options={"handle":g_config.ct_handle})