from unittest.mock import MagicMock
import torch
from emu_sv import SVConfig
from emu_base.pulser_adapter import PulserData
from emu_sv.sv_backend_impl import SVBackendImpl
from pulser import NoiseModel

device = "cpu"


def test_sv_impl():
    """test that index_add is called in a no_grad context in forward"""
    config = SVConfig(gpu=False if device == "cpu" else True)
    pulser_data = MagicMock(
        spec=PulserData,
        omega=torch.tensor([[1.0]], requires_grad=True),
        delta=torch.tensor([[1.0]]),
        phi=torch.tensor([[1.0]]),
        full_interaction_matrix=torch.tensor(0.0),
        target_times=[1.0],
        noise_model=NoiseModel(),
    )
    bknd_impl = SVBackendImpl(config, pulser_data)
    bknd_impl._evolve_step(1.0, 0)
