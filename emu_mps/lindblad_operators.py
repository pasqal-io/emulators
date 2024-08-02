from pulser.noise_model import NoiseModel
import torch
import math


def get_lindblad_operators(
    *, noise_type: str, noise_model: NoiseModel
) -> list[torch.Tensor]:
    assert noise_type in noise_model.noise_types

    if noise_type == "relaxation":
        c = math.sqrt(noise_model.relaxation_rate)
        return [
            torch.tensor([[0, c], [0, 0]], dtype=torch.complex128),
        ]

    if noise_type == "dephasing":
        if noise_model.hyperfine_dephasing_rate != 0.0:
            raise NotImplementedError("hyperfine_dephasing_rate is unsupported")

        c = math.sqrt(noise_model.dephasing_rate / 2)
        return [
            torch.tensor([[-c, 0], [0, c]], dtype=torch.complex128),
        ]

    if noise_type == "depolarizing":
        c = math.sqrt(noise_model.depolarizing_rate / 4)
        return [
            torch.tensor([[0, c], [c, 0]], dtype=torch.complex128),
            torch.tensor([[0, 1j * c], [-1j * c, 0]], dtype=torch.complex128),
            torch.tensor([[-c, 0], [0, c]], dtype=torch.complex128),
        ]

    if noise_type == "eff_noise":
        if not all(op.shape == (2, 2) for op in noise_model.eff_noise_opers):
            raise ValueError("Only 2 * 2 effective noise operator matrices are supported")

        return [
            math.sqrt(rate)
            * torch.flip(op if isinstance(op, torch.Tensor) else torch.tensor(op), (0, 1))
            for rate, op in zip(noise_model.eff_noise_rates, noise_model.eff_noise_opers)
        ]

    raise ValueError(f"Unknown noise type: {noise_type}")
