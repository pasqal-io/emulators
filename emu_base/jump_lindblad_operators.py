from pulser.noise_model import NoiseModel
import torch
import math

dtype = torch.complex128


def get_lindblad_operators(
    *,
    noise_type: str,
    noise_model: NoiseModel,
    interact_type: str = "ising",
    dim: int = 2,
) -> list[torch.Tensor]:

    assert noise_type in noise_model.noise_types

    if noise_type == "relaxation":
        c = math.sqrt(noise_model.relaxation_rate)
        relaxation = torch.zeros(dim, dim, dtype=dtype)
        relaxation[0, 1] = c
        return [relaxation]

    if noise_type == "dephasing":
        if noise_model.hyperfine_dephasing_rate != 0.0:
            raise NotImplementedError(
                "hyperfine_dephasing_rate is supported only in the digital basis"
            )

        c = math.sqrt(noise_model.dephasing_rate / 2)
        dephasing = torch.zeros(dim, dim, dtype=dtype)

        dephasing[0, 0] = c
        dephasing[1, 1] = -c

        return [dephasing]

    if noise_type == "depolarizing":
        c = math.sqrt(noise_model.depolarizing_rate / 4)
        depolarizing_x = torch.zeros(dim, dim, dtype=dtype)
        depolarizing_x[0, 1] = c
        depolarizing_x[1, 0] = c

        depolarizing_y = torch.zeros(dim, dim, dtype=dtype)
        depolarizing_y[0, 1] = torch.tensor(-c * 1.0j, dtype=dtype)
        depolarizing_y[1, 0] = torch.tensor(c * 1.0j, dtype=dtype)

        depolarizing_z = torch.zeros(dim, dim, dtype=dtype)
        depolarizing_z[0, 0] = c
        depolarizing_z[1, 1] = -c

        return [depolarizing_x, depolarizing_y, depolarizing_z]

    if noise_type == "eff_noise":
        if not all(
            isinstance(op, torch.Tensor) and op.shape == (dim, dim)
            for op in noise_model.eff_noise_opers
        ):
            raise ValueError(
                f"Only {dim} by {dim} effective noise operator matrices are "
                "supported and it should be given as torch tensors "
            )

        lindblad_ops = [  # lindblad operators with XY pulser basis are fine
            math.sqrt(rate) * torch.as_tensor(op)
            for rate, op in zip(noise_model.eff_noise_rates, noise_model.eff_noise_opers)
        ]

        # pulser ising basis changing to emu-mps ising basis
        if interact_type == "ising":
            for tensor in lindblad_ops:
                tensor[:2, :2] = torch.flip(tensor[:2, :2], (0, 1))

        return lindblad_ops
    if noise_type == "leakage":  # leakage operators are eff_noise
        return []

    raise ValueError(f"Unknown noise type: {noise_type}")


def compute_noise_from_lindbladians(
    lindbladians: list[torch.Tensor], dim: int = 2
) -> torch.Tensor:
    """
    Compute the single-qubit Hamiltonian noise term -0.5i∑L†L from all the
    given lindbladians.
    """
    assert all(
        lindbladian.shape == (dim, dim) for lindbladian in lindbladians
    ), "Only single-qubit lindblad operators are supported"

    zero = torch.zeros(dim, dim, dtype=dtype)

    return -0.5j * sum((L.mH @ L for L in lindbladians), start=zero)
