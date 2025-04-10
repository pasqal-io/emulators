from emu_mps.mps import MPS
import torch


def calculate_entanglement_entropy(mps: MPS, b: int) -> float:
    """
    Calculate the von Neumann entanglement entropy at the bond between sites b-1 and b
    S_E = -Tr(s^2 * log(s^2)), where s are the singular values at the chosen bond
    """
    # orthogonalize the MPS at site b
    mps.orthogonalize(b)

    # perform svd on reshaped tensor at index 'b' of the mps
    tensor_at_site_b = mps.factors[b]
    matrix = tensor_at_site_b.reshape(
        tensor_at_site_b.shape[0] * tensor_at_site_b.shape[1], tensor_at_site_b.shape[2]
    )

    _, s, _ = torch.linalg.svd(matrix, full_matrices=False)

    # Calculate entropy from singular values 's'
    s_sqrd = s**2
    s_sqrd = s_sqrd[s_sqrd > 1e-12]

    S_E = -torch.sum(s_sqrd * torch.log(s_sqrd))

    # shift the orthogonality center back to 0 before the new tdvp step
    mps.orthogonalize(0)
    return S_E.item()
