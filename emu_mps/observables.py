from pulser.backend.state import State
from pulser.backend.observable import Observable
from emu_mps.mps import MPS
from typing import Sequence, Any
import torch


class EntanglementEntropy(Observable):
    """Entanglement Entropy of the state partition at qubit `mps_site`.

    Args:
        mps_site: the qubit index at which the bipartition is made.
            All qubits with index $\\leq$ `mps_site` are put in the left partition.
        evaluation_times: The relative times at which to store the state.
            If left as `None`, uses the ``default_evaluation_times`` of the
            backend's ``EmulationConfig``.
        tag_suffix: An optional suffix to append to the tag. Needed if
            multiple instances of the same observable are given to the
            same EmulationConfig.
    """

    def __init__(
        self,
        mps_site: int,
        *,
        evaluation_times: Sequence[float] | None = None,
        tag_suffix: str | None = None,
    ):
        super().__init__(evaluation_times=evaluation_times, tag_suffix=tag_suffix)
        self.mps_site = mps_site

    @property
    def _base_tag(self) -> str:
        return "entanglement_entropy"

    def _to_abstract_repr(self) -> dict[str, Any]:
        repr = super()._to_abstract_repr()
        repr["mps_site"] = self.mps_site
        return repr

    def apply(self, *, state: State, **kwargs: Any) -> torch.Tensor:
        if not isinstance(state, MPS):
            raise NotImplementedError(
                "Entanglement entropy observable is only available for emu_mps emulator."
            )
        if not (0 <= self.mps_site <= len(state.factors) - 2):
            raise ValueError(
                f"Invalid bond index {self.mps_site}. "
                f"Expected value in range 0 <= bond_index <= {len(state.factors)-2}."
            )
        return state.entanglement_entropy(self.mps_site)
