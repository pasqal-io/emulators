from pulser.backend.state import State
from pulser.backend.observable import Observable
from emu_mps.mps import MPS
from typing import Sequence, Any
import torch


class EntanglementEntropy(Observable):
    """Entanglement Entropy subclass used only in emu_mps"""

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
