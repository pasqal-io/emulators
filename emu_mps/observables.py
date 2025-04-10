from pulser.backend.state import State
from pulser.backend.observable import Observable
from emu_mps.custom_callbacks import calculate_entanglement_entropy
from emu_mps.mps import MPS
from typing import Sequence, Any


class EntanglementEntropy(Observable):
    """Entanglement Entropy subclass used only in emu_mps"""

    def __init__(
        self,
        bond_index: int,
        *,
        evaluation_times: Sequence[float] | None = None,
        tag_suffix: str | None = None,
    ):
        super().__init__(evaluation_times=evaluation_times, tag_suffix=tag_suffix)
        self.bond_index = bond_index

    @property
    def _base_tag(self) -> str:
        return "entanglement_entropy"

    def _to_abstract_repr(self) -> dict[str, Any]:
        repr = super()._to_abstract_repr()
        repr["bond_index"] = self.bond_index
        return repr

    def apply(self, *, state: State, **kwargs: Any) -> float:
        if not isinstance(state, MPS):
            raise NotImplementedError(
                "Entanglement entropy observable is only available for emu_mps emulator."
            )
        if not (0 < self.bond_index < len(state.factors)):
            raise ValueError(
                f"Invalid bond index {self.bond_index}. "
                f"Expected value in range 1 <= bond_index < {len(state.factors)}."
            )
        return float(calculate_entanglement_entropy(state, self.bond_index))
