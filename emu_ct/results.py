from collections import Counter


from pulser.result import Result

from emu_ct import MPS


class MPSBackendResults(Result):
    def __init__(self, final_mps: MPS):
        self.final_mps = final_mps

    def get_samples(self, n_samples: int = 1000) -> Counter[str]:
        return self.final_mps.sample_mps(num_shots=n_samples)

    def get_state(self) -> MPS:
        return self.final_mps

    @property
    def sampling_dist(self) -> dict[str, float]:
        """Sampling distribution of the measured bitstring."""
        raise NotImplementedError("sampling_dist method is not implemented")

    @property
    def _size(self) -> int:
        raise NotImplementedError("_size method is not implemented")

    def _weights(self) -> None:  # TODO: this must be included
        raise NotImplementedError("_weights method is not implemented")

    def sampling_errors(self) -> dict[str, float]:  # TODO: this must be included
        raise NotImplementedError("sampling_errors method is not implemented")

    def plot_histogram(
        self,
        min_rate: float = 0.001,
        max_n_bitstrings: int | None = None,
        show: bool = True,
    ) -> None:
        raise NotImplementedError("plot_histogram method is not implemented")
