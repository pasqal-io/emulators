import logging
import os
import pathlib
import pickle
import time
from collections import Counter

from pulser.backend import EmulatorBackend, Results

from emu_mps.mps_backend_impl import MPSBackendImpl, create_impl
from emu_mps.mps_config import MPSConfig
import emu_mps.optimatrix as opmat

#from pulser.backend import BitStrings, Fidelity, Occupation, CorrelationMatrix
import torch


class MPSBackend(EmulatorBackend):
    """
    A backend for emulating Pulser sequences using Matrix Product States (MPS),
    aka tensor trains.
    """

    default_config = MPSConfig()

    @staticmethod
    def resume(autosave_file: str | pathlib.Path) -> Results:
        """
        Resume simulation from autosave file.
        Only resume simulations from data you trust!
        Unpickling of untrusted data is not safe.
        """
        if isinstance(autosave_file, str):
            autosave_file = pathlib.Path(autosave_file)

        if not autosave_file.is_file():
            raise ValueError(f"Not a file: {autosave_file}")

        with open(autosave_file, "rb") as f:
            impl: MPSBackendImpl = pickle.load(f)

        impl.autosave_file = autosave_file
        impl.last_save_time = time.time()
        impl.config.init_logging()  # FIXME: might be best to take logger object out of config.

        logging.getLogger("global_logger").warning(
            f"Resuming simulation from file {autosave_file}\n"
            f"Saving simulation state every {impl.config.autosave_dt} seconds"
        )

        return MPSBackend._run(impl)

    def run(self) -> Results:
        """
        Emulates the given sequence.

        Returns:
            the simulation results
        """
        assert isinstance(self._config, MPSConfig)

        impl = create_impl(self._sequence, self._config)
        impl.init()  # This is separate from the constructor for testing purposes.
        results = self._run(impl)
        if not self._config.optimise_interaction_matrix:
            return results

        inv_perm = impl.inv_opt_perm
        permute_bitstrings(inv_perm, results)
        permute_occup_and_correlation_mat(inv_perm, results)

        return results

    @staticmethod
    def _run(impl: MPSBackendImpl) -> Results:
        while not impl.is_finished():
            impl.progress()

        if impl.autosave_file.is_file():
            os.remove(impl.autosave_file)

        return impl.results


def permute_bitstrings(perm: list[int], results: Results) -> None:
    if "bitstrings" not in results.get_result_tags():
        return
    uuid_bitstrings = results._find_uuid("bitstrings")
    counter_time_slices = results._results[uuid_bitstrings]
    for t in range(len(counter_time_slices)):
        old_time_slice = counter_time_slices[t]
        new_time_slice = Counter(
            {opmat.permute_string(bstr, perm): c for bstr, c in old_time_slice.items()}
        )
        counter_time_slices[t] = new_time_slice


def permute_occupation(perm: list[int], results: Results) -> None:
    if "occupation" not in results.get_result_tags():
        return
    uuid_occup = results._find_uuid("occupation")
    time_slices = results._results[uuid_occup]
    for t in range(len(time_slices)):
        old_time_slice = time_slices[t]
        new_time_slice = opmat.permute_1D_array(old_time_slice.numpy(), perm)
        time_slices[t] = torch.tensor(new_time_slice)


def permute_correlation_matrix(perm: list[int], results: Results) -> None:
    if "correlation_matrix" not in results.get_result_tags():
        return
    uuid_corr_mat = results._find_uuid("correlation_matrix")
    time_slices = results._results[uuid_corr_mat]
    for t in range(len(time_slices)):
        old_time_slice = time_slices[t]
        new_time_slice = opmat.permute_2D_array(old_time_slice.numpy(), perm)
        time_slices[t] = torch.tensor(new_time_slice)


def permute_occup_and_correlation_mat(perm: list[int], results: Results) -> None:
    for corr in ["occupation", "correlation_matrix"]:
        if corr not in results.get_result_tags():
            return

        uuid_corr = results._find_uuid(corr)
        time_slices = results._results[uuid_corr]
        for t in range(len(time_slices)):
            old_tslice = time_slices[t]
            new_tslice = opmat.permute_array(old_tslice.numpy(), perm)
            time_slices[t] = torch.tensor(new_tslice)