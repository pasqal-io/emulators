from emu_base import Backend, BackendConfig, Results
from emu_mps.mps_config import MPSConfig
from emu_mps.mps_backend_impl import create_impl, MPSBackendImpl
from pulser import Sequence
import pickle
import os
import time
import logging
import pathlib


class MPSBackend(Backend):
    """
    A backend for emulating Pulser sequences using Matrix Product States (MPS),
    aka tensor trains.
    """

    def resume(self, autosave_file: str | pathlib.Path) -> Results:
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

        return self._run(impl)

    def run(self, sequence: Sequence, mps_config: BackendConfig) -> Results:
        """
        Emulates the given sequence.

        Args:
            sequence: a Pulser sequence to simulate
            mps_config: the backends config. Should be of type MPSConfig

        Returns:
            the simulation results
        """
        assert isinstance(mps_config, MPSConfig)

        self.validate_sequence(sequence)

        impl = create_impl(sequence, mps_config)
        impl.init()  # This is separate from the constructor for testing purposes.

        return self._run(impl)

    @staticmethod
    def _run(impl: MPSBackendImpl) -> Results:
        while not impl.is_finished():
            impl.progress()

        if impl.autosave_file.is_file():
            os.remove(impl.autosave_file)

        return impl.results
