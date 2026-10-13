import contextlib
import random
from collections.abc import Generator

import numpy as np
import pytest
import torch


@contextlib.contextmanager
def fork_all_rng() -> Generator:
    """
    Save the python, numpy and torch (CPU and CUDA) RNG states
    and restore them on exit, even if an exception is raised.
    """
    py_state = random.getstate()
    np_state = np.random.get_state()
    try:
        with torch.random.fork_rng():
            yield
    finally:
        np.random.set_state(np_state)
        random.setstate(py_state)


@pytest.fixture
def restore_rng_state() -> Generator:
    with fork_all_rng():
        yield  # run the test
