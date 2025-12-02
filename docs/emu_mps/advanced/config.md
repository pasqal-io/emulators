# Explanation of config values

This page documents all the options in `MPSConfig` that controls how emu-mps evolves quantum states (TDVP and DMRG solvers). Each entry explains the meaning, typical uses, recommendations and short examples where helpful.

- dt
- precision
- max_bond_dim
- max_krylov_dim
- extra_krylov_tolerance
- num_gpus_to_use
- autosave_dt
- optimize_qubit_ordering
- interaction_cutoff
- solver

## dt

Note that emu-mps assumes the Hamiltonian is piece-wise constant in time for intervals of `dt`. It then constructs the Hamiltonian by sampling the amplitude, detuning and phase of the pulse midway through the interval, and making a single Hamiltonian. The TDVP or DMRG algorithms are then used to evolve the state by `dt`.

There are two sources of error related to `dt`.

- The discretization of the pulse
- [TDVP](errors.md)

Both sources of error dictate that `dt` shall not be too small, but the functioning of TDVP also dictates that a very small `dt` requires improving the precision, as described in the next section.

Example:

```python
mpsconfig = MPSConfig(dt=10, ...)
```

## precision

The 2-site TDVP and DMRG algorithms used in emu-mps work by repeatedly time-evolving two neighbouring qubits in the MPS, and then truncating the result. Truncation is done by applying an [SVD](https://en.wikipedia.org/wiki/Singular_value_decomposition) to the matrix representing the 2-qubit subsystem.
The singular values give much information about the state. Denote the singular values by $d_i$, and assume they are ordered in decreasing magnitude.
Then the norm of the state will be $\sum_i d_i^2$ and the entanglement entropy between the left and right parts of the state will be $\sum_i d_i \log_2(d_i)$, for example.

The truncation mentioned above functions by throwing away the smallest singular values, until their squared sum exceeds $precision^2$. The result is that the truncation procedure finds the smallest MPS whose distance is less than `precision` away from the original state.
As described in [the page of errors in TDVP](errors.md#truncation-of-the-state), the error in TDVP increases with the number of timesteps, so for long sequences or small `dt`, improving the precision might be required.

Implications:

- Smaller `precision` ⟶ more singular values kept ⟶ higher accuracy and larger memory/CPU cost.

- For long sequences or very small `dt`, improving (reducing) `precision` may be necessary to avoid accumulated truncation error.

Recommendation

- Start with the default precision and lower it only if observables (fidelity, magnetization) show a systematic drift.

## max_bond_dim

In addition to the above procedure, at each truncation step, no more than `max_bond_dim` singular values are kept. This parameter will impose a hard cap on the memory consumed by the quantum state, at the cost of losing control over the magnitude of the truncation errors.

Guidance:

- Increase `max_bond_dim` when precision-limited truncation repeatedly hits the cap.

- Monitor the statistics (`results.statistics` like maximum bond dimension used) and if some memory is still free, raise the cap.

## max_krylov_dim

Time evolution of each qubit pair is done by the [Lanczos algorithm](https://en.wikipedia.org/wiki/Lanczos_algorithm). This algorithm works by iteratively constructing a basis in which the Hamiltonian is close to diagonal. In this basis, the time-evolution of the state can efficiently be approximated by exponentiating a truncated Hamiltonian in the case of TDVP, or minimizing the truncated Hamiltonian in the DMRG case. The algorithm will never construct more than `max_krylov_dim` basis vectors, limiting the runtime and memory consumption of the time-evolution.

Note that the number of iterations the Lanczos algorithm needs to converge to the required tolerance depends on the `dt` parameter also present in the config (see the [api specification](../api.md#mpsconfig)). The default value of `max_krylov_dim` should work for most reasonable values of `dt`, so if you get a recursion error out of the Lanczos algorithm, ensure you understand how the [errors in TDVP](errors.md) depend on `dt`.

Guidance:

- The default value should rarely be changed.

- If there is a recursion error from the Lanczos algorithm, lower `dt` rather than increasing `max_krylov_dim`.

## extra_krylov_tolerance

In addition to the above hard cap on the number of basis vectors, the algorithm will also attempt to estimate the error incurred by computing the matrix exponential using only the current basis vectors. In principle, it is not needed to compute the time-evolution more precisely than `precision` since extra precision will be lost in the truncation. However, in practice it turns out that existing error estimates tend to underestimate the error. `extra_krylov_tolerance` is a fudge factor for how big the desired precision should be compared to `precision`. Its default value is `1e-3`.

Numerical safety

- If `precision * extra_krylov_tolerance` becomes extremely small (close to machine precision ~1e-16 for double precision), the algorithm cannot reliably distinguish noise from convergence and the simulation may produce incorrect results.

- Practical lower bound: keep `precision * extra_krylov_tolerance >= 1e-12` as a conservative guideline. If precision is very small like ~1e-16, then increase `extra_krylov_tolerance` to obey the bound.

Example:

```python
# keep a safe heuristic tolerance
mpsconfig = MPSConfig(precision=1e-6, extra_krylov_tolerance=1e-3, ...)
# heuristic product = 1e-9 (safe)
```

## num_gpus_to_use

The `num_gpus_to_use` parameter sets the number of GPUs over which the MPS tensors are distributed during the simulation.
Setting `num_gpus_to_use = 0` runs the entire computation on the CPU.
Using multiple GPUs can reduce memory usage per GPU, though the overall runtime remains similar. Also, the default value is `None` and emu-mps internally picks up the available GPUs in the machine, otherwise it uses the CPU.

**Example:**
num_gpus_to_use = 2  # use 2 GPUs if available, otherwise fallback to 1 or CPU

```python
mpsconfig = MPSConfig(num_gpus_to_use=2, ...)
```

## optimize_qubit_ordering

The `optimize_qubit_ordering` parameter enables the reordering of qubits in the register. This can be useful in cases where the initial qubit ordering (chosen by the user or by Pulser) is not optimal. In such cases, setting `optimize_qubit_ordering = True` re-orders the qubits more efficiently, and that has been shown to improve performance and accuracy. The default value is `True`.

**Note:** this option (`True`) is not compatible with certain features, such as using a user-provided initial state or the StateResult observable. In this case the option will be changed to `False` automatically after throwing a warning message.

## log_level

Selects the logging verbosity for emu-mps using the standard Python logging levels (integers or logging constants).

Typical inputs and meanings:

- `logging.DEBUG` (10): very detailed debug information (internal state, timing breakdowns).
- `logging.INFO` (20): normal progress messages and high-level events (default).
- `logging.WARNING` (30): recoverable problems and automatic fallbacks.
- `logging.ERROR` (40): errors that prevented an operation from completing.
- `logging.CRITICAL` (50): severe failures.

Example:

```python
import logging
from pathlib import Path

mpsconfig = MPSConfig(
    dt=dt,
    observables=[...],
    log_level=logging.INFO,            # or logging.WARNING, ...
    log_file=Path("emu_mps_run.log"),  # optional: write logs to file
)
```

Recommendations:

- Use `logging.INFO` for interactive runs and examples.
- Use `logging.WARNING` or `logging.ERROR` when running large batches / Monte Carlo simulations to reduce output.
- Consider using `log_file` for long-running simulations to preserve diagnostics without cluttering the console.

## log_file

Saves the logging output in a file. The format is given by the user. Ex: "log_file.log"

## autosave_dt

The `autosave_dt` parameter defines the minimum time interval between two automatic saves of the MPS state. It is given in seconds with a default value `600` ($10$ minutes).
Saving the quantum state for later use (for e.g. to resume the simulation) will only occur at times that are multiples of `autosave_dt`.

Example:

```python
mpsconfig = MPSConfig(autosave_dt=600, autosave_prefix="run_01", ...)
```

## autosave_prefix

Prefix used when writing autosave files (e.g., `<autosave_prefix>_step123.dat`). Helps organize and resume simulations.

## interaction_cutoff

A floating-point threshold below which pairwise interaction matrix elements $U_{ij}$ are treated as zero when building the Hamiltonian. Setting small interactions to 0.0 sparsifies the Hamiltonian and can reduce both memory usage and runtime, at the cost of neglecting weak long-range couplings.

Guidance

- Default: `0.0` (no interactions removed).

- Choose a cutoff as a small fraction of the largest interaction (e.g., 1e-3 … 1e-6 of max|U|) and verify convergence by lowering the cutoff.

Example:

```python
mpsconfig = MPSConfig(interaction_cutoff=1e-6, ...)
```

## solver

The `solver` parameter selects the algorithm used to evolve the system using a Pulser sequence. The `Solver` class is then defined with two possible values:

- `TDVP` — the default value, used to perform real-time evolution of the MPS using the two-site TDVP algorithm.
- `DMRG` is an alternative solver that variationally minimizes the effective Hamiltonian using the two-site DMRG algorithm, **typically applied for simulating adiabatic sequences**.

For a detailed description of the currently available solvers, please refer to the current [algorithms](algorithms.md).

## Practical tips and recommendations

- Validate convergence: always run a small convergence study varying `precision`, `max_bond_dim`, and `interaction_cutoff` to confirm that observables of interest are stable.
- Numerical safety: ensure `precision * extra_krylov_tolerance` is not below ~1e-12 to avoid machine-precision problems.
- Monitoring: check `results.statistics` after a run (bond dimensions, memory, timings) to tune configuration values.
- Reproducibility: specify `log_file` and keep `autosave_prefix` to resume or inspect long jobs.

For a detailed description of algorithmic behavior and error sources, see the algorithms and errors pages:

- Algorithms: [algorithms.md](algorithms.md)
- TDVP/Truncation errors: [errors.md](errors.md)
