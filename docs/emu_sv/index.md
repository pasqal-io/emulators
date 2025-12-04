# Welcome to emu-sv

You have found the documentation for emu-sv. The emulator **emu-sv** is a backend for the [Pulser low-level Quantum Programming toolkit](https://pulser.readthedocs.io) that lets you run quantum algorithms on a simulated device, using GPU acceleration if available. More in depth, emu-sv is designed to **emu**late the dynamics of programmable arrays of neutral atoms, using the **s**tate **v**ector representation. Our benchmarks indicate that on the gpu emu-sv is both faster and more accurate than emu-mps wherever a simulation fits in memory. For typical sequences this means up to ~27 qubits.

## Supported features

The following features are currently supported:

- All Pulser sequences that use only the rydberg channel
- States and Operators can be constructed using the abstract Pulser format.
- All noise from the pulser `NoiseModel` except leakage
    - Effective noise is included using the density matrix formalism
- The following [basis states](https://pulser.readthedocs.io/en/stable/conventions.html) in a sequence:
    - [ground-rydberg](./notebooks/getting_started.ipynb)
- The following properties from a Pulser Sequence are also correctly applied:
    - [hardware modulation](https://pulser.readthedocs.io/en/stable/tutorials/output_mod_eom.html)
    - [SLM mask](https://pulser.readthedocs.io/en/stable/tutorials/slm_mask.html)
- Customizable output, with the following inbuilt options:
    - The quantum state in state vector format
    - Bitstrings
    - The fidelity with respect to a given state
    - The expectation of a given operator
    - The qubit density (magnetization)
    - The correlation matrix
    - The mean, second moment and variance of the energy
- Specification of
    - Initial state
    - Various precision parameters
    - Whether to run on cpu or gpu
    - The $U_{ij}$ coefficients from [here](../emu_mps//advanced/hamiltonian.md)
- In the noiseless case, the emulator is differentiable


## Planned features

- Leakage noise

## More Info
Please see the API specification for a list of available config options ([see here](api.md)).
For notebooks with examples for how to do various things, please see the notebooks page ([see here](./notebooks/index.md)).
