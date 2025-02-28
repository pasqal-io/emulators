# Welcome to emu-mps

You have found the documentation for emu-mps. The emulator **emu-mps** is a backend for the [Pulser low-level Quantum Programming toolkit](https://pulser.readthedocs.io) that lets you run Quantum algorithms on a simulated device, using GPU acceleration if available. More in depth, emu-mps is designed to **emu**late the dynamics of programmable arrays of neutral atoms, with matrix product states (**mps**). While benchmarking is incomplete as of this writing, early results suggest that this design makes emu-mps faster and more memory-efficient than previous generations of quantum emulators at running simulations with large numbers of qubits.

## Supported features

The following features are currently supported:

- All Pulser sequences that use only the rydberg channel
- MPS and MPO can be constructed using the abstract Pulser format.
- The following noise types:
    - [SPAM](https://pulser.readthedocs.io/en/stable/tutorials/spam.html)
    - [Monte Carlo quantum jumps](https://pulser.readthedocs.io/en/stable/tutorials/effective_noise.html)
    - A Gaussian laser waist for the global pulse channels.
- The following [basis states](https://pulser.readthedocs.io/en/stable/conventions.html) in a sequence:
    - [ground-rydberg](https://pulser.readthedocs.io/en/stable/review.html#programmable-arrays-of-rydberg-atoms)
    - [XY](https://pulser.readthedocs.io/en/stable/tutorials/xy_spin_chain.html)
- The following properties from a Pulser Sequence are also correctly applied:
    - [hardware modulation](https://pulser.readthedocs.io/en/stable/tutorials/output_mod_eom.html)
    - [SLM mask](https://pulser.readthedocs.io/en/stable/tutorials/slm_mask.html)
    - A complex phase for the omega parameter
- Customizable output, with the folowing inbuilt options:
    - The quantum state in MPS format
    - Bitstrings
    - The fidelity with respect to a given state
    - The expectation of a given operator
    - The qubit density (magnetization)
    - The correlation matrix
    - The mean, second moment and variance of the energy
- Specification of
    - initial state
    - various precision parameters
    - whether to run on cpu or gpu(s)
    - the $U_{ij}$ coefficients from [here](./advanced/hamiltonian.md)
    - A cutoff below which $U_{ij}$ are set to 0 (this makes the computation more memory efficient)

## Planned features

- Parallel TDVP on multiple GPUs
- More noise:
    - the currently unsupported noises in the Pulser `NoiseModel`
- Differentiability

## More Info
Please see the API specification for a list of available config options ([see here](api.md)).
Those configuration options relating to the mathematical functioning of the backend are explained in more detail in the config page ([see here](advanced/config.md)).
For notebooks with examples for how to do various things, please see the notebooks page ([see here](./notebooks/index.md)).
