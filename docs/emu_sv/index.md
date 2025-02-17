# Welcome to emu-sv

You have found the documentation for emu-sv. **Emu-sv** is a backend for the [Pulser low-level Quantum Programming toolkit](https://pulser.readthedocs.io) that lets you run Quantum Algorithms on a simulated device, using GPU acceleration if available. More in depth, emu-sv is designed to **emu**late the dynamics of programmable arrays of neutral atoms, using the **s**tate **v**ector representation, . While benchmarking is incomplete as of this writing, early results suggest that twhich is fastest, and most accurate up to ~20 qubits when compared to tensor network methods such as used in emu-mpshis design makes emu-mps faster and more memory-efficient than previous generations of quantum emulators at running simulations with large numbers of qubits.

## Supported features

The following features are currently supported:

- All Pulser sequences that use only the rydberg channel without complex phase
- MPS and MPO can be constructed using the abstract Pulser format.
- The following noise types:
    - None currently
- The following [basis states](https://pulser.readthedocs.io/en/stable/conventions.html) in a sequence:
    - [ground-rydberg](https://pulser.readthedocs.io/en/stable/review.html#programmable-arrays-of-rydberg-atoms)
- The following properties from a Pulser Sequence are also correctly applied:
    - [hardware modulation](https://pulser.readthedocs.io/en/stable/tutorials/output_mod_eom.html)
    - [SLM mask](https://pulser.readthedocs.io/en/stable/tutorials/slm_mask.html)
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

## Planned features

- Parallel TDVP on multiple GPUs
- The noises from the pulser `NoiseModel` using the density matrix formalism
- Differentiability
- Complex phases in the pulse.
- Overriding the interaction matrix

## More Info
Please see the API specification for a list of available config options ([see here](api.md)).
For notebooks with examples for how to do various things, please see the notebooks page ([see here](./notebooks/index.md)).
