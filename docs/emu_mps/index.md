# Welcome to emu-mps

You have found the documentation for emu-mps. The emulator **emu-mps** is a backend for the [Pulser low-level Quantum Programming toolkit](https://pulser.readthedocs.io) that lets you run Quantum algorithms on a simulated device, using GPU acceleration if available. More in depth, emu-mps is designed to **emu**late the dynamics of programmable arrays of neutral atoms, with matrix product states (**mps**). While benchmarking is incomplete as of this writing, early results suggest that this design makes emu-mps faster and more memory-efficient than previous generations of quantum emulators at running simulations with large numbers of qubits.

## Supported features

The following features are currently supported:

- All Pulser sequences that use only rydberg (`ground-rydberg` basis) and only microwave (`XY` basis) channel
- MPS and MPO can be constructed using the abstract Pulser format and following the correspondent basis format
- All noise from the pulser `NoiseModel`
  - Effective noise (`eff_noise`) is included using jump or [collapse operators](https://pulser.readthedocs.io/en/stable/noise_model.html#Describing-noise-in-neutral-atom-QPUs-with-a-NoiseModel) and emu-mps uses the quantum jump method or Monte Carlo wave function (MCWF) approach.
- The following [basis states](https://pulser.readthedocs.io/en/stable/conventions.html) in a sequence:

  - [ground-rydberg](./notebooks/getting_started.ipynb)

  - [XY](https://pulser.readthedocs.io/en/stable/tutorials/xy_spin_chain.html)

- The following properties from a Pulser Sequence are also correctly applied:

  - [hardware modulation](https://pulser.readthedocs.io/en/stable/tutorials/output_mod_eom.html)

  - [SLM mask](https://pulser.readthedocs.io/en/stable/tutorials/slm_mask.html)

  - A complex phase for the omega parameter, i.e. the phase $\phi$ in the [driving Hamiltonian](https://pulser.readthedocs.io/en/stable/programming.html#driving-hamiltonian)

- Customizable output, with the folowing inbuilt options:

  - The quantum state in MPS format

  - Bitstrings

  - The fidelity with respect to a given state

  - The expectation of a given operator (as `MPO` or `MPO._from_operator_repr`)

  - The qubit density (magnetization)

  - The correlation matrix

  - The mean, second moment and variance of the energy

  - Entanglement entropy

  - computational statistics: each time step during the simulation will generate the following information:

    - $\chi$ : is the maximum bond dimension of the MPS

    - $|\Psi|$: MPS (the state) memory footprint

    - RSS: max memory allocation

    - $\Delta t$: time that the step took to run (given in seconds)

- Specification of:

  - Initial state ( as `MPS` or `MPS._from_state_amplitudes`)

  - Various precision parameters

  - Whether to run on cpu or gpu(s)

  - The interaction coefficients $U_{ij}$ from [here](./advanced/hamiltonian.md#qpu-hamiltonian)

  - A cutoff below which $U_{ij}$ are set to 0 (this makes the computation more memory efficient)

## Planned features

- More efficient use of GPU by storing tensors on CPU where possible.
- Differentiability.

## More Info

Please see the API specification for a list of available config options ([see here](api.md)).
Those configuration options relating to the mathematical functioning of the backend are explained in more detail in the config page ([see here](advanced/config.md)).
For notebooks with examples for how to do various things, please see the notebooks page ([see here](./notebooks/index.md)).
