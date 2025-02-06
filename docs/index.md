# Welcome to emu-mps

**Emu-mps** is a backend for the [Pulser low-level Quantum Programming toolkit](https://pulser.readthedocs.io) that lets you run Quantum Algorithms on a simulated device, using GPU acceleration if available. More in depth, emu-mps is designed to **emu**late the dynamics of programmable arrays of neutral atoms, with matrix product states (**mps**). While benchmarking is incomplete as of this writing, early results suggest that this design makes emu-mps faster and more memory-efficient than previous generations of quantum emulators at running simulations with large numbers of qubits.

As of this writing, Emu-MPS is provided for Linux and macOS but will not work under Windows.

## Installation

**Warning:** installing emu-mps will update pulser-core

### Using `hatch`, `uv` or any pyproject-compatible Python manager

To add `emu-mps` to your project, edit your `pyproject.toml` to add the line

```toml
  "emu-mps"
```

to the list of `dependencies`.


### Using `pip` or `pipx`
To install the `pipy` package using `pip` or `pipx`

1. Create a `venv` if that's not done yet

```sh
$ python -m venv venv

```

2. Enter the venv

If you're running Unix:

```sh
$ . venv/bin/activate
```

If you're running Windows:

```sh
C:\> /path/to/new/virtual/environment/Scripts/activate
```

3. Install the package

```sh
$ pip install emu-mps
# or
$ pipx install emu-mps
```


Join us on [Slack](https://pasqalworkspace.slack.com/archives/C07MUV5K7EU) or by [e-mail](mailto:emulation@pasqal.com) to give us feedback about how you plan to use Emu-MPS or if you require specific feature-upgrades.

## Usage

For the time being, the easiest way to learn how to use this package is to look
at the [examples](examples/emu_mps_examples) and [notebooks](https://pasqal-io.github.io/emulators/latest/).

See also the [full documentation](https://github.com/pasqal-io/emulators/blob/main/docs/index.md) for
the API, information about contributing, benchmarks, etc.


## Getting in touch

- [Pasqal Community Portal](https://community.pasqal.com/) (forums, chat, tutorials, examples, code library).
- [GitHub Repository](https://github.com/pasqal-io/quantum-evolution-kernel) (source code, issue tracker).
- [Professional Support](https://www.pasqal.com/contact-us/) (if you need tech support, custom licenses, a variant of this library optimized for your workload, your own QPU, remote access to a QPU, ...)

## Running a Pulser sequence and getting results

Several example notebooks are included in the online documentation. The index page for them can be found [here](./notebooks/index.md).

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
Please see the API specification for a list of available config options ([see here](api.md)), and the Observables page for how to compute observables ([see here](observables.md)).
Those configuration options relating to the mathematical functioning of the backend are explained in more detail in the Config page ([see here](advanced/config.md)).
For notebooks with examples for how to do various things, please see the notebooks page ([see here](./notebooks/index.md)).
