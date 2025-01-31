<div align="center">
  <img src="docs/logos/LogoTaglineSoftGreen.svg">

  # Emu-MPS
</div>

**EMU-MPS** is a backend for the [Pulser low-level Quantum Programming toolkit](https://pulser.readthedocs.io). EMU-MPS lets you transparently run Quantum Algorithms on a simulated device, using GPU acceleration if available. More in depth, EMU-MPS is designed to **EMU**late the dynamics of programmable arrays of neutral atoms, with matrix product states (**MPS**). While benchmarking is incomplete as of this writing, early results suggest that this design makes EMU-MPS faster and more memory-efficient than previous generations of quantum emulators at running simulations with large numbers of qubits.

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
