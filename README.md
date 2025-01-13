<div align="center">
  <img src="docs/logos/LogoTaglineSoftGreen.svg">

  # Emu-MPS
</div>

**EMU-MPS** is a backend for the [Pulser low-level Quantum Programming toolkit](https://pulser.readthedocs.io). EMU-MPS lets you transparently run Quantum Algorithms on a simulated device, using GPU acceleration if available. More in depth, EMU-MPS is designed to **EMU**late the dynamics of programmable arrays of neutral atoms, with matrix product states (**MPS**). This design makes it faster and more memory-efficient than previous generations of quantum emulators, which means that you can emulate systems with larger number of qubits, faster.

## Installation

**Warning:** installing emu-mps will update pulser-core

### Using `hatch`, `uv` or any pyproject-compatible Python manager

Edit file `pyproject.toml` to add the line

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
at the [examples](examples/emu_mps_examples).

See also the [full API documentation](https://pqs.pages.pasqal.com/emu_mps/).

For more information, you can check the tutorials and examples located in the [examples folder](https://gitlab.pasqal.com/emulation/rydberg-atoms/emu-ct/-/tree/main/examples?ref_type=heads)

## Documentation

Please check the [documentation](./docs/index.md) page for more info about contributing, the API, benchmarks, etc.


## Getting in touch

- [Pasqal Community Portal](https://community.pasqal.com/) (forums, chat, tutorials, examples, code library).
- [GitHub Repository](https://github.com/pasqal-io/quantum-evolution-kernel) (source code, issue tracker).
- [Professional Support](https://www.pasqal.com/contact-us/) (if you need tech support, custom licenses, a variant of this library optimized for your workload, your own QPU, remote access to a QPU, ...)
