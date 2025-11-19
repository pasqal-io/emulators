# Welcome to the Pasqal analog emulators

Emu-sv and emu-mps are [Pulser](https://pulser.readthedocs.io/en/stable/) high-performance backend emulators for analog quantum simulation, developed by Pasqal. They support all major operating systems (Linux, macOS and Windows), and are designed to faithfully emulate neutral-atom quantum hardware with both high fidelity and scalability.

These tools are described in our publication *“Efficient Emulation of Neutral Atom Quantum Hardware”* ([arXiv:2510.09813](https://arxiv.org/abs/2510.09813)) — in summary:

- emu-sv uses a state-vector representation to deliver numerically exact dynamics for medium-sized systems.

- emu-mps uses a Matrix Product State ([MPS](https://tensornetwork.org/mps/)) representation to scale efficiently to larger system sizes.

- Both integrate with the Pulser framework, modelling pulse sequences, noise and decoherence in neutral-atom arrays.

---

## Which emulator to choose

First, compare the supported features for [emu-sv](./emu_sv/index.md) and [emu-mps](./emu_mps/index.md), since the two emulators do not implement exactly the same feature set. Also consult the in-depth benchmarks for [emu-sv](./emu_sv/benchmarks/index.md) and [emu-mps](./emu_mps/benchmarks/index.md) to determine which emulator - and which parameter settings - best fit your problem.

General guidance:

- **emu-sv** (default precision) is typically the best choice for noiseless simulations of up to ~25 qubits.

- In noise‐simulation mode, emu-sv solves the full Lindblad master equation, but this effectively halves the number of qubits that can be simulated.

- For larger number of qubits, emu-sv may not fit on a GPU or may become significantly slower; in those cases **emu-mps** is often the better choice. Unless you require extreme accuracy, emu-mps offers better scalability — but be sure to read its documentation to configure it correctly.

---

## Installation

**Warning:** installing any emulator will update pulser-core

### Using `hatch`, `uv` or any pyproject-compatible Python manager

To add `emu-sv` or `emu-mps`to your project, edit your `pyproject.toml` to add the line

```toml
  "<emulator>"
```

to the list of `dependencies`.

### Using `pip` or `pipx`

To install the `pipy` package using `pip` or `pipx`

1. Create a `venv` if that's not done yet

```sh
python -m venv venv

```

2. Activate the venv

If you're running Unix:

```sh
. venv/bin/activate
```

3. Install the package

```sh
$ pip install <emulator>
# or
$ pipx install <emulator>
```

Join us on [Slack](https://pasqalworkspace.slack.com/archives/C07MUV5K7EU) or by [e-mail](mailto:emulation@pasqal.com) to give us feedback about how you plan to use the emulators or if you require specific feature-upgrades.

## Usage

For the time being, the easiest way to learn how to use the emulators is to look
at the examples in the [repo](https://github.com/pasqal-io/emulators), [the emu-sv notebooks](emu_sv/notebooks/index.md) and [the emu-mps notebooks](emu_mps/notebooks/index.md).

See also the emulator specific documentation for supported features, benchmarks etc.

## Getting in touch

- [Pasqal Community Portal](https://community.pasqal.com/) (forums, chat, tutorials, examples, code library).
- [GitHub Repository](https://github.com/pasqal-io/emulators) (source code, issue tracker).
- [Professional Support](https://www.pasqal.com/contact-us/) (if you need tech support, custom licenses, a variant of this library optimized for your workload, your own QPU, remote access to a QPU, ...)

## Running a Pulser sequence and getting results

Several example notebooks are included in this documentation. Please see the links provided under [usage](#usage)

## More Info

Many usage patterns are shared between all emulators. The computing observable page details how to compute observables ([see here](observables.md)). The base classes enforcing the usage pattern are documented [here](./base_classes.md).
