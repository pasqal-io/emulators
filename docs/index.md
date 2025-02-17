# Welcome to the Pasqal analog emulators

There are currently two emulators, emu-sv and emu-mps, the specific documentation for which you can access from the list above.
As of this writing, the emulators are provided for Linux and macOS but will not work under Windows.

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
$ pip install <emulator>
# or
$ pipx install <emulator>
```


Join us on [Slack](https://pasqalworkspace.slack.com/archives/C07MUV5K7EU) or by [e-mail](mailto:emulation@pasqal.com) to give us feedback about how you plan to use Emu-MPS or if you require specific feature-upgrades.

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
