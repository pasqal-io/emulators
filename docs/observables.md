# The callback mechanism

Since the desired output of the emulator can be quite user dependent, emu-mps uses a callback mechanism to specify its output.
The available callbacks, together with examples for how to create them are on this page. How to use created callbacks to obtain results from an emulation is shown in the example notebooks [for emu-sv](./emu_sv/notebooks/getting_started.ipynb) and [for emu-mps](./emu_mps/notebooks/getting_started.ipynb).

The following default callbacks are available. Please take the listed source code with a grain of salt, since individual emulators can overwrite the implementation for performance reasons.

## StateResult
::: emu_base.base_classes.default_callbacks.StateResult

## BitStrings
::: emu_base.base_classes.default_callbacks.BitStrings

## Fidelity
::: emu_base.base_classes.default_callbacks.Fidelity

## Expectation
::: emu_base.base_classes.default_callbacks.Expectation

## CorrelationMatrix
::: emu_base.base_classes.default_callbacks.CorrelationMatrix

## QubitDensity
::: emu_base.base_classes.default_callbacks.QubitDensity

## Energy
::: emu_base.base_classes.default_callbacks.Energy

## SecondMomentOfEnergy
::: emu_base.base_classes.default_callbacks.SecondMomentOfEnergy

## EnergyVariance
::: emu_base.base_classes.default_callbacks.EnergyVariance


## Defining your own callbacks
Most commonly desired information can be obtained using the classes documented above

- arbitrary observables can be measured using `Expectation`
- fidelities on arbitrary states can be computed using `Fidelity`
- Information about the time dependent states and Hamiltonians is available via `StateResult`, `Energy` etc.

If additional behaviour is desired (e.g. the kurtosis of the energy, or entanglement entropy), the user can subclass the `Callback` class to implement any behaviour only depending on the parameters of its `__call__` method ([see here](base_classes.md#callback)). Computation of the entanglement entropy, for example, cannot be done in a backend-independent manner, so it is unlikely to ever make it into the above default list. However, we do intend to define backend-specific callbacks in the future, which would belong to the API of a specific backend. Callbacks that can be implemented in a backend-independent manner can be added to the above list upon popular request.
