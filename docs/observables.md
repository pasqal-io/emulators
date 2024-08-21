# The observables mechanism

Since the desired output of the emulator can be quite user dependent, EMU-MPS uses a callback mechanism to specify its output.
Code snippets for doing this are available on the [main page](index.md). Furthermore, it is possible for users to subclass the
[Callback](base_classes.md#callback) class to define custom results. However, the majority of users are probably satisfied with a combination of the default callbacks included.

The following default observables are available:

## StateResult
::: emu_mps.base_classes.default_callbacks.StateResult

## BitStrings
::: emu_mps.base_classes.default_callbacks.BitStrings

## Fidelity
::: emu_mps.base_classes.default_callbacks.Fidelity

## Expectation
::: emu_mps.base_classes.default_callbacks.Expectation

## CorrelationMatrix
::: emu_mps.base_classes.default_callbacks.CorrelationMatrix

## QubitDensity
::: emu_mps.base_classes.default_callbacks.QubitDensity

## Energy
::: emu_mps.base_classes.default_callbacks.Energy

## EnergyVariance
::: emu_mps.base_classes.default_callbacks.EnergyVariance
