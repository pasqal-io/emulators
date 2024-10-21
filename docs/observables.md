# The observables mechanism

Since the desired output of the emulator can be quite user dependent, EMU-MPS uses a callback mechanism to specify its output.
Code snippets for doing this are available on the main page ([see here](index.md)).

The following default observables are available:

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


## Defining your own observables
If the above defaults are not sufficient, it is possible for users to subclass the
`Callback` class to define custom results ([see here](base_classes.md#callback)). However, the majority of users are probably satisfied with a combination of the default callbacks included.
