import numpy as np
import pulser
import emu_sv

# Tutorial on how to use the SPAM erros in emu_svds
# 1.- using the state vector
# 2.- using the density matrix: we need to add an effective noise channel

# Create the sequence
natoms = 4
reg = pulser.Register.rectangle(1, natoms, spacing=10, prefix="q")
seq = pulser.Sequence(reg, pulser.MockDevice)
seq.declare_channel("ch0", "rydberg_global")
duration = 1250
pulse = pulser.Pulse.ConstantPulse(duration, 4 * np.pi, 0.0, 0.0)
seq.add(pulse, "ch0")


# SPAM noise model using state vector
noise_model = pulser.NoiseModel(
    p_false_neg=0.1,  # epsilon prime
    p_false_pos=0.01,
    state_prep_error=0.2,
    runs=1,  # not supported for other values in emu_sv
    samples_per_run=1,  # not supported for other values in emu_sv
)


dt = 10.0
times = [1.0]
sv_config = emu_sv.SVConfig(
    dt=dt,
    krylov_tolerance=1e-5,
    observables=[
        emu_sv.StateResult(evaluation_times=times),
        emu_sv.BitStrings(evaluation_times=times, num_shots=1000),
        emu_sv.Energy(evaluation_times=times),
        emu_sv.Occupation(evaluation_times=times),
    ],
    noise_model=noise_model,
)

backend_sv = emu_sv.SVBackend(seq, config=sv_config)
result_sv = backend_sv.run()

print(result_sv.bitstrings[-1])
print(result_sv.energy[-1])
print(result_sv.occupation[-1])

#


# SPAM noise model using density matrix
# We need to add an effective noise channel to the sequence
noise_model = pulser.NoiseModel(
    p_false_neg=0.1,  # epsilon prime
    p_false_pos=0.1,
    state_prep_error=0.2,
    runs=1,  # not supported for other values in emu_sv
    samples_per_run=1,  # not supported for other values in emu_sv
    depolarizing_rate=0.001,  # effective noise channel
)


dt = 10.0
times = [1.0]
sv_config = emu_sv.SVConfig(
    dt=dt,
    krylov_tolerance=1e-5,
    observables=[
        emu_sv.StateResult(evaluation_times=times),
        emu_sv.BitStrings(evaluation_times=times, num_shots=1000),
        emu_sv.Energy(evaluation_times=times),
        emu_sv.Occupation(evaluation_times=times),
    ],
    noise_model=noise_model,
)

backend_sv = emu_sv.SVBackend(seq, config=sv_config)
result_sv = backend_sv.run()

print(result_sv.bitstrings[-1])
print(result_sv.energy[-1])
print(result_sv.occupation[-1])
