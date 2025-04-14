import pulser
import emu_mps
from emu_mps.observables import EntanglementEntropy
import torch


def create_constant_pulse_sequence():
    # Choosing the device
    device = pulser.AnalogDevice

    # creating the register
    square_length = 3
    omega = 12
    delta = 4
    R_interatomic = pulser.AnalogDevice.rydberg_blockade_radius(2 * omega)
    register = pulser.Register.square(square_length, R_interatomic, prefix="q")

    # creating the sequence
    seq = pulser.Sequence(register, device)
    seq.declare_channel("rydberg_global", "rydberg_global")

    # adding constant pulses to the sequence
    pulse1 = pulser.Pulse.ConstantPulse(100, 0, 0, 0)
    pulse2 = pulser.Pulse.ConstantPulse(200, 0.5 * omega, 0.2 * delta, 0)
    pulse3 = pulser.Pulse.ConstantPulse(200, 0.2 * omega, 0.3 * delta, 0)
    seq.add(pulse1, "rydberg_global")
    seq.add(pulse2, "rydberg_global")
    seq.add(pulse3, "rydberg_global")

    return seq


def test_zero_entropy_product_state():
    """
    starting from a product state, the S_E should be zero across all bonds
    """
    initial_state = emu_mps.MPS.from_state_amplitudes(
        eigenstates=("r", "g"), amplitudes={"rrrrrrrrr": 1.0}
    )
    entropies = []
    for b in range(8):
        ent_entropy = EntanglementEntropy(mps_site=b, evaluation_times=[1 / 100])
        config = emu_mps.MPSConfig(observables=[ent_entropy], initial_state=initial_state)
        seq = create_constant_pulse_sequence()
        backend = emu_mps.MPSBackend(sequence=seq, config=config)
        result = backend.run()
        S_E_res = result.get_result(ent_entropy, time=1 / 100)
        entropies.append(S_E_res)

    assert all(s < 1e-7 for s in entropies)


def create_simple_superposition_state():
    """
    Create the max entangled state (|rrrrrrrrr> + |ggggggggg>)/sqrt(2)
    """
    return {"r" * 9: 1 / 2**0.5, "g" * 9: 1 / 2**0.5}


def test_entropy_superposition_state():
    """
    The S_E for the Bell state should be log(2) across all bonds initially
    After the simulation, S_E (center bonds) > S_E (edges)
    """
    amplitude_superposition = create_simple_superposition_state()
    initial_state = emu_mps.MPS.from_state_amplitudes(
        eigenstates=("r", "g"), amplitudes=amplitude_superposition
    )

    entropies_t_initial = []
    entropies_t_finals = []

    for b in range(8):
        ent_entropy = EntanglementEntropy(mps_site=b, evaluation_times=[1 / 100, 1.0])
        config = emu_mps.MPSConfig(observables=[ent_entropy], initial_state=initial_state)
        seq = create_constant_pulse_sequence()
        backend = emu_mps.MPSBackend(sequence=seq, config=config)
        result = backend.run()
        entropies_t_initial.append(result.get_result(ent_entropy, time=1 / 100))
        entropies_t_finals.append(result.get_result(ent_entropy, time=1))

    # --- Check at t = 1/100: entropy = log (2) for the max entangled Bell state
    entropies_t_initial = torch.tensor(entropies_t_initial)
    expected = torch.log(torch.tensor(2.0, dtype=torch.float64))

    assert torch.allclose(
        entropies_t_initial, expected, atol=1e-6
    ), f"Entanglement entropies should be log(2): {entropies_t_initial}"

    # --- Check at the end of simulation: central bonds have higher entropy than the edges
    center_bonds = [4, 5]
    center_entropies = [entropies_t_finals[b] for b in center_bonds]
    edge_entropies = [entropies_t_finals[0], entropies_t_finals[-1]]

    assert all(
        SE_center > SE_edge
        for SE_center, SE_edge in zip(center_entropies, edge_entropies)
    ), "Central bonds should have higher entropy than the edge bonds."
