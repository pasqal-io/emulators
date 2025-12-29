import numpy as np
from pulser import Sequence, Register, Pulse, AnalogDevice
from pulser.backend import BitStrings, Occupation
from emu_sv import SVBackend, SVConfig
from pulser_simulation import QutipBackendV2, QutipConfig
import logging
import math

reg = Register({"q0": [-3, 0], "q1": [3, 0]})
seq = Sequence(reg, AnalogDevice)
seq.declare_channel("ryd", "rydberg_global")
pulse = Pulse.ConstantPulse(400, 1, 0, 0)
seq.add(pulse, channel="ryd")

relat_evalulation_times = np.linspace(0, 1, 13).tolist() # [0, 1] div 13
evalulation_times = [t*pulse.duration for t in relat_evalulation_times]
bitstrings = BitStrings(evaluation_times=relat_evalulation_times)
occup = Occupation(evaluation_times=relat_evalulation_times)
observables = (bitstrings, occup)

sv_config = SVConfig(observables=observables, log_level=logging.WARN)
sv_backend = SVBackend(seq, config=sv_config)
sv_results = sv_backend.run()

qutip_config = QutipConfig(observables=observables)
qutip_backend = QutipBackendV2(seq, config=qutip_config)
qutip_results = qutip_backend.run()


sv_times = sv_results.get_result_times(bitstrings)
qutip_times = qutip_results.get_result_times(bitstrings)

sv_occ = list(zip(sv_results.get_result_times(occup), sv_results.occupation))
qutip_occ = list(zip(qutip_results.get_result_times(occup), qutip_results.occupation))



# I fail because emu_sv is missing time = 0 result
assert np.allclose(sv_times,qutip_times)


for s, q in zip(sv_occ, qutip_occ):  #[1:]
    sv_t, sv_occ = s
    q_t, q_occ = q
    assert math.isclose(sv_t, q_t), f"{sv_t} != {q_t}"

    sv_occ = sv_occ.numpy()
    q_occ = np.array(q_occ)
    ok = np.allclose(sv_occ, q_occ, atol=1e-4)
    assert ok, f"t={sv_t * pulse.duration} : {sv_occ} != {q_occ}"


# I fail because times are slightly different
for s, q in zip(sv_times, qutip_times):
    #print(f"{s * pulse.duration} != {q * pulse.duration}")
    assert math.isclose(s, q), f"{s} != {q}"

assert np.allclose(sv_times, qutip_times)
