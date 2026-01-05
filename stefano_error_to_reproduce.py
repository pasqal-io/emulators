import numpy as np
from pulser import Sequence, Register, Pulse, AnalogDevice
from pulser.backend import BitStrings, Occupation
from emu_sv import SVBackend, SVConfig
from emu_mps import MPSBackend, MPSConfig
from pulser_simulation import QutipBackendV2, QutipConfig
import logging
import math

reg = Register({"q0": [-3, 0], "q1": [3, 0]})
seq = Sequence(reg, AnalogDevice)
seq.declare_channel("ryd", "rydberg_global")
pulse = Pulse.ConstantPulse(400, 1, 0, 0)
seq.add(pulse, channel="ryd")

relat_evalulation_times = np.linspace(0, 1, 13).tolist()  # [0, 1] div 13
evalulation_times = [t * pulse.duration for t in relat_evalulation_times]
bitstrings = BitStrings(evaluation_times=relat_evalulation_times)
occup = Occupation(evaluation_times=relat_evalulation_times)
observables = (bitstrings, occup)

mps_config = MPSConfig(observables=observables, log_level=logging.WARN)
mps_backend = MPSBackend(seq, config=mps_config)
mps_results = mps_backend.run()

sv_config = SVConfig(observables=observables, log_level=logging.WARN)
sv_backend = SVBackend(seq, config=sv_config)
sv_results = sv_backend.run()

qutip_config = QutipConfig(observables=observables)
qutip_backend = QutipBackendV2(seq, config=qutip_config)
qutip_results = qutip_backend.run()


mps_occ = list(zip(mps_results.get_result_times(occup), mps_results.occupation))
sv_occ = list(zip(sv_results.get_result_times(occup), sv_results.occupation))
qutip_occ = list(zip(qutip_results.get_result_times(occup), qutip_results.occupation))


# I fail because emu_sv/emu_mps is missing time = 0 result
qt = qutip_results.get_result_times(occup)
for name, res in (("sv", sv_results), ("mps", mps_results)):
    err_msg = f"{name} times != qutip times"
    assert len(res.get_result_times(occup)) == len(qt), err_msg
    assert np.allclose(res.get_result_times(occup), qt), err_msg


for m, s, q in zip(mps_occ, sv_occ, qutip_occ):  # [1:]
    mps_t, mps_occ = m
    sv_t, sv_occ = s
    q_t, q_occ = q
    assert math.isclose(mps_t, q_t), f"{mps_t} != {q_t}"
    assert math.isclose(sv_t, q_t), f"{sv_t} != {q_t}"

    mps_occ = mps_occ.numpy()
    sv_occ = sv_occ.numpy()
    q_occ = np.array(q_occ)

    ok_mps = np.allclose(mps_occ, q_occ, atol=1e-4)
    assert ok_mps, f"t={mps_t * pulse.duration} : {mps_occ} != {q_occ}"

    ok_sv = np.allclose(sv_occ, q_occ, atol=1e-4)
    assert ok_sv, f"t={sv_t * pulse.duration} : {sv_occ} != {q_occ}"


# I fail because times are slightly different
for s, q in zip(
    sv_results.get_result_times(occup), qutip_results.get_result_times(occup)
):
    # print(f"{s * pulse.duration} != {q * pulse.duration}")
    assert math.isclose(s, q), f"{s} != {q}"

assert np.allclose(
    sv_results.get_result_times(occup), qutip_results.get_result_times(occup)
)
