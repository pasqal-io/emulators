import numpy as np
from pulser import Sequence, Register, Pulse, AnalogDevice
from pulser.backend import BitStrings
from emu_sv import SVBackend, SVConfig
from pulser_simulation import QutipBackendV2, QutipConfig
import logging
import math

reg = Register({"q0": [-3, 0], "q1": [3, 0]})
seq = Sequence(reg, AnalogDevice)
seq.declare_channel("ryd", "rydberg_global")
pulse = Pulse.ConstantPulse(400, 1, 0, 0)
seq.add(pulse, channel="ryd")

evalulation_times = np.linspace(0, 1, 13).tolist()
bitstrings = BitStrings(evaluation_times=evalulation_times)
observables = (bitstrings,)

sv_config = SVConfig(observables=observables, log_level=logging.WARN)
sv_backend = SVBackend(seq, config=sv_config)
sv_results = sv_backend.run()

qutip_config = QutipConfig(observables=observables)
qutip_backend = QutipBackendV2(seq, config=qutip_config)
qutip_results = qutip_backend.run()


sv_times = sv_results.get_result_times(bitstrings)
qutip_times = qutip_results.get_result_times(bitstrings)

# I fail because emu_sv is missing time = 0 result
# assert np.allclose(sv_times,qutip_times)

# I fail because times are slightly different

for s, q in zip(sv_times, qutip_times[1:]):
    assert math.isclose(s, q), f"{s} != {q}"

assert np.allclose(sv_times, qutip_times[1:])
