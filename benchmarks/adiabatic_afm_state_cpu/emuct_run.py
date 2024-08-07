import sys

import pulser

import emu_mps


seq = sys.argv[1]

config = emu_mps.MPSConfig(num_devices_to_use=0)
backend = emu_mps.MPSBackend()

backend.run(pulser.Sequence.from_abstract_repr(seq), mps_config=config)
