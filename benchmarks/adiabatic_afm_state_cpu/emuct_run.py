import sys
import emu_ct
import pulser

seq = sys.argv[1]

config = emu_ct.MPSConfig(num_devices_to_use=0)
backend = emu_ct.MPSBackend()

backend.run(pulser.Sequence.from_abstract_repr(seq), mps_config=config)
