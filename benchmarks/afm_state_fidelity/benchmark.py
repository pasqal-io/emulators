from pathlib import Path


script_dir = Path(__file__).parent
res_dir = script_dir / "results"
res_dir.mkdir(exist_ok=True)
# store all additional benchmark results in /log
log_dir = res_dir / "log"
log_dir.mkdir(exist_ok=True)

title = "Adiabatic AFM state 2d Fidelity - CPU"
print(f"Starting {title} benchmark")

try:
    from benchmarkutils.plotutils import plot_fidelity_benchmark
    from benchmarkutils.pulserutils import run_with_pulser
    from benchmarkutils.sequenceutils import make_adiabatic_afm_state_2d_seq

    import emu_mps

    # ----PULSER SEQUENCE
    seq = make_adiabatic_afm_state_2d_seq(3, 3)

    # --COMMON SIMULATION PARAMS
    dt = 10.0
    output_name = "afm_state_fidelity.json"

    # ----PYEMUTN RUN
    backend = emu_mps.MPSBackend()
    times = [dt * (i + 1) for i in range(int(seq.get_duration() / dt))]
    obs = [
        emu_mps.QubitDensity(
            basis={"r", "g"}, qubits=seq.register.qubit_ids, times=times
        ),
        emu_mps.Energy(times=times),
        emu_mps.EnergyVariance(times=times),
    ]
    emuct_res = {}
    dts = [5, 10]
    precisions = [1e-6, 1e-5]
    for dt in dts:  # ns
        for j, precision in enumerate(precisions):
            config = emu_mps.MPSConfig(
                num_devices_to_use=0, dt=dt, precision=precision, observables=obs
            )
            emuct_res[f"dt={dt}, prec={precision}"] = backend.run(seq, config)

    # ----PULSER RUN
    _, res_dict = run_with_pulser(
        seq,
        output_dir=str(log_dir),
        output_name=output_name,
        timestep=dt,
        skip_write_output=False,
    )

    plot_fidelity_benchmark(emuct_res, res_dict, title, res_dir)
except Exception:
    raise
finally:
    Path.touch(res_dir / "DONE")
    print(f"{title} benchmark DONE!")
