from __future__ import annotations
from time import time
from resource import RUSAGE_SELF, getrusage
import json
from pathlib import Path

import torch
from torch.nn import Module
from pulser import Sequence
from emu_sv import SVConfig, SVBackend
from pulser.backend import Results


class EmulationModel(Module):
    """
    Emulation model for torch.optim loops.

    TODO make it a base class as it only needs Config, Backend and Results classes
    """

    def __init__(
        self,
        parametrized_seq: Sequence,
        trainable_params: dict[str, torch.Tensor],
        config: SVConfig,
    ):

        super().__init__()

        self.config = config
        self.sim = SVBackend(parametrized_seq, config=config)

        if not parametrized_seq.is_parametrized():
            msg = "EmulationModel can only be initialized with a parametrized sequence."
            raise AttributeError(msg)
        self.parametrized_seq = parametrized_seq

        # register trainable parameters
        if trainable_params is not None:
            self.trainable_params = torch.nn.ParameterDict(
                {
                    name: torch.nn.Parameter(val, requires_grad=True)
                    for name, val in trainable_params.items()
                }
            )
        else:
            self.trainable_params = torch.nn.ParameterDict(
                {
                    name: torch.nn.Parameter(torch.rand(1) + 1.0, requires_grad=True)
                    for name in parametrized_seq.declared_variables
                }
            )

        # build sequence from parameterized one
        self.update_sequence()

        # init logger for stats, loss and parameters
        self.log: dict = {
            "epoch": [],
            "loss": [],
            "expectation": [],
            "Δt": [],
            "RSSgpu": [],
            "RSScpu": [],
            "timestamp": time(),
        }
        for name in self.trainable_params:
            self.log[name] = []
            self.log[f"{name}.grad"] = []

    def update_sequence(self) -> None:
        """Builds a pulser.Sequence from a dict of updated torch parameters"""
        params_for_sequence = dict(self.trainable_params.items())
        self.built_seq = self.parametrized_seq.build(**params_for_sequence)

    def run(self) -> Results:
        result = self.sim.run()
        self.run_stats = result.statistics["steps"]
        return result

    def log_epoch(
        self, epoch: int, loss: torch.Tensor, expectation: torch.Tensor
    ) -> None:
        self._log_epoch_results(loss, expectation)
        self._log_epoch_stats(epoch)
        self._print_log(epoch, loss)

    def _print_log(self, epoch: int, loss: torch.Tensor) -> None:
        print(f"{epoch})", "loss:", f"{loss.item():>6f}")

        params_log = "\t"
        for name, param in self.trainable_params.items():
            params_log += f"{name}: {param.item():.3f}\t"
        print(params_log)

        delta_time = self.log["Δt"][epoch]
        RSSgpu = self.log["RSSgpu"][epoch]
        RSScpu = self.log["RSScpu"][epoch]
        print(
            f"Δt = {delta_time:.3f} s, "
            + f"RSSgpu = {RSSgpu:.3f} MB, "
            + f"RSScpu = {RSScpu:.3f} MB, "
        )

    def _log_epoch_results(self, loss: torch.Tensor, expectation: torch.Tensor) -> None:
        self.log["loss"].append(loss.item())
        self.log["expectation"].append(expectation.item())
        for name, param in self.trainable_params.items():
            self.log[name].append(param.item())

    def log_grads(self) -> None:
        for name, param in self.trainable_params.items():
            self.log[f"{name}.grad"].append(param.grad.item())

    def _log_epoch_stats(self, epoch: int) -> None:
        max_mem_cpu = getrusage(RUSAGE_SELF).ru_maxrss * 1e-3
        max_mem_gpu = 0.0
        if torch.cuda.device_count():
            max_mem_per_device = (
                torch.cuda.max_memory_allocated(device) * 1e-6
                for device in range(torch.cuda.device_count())
            )
            max_mem_gpu = max(max_mem_per_device)

        delta_time = time() - self.log["timestamp"]
        self.log["timestamp"] = time()

        self.log["epoch"].append(epoch)
        self.log["Δt"].append(delta_time)
        self.log["RSSgpu"].append(max_mem_gpu)
        self.log["RSScpu"].append(max_mem_cpu)

    def save_log(self, output: Path) -> None:
        with open(output, "w") as f:
            f.write(json.dumps(self.log, indent=4))
