from __future__ import annotations
from time import time
from resource import RUSAGE_SELF, getrusage
import json
from pathlib import Path
from typing import Mapping
import torch
from pulser import Sequence
from emu_sv import SVBackend
from pulser.backend import Results


class EmulationModel:
    """
    Emulation model for torch.optim loops.

    TODO make it a base class as it only needs Config, Backend and Results classes
    """

    def __init__(
        self,
        parametrized_seq: Sequence,
        bknd: SVBackend,
        trainable_params: Mapping,
    ):
        if not isinstance(bknd, SVBackend):
            raise AttributeError("`config` must be a SVBackend.")
        self.bknd = bknd

        if not parametrized_seq.is_parametrized():
            raise AttributeError(
                "EmulationModel can only be initialized with parametrized sequences."
            )
        self.parametrized_seq = parametrized_seq

        if trainable_params is None:
            raise AttributeError("No trainable parameters were provided.")
        self.trainable_params = {
            name: torch.as_tensor(val).requires_grad_(True)
            for name, val in trainable_params.items()
        }

        self.update()
        self.logger = EmulationModelLogger(self.trainable_params)

    def update(self) -> None:
        """Update the sequence with the new parameters"""
        built_seq = self.parametrized_seq.build(**self.trainable_params)  # type: ignore
        self.bknd._sequence = built_seq

    def run(self) -> Results:
        return self.bknd.run()

    def parameters(self) -> list[torch.Tensor]:
        return [*self.trainable_params.values()]

    def log_epoch(
        self, epoch: int, loss: torch.Tensor, expectation: torch.Tensor
    ) -> None:
        self.logger._log_epoch_results(loss, expectation)
        self.logger._log_epoch_stats(epoch)
        self.logger._print_log(epoch, loss)


class EmulationModelLogger:
    def __init__(self, trainable_params: Mapping[str, torch.Tensor]):
        self.log: dict = {
            "epoch": [],
            "loss": [],
            "expectation": [],
            "Δt": [],
            "RSSgpu": [],
            "RSScpu": [],
            "timestamp": time(),
        }
        self.trainable_params = trainable_params
        for name in self.trainable_params:
            self.log[name] = []
            self.log[f"{name}.grad"] = []

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

    # def log_grads(self) -> None:
    #    for name, param in self.trainable_params.items():
    #        self.log[f"{name}.grad"].append(param.grad.item())

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
