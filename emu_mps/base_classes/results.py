from dataclasses import dataclass, field
from typing import Any


@dataclass
class Results:
    _results: dict[str, dict[int, Any]] = field(default_factory=dict)

    def __setitem__(self, name: str, value: Any) -> None:
        self._results[name] = value

    def __getitem__(self, name: str) -> dict[int, Any]:
        if self._results.get(name) is None:
            self._results[name] = {}
        return self._results[name]

    def get_result_names(self) -> list[str]:
        return list(self._results.keys())

    def get_result_times(self, name: str) -> list[int]:
        return list(self._results[name].keys())

    def get_result(self, name: str, time: int) -> Any:
        return self._results[name][time]
