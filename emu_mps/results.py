from __future__ import annotations
from pulser.backend import Results
import json
from typing import cast, Any
import numpy as np
import torch
from uuid import UUID
from copy import deepcopy


class AbstractReprEncoder(json.JSONEncoder):
    """The custom encoder for abstract representation of Pulser objects."""

    def default(self, o: Any) -> dict[str, Any] | list | int | float | str:
        """Handles JSON encoding of objects not supported by default."""
        if hasattr(o, "_to_abstract_repr"):
            return cast(dict, o._to_abstract_repr())
        elif isinstance(o, np.ndarray):
            return cast(list, o.tolist())
        elif isinstance(o, torch.Tensor):
            if len(o.shape) == 0:
                return o.item()
            else:
                return o.tolist()
        elif isinstance(o, UUID):
            return str(o)
        elif isinstance(o, np.integer):
            return int(o)
        elif isinstance(o, set):
            return list(o)
        elif isinstance(o, complex):
            return dict(real=o.real, imag=o.imag)
        else:  # pragma: no cover
            return cast(dict, json.JSONEncoder.default(self, o))


class MPSResults(Results):
    def _to_abstract_repr(self) -> dict:
        d = deepcopy(self.__dict__)
        d["_tagmap"] = {key: str(value) for key, value in d["_tagmap"].items()}
        d["_results"] = {str(key): value for key, value in d["_results"].items()}
        d["_times"] = {str(key): value for key, value in d["_times"].items()}
        return d

    @classmethod
    def _from_abstract_repr(cls, dict: dict) -> MPSResults:
        results = cls(
            atom_order=tuple(dict["atom_order"]), total_duration=dict["total_duration"]
        )
        for key, value in dict["_tagmap"].items():
            results._tagmap[key] = UUID(value)
        for key, value in dict["_results"].items():
            results._results[UUID(key)] = value
        for key, value in dict["_times"].items():
            results._times[UUID(key)] = value
        return results

    def to_abstract_repr(self) -> str:
        return json.dumps(self._to_abstract_repr(), cls=AbstractReprEncoder)

    @classmethod
    def from_abstract_repr(cls, repr: str) -> MPSResults:
        d = json.loads(repr)
        return cls._from_abstract_repr(d)
