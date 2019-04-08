from typing import Any, List, NamedTuple

from torch.nn import Module

from formatml.utils.from_params import from_params


class ModelOutput(NamedTuple):
    output: Any
    loss: Any


@from_params
class Model(Module):
    """Base class for complete models."""

    def forward(self, *inputs: List[Any]) -> ModelOutput:
        raise NotImplementedError()
