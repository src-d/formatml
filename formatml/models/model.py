from typing import Any, List, NamedTuple

from torch.nn import Module


class ModelOutput(NamedTuple):
    output: Any
    loss: Any


class Model(Module):
    """Base class for complete models."""

    def forward(self, *inputs: List[Any]) -> ModelOutput:
        raise NotImplementedError()
