from typing import Any, Dict

from torch.nn import Module


class Model(Module):
    """Base class for complete models."""

    def forward(self, sample: Dict[str, Any]) -> Dict[str, Any]:  # type: ignore
        raise NotImplementedError()
