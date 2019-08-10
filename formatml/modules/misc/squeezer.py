from typing import Optional

from torch import Tensor
from torch.nn import Module


class Squeezer(Module):
    """Wrapper around squeezing a tensor."""

    def __init__(self, dim: Optional[int] = None):
        super().__init__()
        self.dim = dim

    def forward(self, tensor: Tensor) -> Tensor:  # type: ignore
        """Squeeze the tensor at the given dimension."""
        if self.dim is not None:
            return tensor.squeeze(dim=self.dim)
        else:
            return tensor.squeeze()
