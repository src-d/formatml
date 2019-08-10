from torch import Tensor
from torch.nn import Module


class Unsqueezer(Module):
    """Wrapper around unsqueezing a tensor."""

    def __init__(self, dim: int = 0):
        super().__init__()
        self.dim = dim

    def forward(self, tensor: Tensor) -> Tensor:  # type: ignore
        """Unsqueeze the tensor at the given dimension."""
        return tensor.unsqueeze(self.dim)
