from torch import Tensor
from torch.nn import Module


class Selector(Module):
    """Wrapper around selection in a tensor."""

    def forward(self, *, tensor: Tensor, indexes: Tensor) -> Tensor:  # type: ignore
        """Select given indexes in the first axis of a tensor."""
        return tensor[indexes]
