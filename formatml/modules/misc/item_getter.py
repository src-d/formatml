from torch import Tensor
from torch.nn import Module


class ItemGetter(Module):
    """Wrapper around getting an item in a sequence."""

    def __init__(self, index: int):
        super().__init__()
        self.index = index

    def forward(self, tensor: Tensor) -> Tensor:  # type: ignore
        """Get the item the provided index."""
        return tensor[self.index]
