from torch.nn import Module

from formatml.utils.from_params import from_params


@from_params
class Decoder(Module):
    """Base class for decoders."""

    pass
