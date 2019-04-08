from torch.nn import GRU, LSTM, Module

from formatml.utils.from_params import from_params
from formatml.utils.registrable import register


@from_params
class Recurrent(Module):
    pass


register(cls=Recurrent, name="lstm", no_from_params=True)(LSTM)
register(cls=Recurrent, name="gru", no_from_params=True)(GRU)
