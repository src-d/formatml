from torch.nn import Linear, Module
from torch.optim.adam import Adam
from torch.optim.optimizer import Optimizer

from formatml.utils.from_params import from_params
from formatml.utils.registrable import register


from_params(Module)
from_params(Optimizer)
register(cls=Optimizer, name="adam", no_from_params=True)(Adam)
register(cls=Module, name="linear", no_from_params=True)(Linear)
