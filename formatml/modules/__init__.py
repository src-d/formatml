from torch.optim.adam import Adam
from torch.optim.optimizer import Optimizer

from formatml.utils.from_params import from_params
from formatml.utils.registrable import register


from_params(Optimizer)
register(cls=Optimizer, name="adam", no_from_params=True)(Adam)
