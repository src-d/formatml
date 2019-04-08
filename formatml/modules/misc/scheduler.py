from torch.optim.lr_scheduler import _LRScheduler, StepLR

from formatml.utils.from_params import from_params
from formatml.utils.registrable import register


@from_params
class Scheduler(_LRScheduler):
    pass


register(cls=Scheduler, name="step_lr", no_from_params=True)(StepLR)
