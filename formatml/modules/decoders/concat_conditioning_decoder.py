from torch import cat, Tensor
from torch.nn import Module
from torch.nn.utils.rnn import PackedSequence

from formatml.modules.decoders.decoder import Decoder


class ConcatConditioningDecoder(Decoder):
    """Decoder whose inputs are augmented with a conditionning tensor."""

    def __init__(self, recurrent: Module):
        super().__init__()
        self.recurrent = recurrent

    def forward(  # type: ignore
        self, inputs: PackedSequence, conditions: Tensor
    ) -> Tensor:
        conditions_expanded = cat(
            [conditions[:batch_size] for batch_size in inputs.batch_sizes], dim=0
        )
        conditionned = cat([inputs.data, conditions_expanded], dim=1)
        return self.recurrent(PackedSequence(conditionned, inputs.batch_sizes))
