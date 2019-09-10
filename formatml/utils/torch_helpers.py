from typing import Any, Iterator, List, Tuple, Union

from matplotlib import use as matplotlib_use

matplotlib_use("Agg")  # noqa
from matplotlib.lines import Line2D  # noqa: I202
import matplotlib.pyplot as plt
from numpy import arange
from torch import Tensor, tensor
from torch.nn.parameter import Parameter
from torch.nn.utils.rnn import PackedSequence
from torch.utils.tensorboard import SummaryWriter


def unpack_packed_sequence(packed_sequence: PackedSequence) -> List[Tensor]:
    result: List[List[Any]] = [[] for _ in packed_sequence.batch_sizes]
    batch_sizes = packed_sequence.batch_sizes.clone()
    current = 0
    while batch_sizes[0] > 0:
        i = 0
        while i < len(batch_sizes) and batch_sizes[i] > 0:
            result[i].append(packed_sequence.data[current])
            current += 1
            batch_sizes[i] -= 1
            i += 1
    return [tensor(l, dtype=packed_sequence.data.dtype) for l in result]


def data_if_packed(input: Union[PackedSequence, Tensor]) -> Tensor:
    if isinstance(input, PackedSequence):
        return input.data
    return input


def log_grad_flow(
    named_parameters: Iterator[Tuple[str, Parameter]],
    writer: SummaryWriter,
    iteration: int,
) -> None:
    """
    Plot the gradients flowing through different layers in the net during training.

    See https://discuss.pytorch.org/t/check-gradient-flow-in-network/15063/10 for
    credit.
    """
    ave_grads = []
    max_grads = []
    layers = []
    for n, p in named_parameters:
        if p.requires_grad and "bias" not in n and p.grad is not None:
            layers.append(n)
            ave_grads.append(p.grad.abs().mean())
            max_grads.append(p.grad.abs().max())
    plt.bar(arange(len(max_grads)), max_grads, alpha=0.1, lw=1, color="c")
    plt.bar(arange(len(max_grads)), ave_grads, alpha=0.1, lw=1, color="b")
    plt.hlines(0, 0, len(ave_grads) + 1, lw=2, color="k")
    plt.xticks(range(0, len(ave_grads), 1), layers, rotation="vertical")
    plt.xlim(left=0, right=len(ave_grads))
    plt.ylim(bottom=-0.001, top=0.02)  # zoom in on the lower gradient regions
    plt.xlabel("Layers")
    plt.ylabel("average gradient")
    plt.title("Gradient flow")
    plt.grid(True)
    plt.legend(
        [
            Line2D([0], [0], color="c", lw=4),
            Line2D([0], [0], color="b", lw=4),
            Line2D([0], [0], color="k", lw=4),
        ],
        ["max-gradient", "mean-gradient", "zero-gradient"],
    )
    writer.add_figure("gradient-flow", plt.gcf(), global_step=iteration)
