from typing import List

from dgl import DGLGraph
from torch import Tensor
from torch.nn import Module

from formatml.utils.from_params import from_params


@from_params
class GraphEncoder(Module):
    def forward(  # type: ignore
        self, *, graph: DGLGraph, edge_types: List[Tensor]
    ) -> DGLGraph:
        raise NotImplementedError()
