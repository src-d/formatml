from functools import partial
from logging import getLogger
from typing import Callable, Dict, List, Tuple


from dgl import DGLGraph
from dgl.function import sum as dgl_sum
from dgl.init import zero_initializer
from dgl.udf import EdgeBatch, NodeBatch
from torch import Tensor, zeros
from torch.nn import GRUCell, Linear, ModuleList

from formatml.modules.graph_encoders.graph_encoder import GraphEncoder


EdgeUDF = Callable[[EdgeBatch], Dict[str, Tensor]]
NodeUDF = Callable[[NodeBatch], Dict[str, Tensor]]


class GGNN(GraphEncoder):
    """GGNN layer."""

    _logger = getLogger(__name__)

    def __init__(
        self, iterations: int, n_types: int, x_dim: int, h_dim: int, m_dim: int
    ) -> None:
        """Construct a GGNN layer."""
        super().__init__()
        self.iterations = iterations
        self.n_types = n_types
        self.x_dim = x_dim
        self.h_dim = h_dim
        self.m_dim = m_dim
        self.linears = ModuleList([Linear(h_dim, m_dim) for _n in range(n_types)])
        self.gru = GRUCell(input_size=m_dim, hidden_size=h_dim)

    def forward(  # type: ignore
        self, *, graph: DGLGraph, edge_types: List[Tensor]
    ) -> DGLGraph:
        """
        Perform iterative graph updates.

        :param graph: Graph containing the node annotations in ndata["x"].
        :param edge_types: List of index of edges for each edge type.
        :return: Graph containing the node representations in ndata["h"].
        """
        by_type: List[Tuple[Tensor, EdgeUDF]] = [
            (edge_type, partial(self._message, i))
            for i, edge_type in enumerate(edge_types)
        ]
        graph.apply_nodes(self._initialize)
        graph.set_n_initializer(zero_initializer)
        graph.set_e_initializer(zero_initializer)

        reduce_function = dgl_sum(msg="m", out="s")

        for _ in range(self.iterations):
            for tensor_indexes, message_function in by_type:
                graph.send(edges=tensor_indexes, message_func=message_function)
            graph.recv(apply_node_func=self._update, reduce_func=reduce_function)

        return graph.ndata["h"]

    def _message(self, edge_type: int, edge_batch: EdgeBatch) -> Dict[str, Tensor]:
        return {"m": self.linears[edge_type](edge_batch.src["h"])}

    def _update(self, node_batch: NodeBatch) -> Dict[str, Tensor]:
        return {"h": self.gru(input=node_batch.data["s"], hx=node_batch.data["h"])}

    def _initialize(self, node_batch: NodeBatch) -> Dict[str, Tensor]:
        h = zeros((node_batch.data["x"].shape[0], self.h_dim))
        h[:, : node_batch.data["x"].shape[1]] = node_batch.data["x"]
        return {"h": h}
