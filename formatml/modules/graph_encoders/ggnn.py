from functools import partial
from logging import getLogger
from typing import Callable, Dict, List, Tuple


from dgl import DGLGraph
from dgl.function import sum as dgl_sum
from dgl.udf import EdgeBatch, NodeBatch
from torch import cat, Tensor
from torch.nn import GRUCell, Linear, ModuleList

from formatml.modules.graph_encoders.graph_encoder import GraphEncoder


EdgeUDF = Callable[[EdgeBatch], Dict[str, Tensor]]
NodeUDF = Callable[[NodeBatch], Dict[str, Tensor]]


class GGNN(GraphEncoder):
    """GGNN layer."""

    _logger = getLogger(__name__)

    def __init__(
        self, in_feats: int, out_feats: int, n_steps: int, n_etypes: int
    ) -> None:
        """Construct a GGNN layer."""
        super().__init__()
        self.in_feats = in_feats
        self.out_feats = out_feats
        self.n_steps = n_steps
        self.n_etypes = n_etypes
        self._linears = ModuleList(
            [Linear(out_feats, out_feats) for _n in range(n_etypes)]
        )
        self._gru = GRUCell(input_size=out_feats, hidden_size=out_feats)

    def forward(  # type: ignore
        self, graph: DGLGraph, feat: Tensor, etypes: List[Tensor]
    ) -> DGLGraph:
        """
        Perform iterative graph updates.

        :param graph: Graph containing the node annotations in ndata["x"].
        :param feat: Node features.
        :param etypes: List of index of edges for each edge type.
        :return: Encoded node features.
        """
        by_type: List[Tuple[Tensor, EdgeUDF]] = [
            (etype, partial(self._message, i)) for i, etype in enumerate(etypes)
        ]
        zero_pad = feat.new_zeros((feat.shape[0], self.out_feats - feat.shape[1]))
        graph.ndata["h"] = cat([feat, zero_pad], -1)

        reduce_function = dgl_sum(msg="m", out="s")

        for _ in range(self.n_steps):
            for tensor_indexes, message_function in by_type:
                graph.send(edges=tensor_indexes, message_func=message_function)
            graph.recv(apply_node_func=self._update, reduce_func=reduce_function)

        return graph.ndata["h"]

    def _message(self, edge_type: int, edge_batch: EdgeBatch) -> Dict[str, Tensor]:
        return {"m": self._linears[edge_type](edge_batch.src["h"])}

    def _update(self, node_batch: NodeBatch) -> Dict[str, Tensor]:
        return {"h": self._gru(input=node_batch.data["s"], hx=node_batch.data["h"])}
