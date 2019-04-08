from itertools import islice
from typing import Iterable, List, NamedTuple

from dgl import batch as dgl_batch, DGLGraph
from torch import cat, long as torch_long, Tensor, tensor

from formatml.data.fields.field import Field
from formatml.data.fields.graph_fields.graph_field import GraphField
from formatml.parser import NodesSample
from formatml.resources.vocabulary import Vocabulary
from formatml.utils.registrable import register


class TypedGraphFieldOutput(NamedTuple):
    """Output of the graph field."""

    graph: DGLGraph
    edges_by_type: List[Tensor]


@register(cls=Field, name="typed_dgl_graph")
class TypedDGLGraphField(GraphField[TypedGraphFieldOutput]):
    def __init__(self, edge_types: List[str], vocabulary: Vocabulary) -> None:
        self.vocabulary = vocabulary
        self.vocabulary.add_items(edge_types)

    def tensorize(self, sample: NodesSample) -> TypedGraphFieldOutput:
        nodes, node_index, token_indexes = sample
        sources, targets, types = [], [], []

        for node in nodes:
            if node.parent is not None:
                if "child" in self.vocabulary:
                    sources.append(node_index[id(node.parent)])
                    targets.append(node_index[id(node)])
                    types.append(self.vocabulary.get_index("child"))
                if "parent" in self.vocabulary:
                    sources.append(node_index[id(node.parent)])
                    targets.append(node_index[id(node)])
                    types.append(self.vocabulary.get_index("parent"))

        for previous_token_index, next_token_index in zip(
            islice(token_indexes, 0, None), islice(token_indexes, 1, None)
        ):
            sources.append(previous_token_index)
            targets.append(next_token_index)
            types.append(self.vocabulary.get_index("next_token"))

            sources.append(next_token_index)
            targets.append(previous_token_index)
            types.append(self.vocabulary.get_index("previous_token"))

        graph = DGLGraph()
        graph.add_nodes(len(nodes))
        graph.add_edges(sources, targets)
        edges_list_by_type: List[List[int]] = [[] for _ in range(len(self.vocabulary))]

        for i, edge_type in enumerate(types):
            edges_list_by_type[edge_type].append(i)

        edges_by_type = [tensor(l, dtype=torch_long) for l in edges_list_by_type]
        return TypedGraphFieldOutput(graph=graph, edges_by_type=edges_by_type)

    def collate(
        self, tensors: Iterable[TypedGraphFieldOutput]
    ) -> TypedGraphFieldOutput:
        tensors = list(tensors)
        return TypedGraphFieldOutput(
            graph=dgl_batch([tensor.graph for tensor in tensors]),
            edges_by_type=[
                cat([tensor.edges_by_type[i] for tensor in tensors], dim=0)
                for i in range(len(tensors[0].edges_by_type))
            ],
        )
