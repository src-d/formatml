from itertools import islice
from typing import Iterable, List, NamedTuple

from dgl import batch as dgl_batch, DGLGraph
from torch import cat, device as torch_device, long as torch_long, Tensor, tensor

from formatml.data.fields.graph_fields.graph_field import GraphField
from formatml.data.vocabulary import Vocabulary
from formatml.parsing.parser import Nodes


class TypedDGLGraphFieldOutput(NamedTuple):
    """Output of the graph field."""

    graph: DGLGraph
    edges_by_type: List[Tensor]


class TypedDGLGraphField(GraphField[TypedDGLGraphFieldOutput]):
    def __init__(self, edge_types: List[str]) -> None:
        self.vocabulary: Vocabulary[str] = Vocabulary()
        self.vocabulary.add_items(edge_types)

    def tensorize(self, sample: Nodes) -> TypedDGLGraphFieldOutput:
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
            if "next_token" in self.vocabulary:
                sources.append(previous_token_index)
                targets.append(next_token_index)
                types.append(self.vocabulary.get_index("next_token"))

            if "previous_token" in self.vocabulary:
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
        return TypedDGLGraphFieldOutput(graph=graph, edges_by_type=edges_by_type)

    def collate(
        self, tensors: Iterable[TypedDGLGraphFieldOutput]
    ) -> TypedDGLGraphFieldOutput:
        tensors = list(tensors)
        n_types = range(len(tensors[0].edges_by_type))
        offset_edges_by_type: List[List[Tensor]] = [[] for _ in n_types]
        offset = 0
        for t in tensors:
            for i in n_types:
                offset_edges_by_type[i].append(t.edges_by_type[i] + offset)
            offset += t.graph.number_of_edges()
        return TypedDGLGraphFieldOutput(
            graph=dgl_batch([tensor.graph for tensor in tensors]),
            edges_by_type=[cat(offset_edges_by_type[i], dim=0) for i in n_types],
        )

    def to(
        self, tensor: TypedDGLGraphFieldOutput, device: torch_device
    ) -> TypedDGLGraphFieldOutput:
        tensor.graph.to(device)
        return TypedDGLGraphFieldOutput(
            graph=tensor.graph,
            edges_by_type=[t.to(device) for t in tensor.edges_by_type],
        )
