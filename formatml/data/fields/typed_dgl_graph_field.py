from itertools import islice
from typing import Iterable, List, NamedTuple

from dgl import batch as dgl_batch, DGLGraph
from torch import cat, device as torch_device, long as torch_long, Tensor, tensor

from formatml.data.fields.field import Field
from formatml.data.vocabulary import Vocabulary
from formatml.parsing.parser import Nodes


class TypedDGLGraphFieldOutput(NamedTuple):
    """Output of the graph field."""

    graph: DGLGraph
    etypes: Tensor


class TypedDGLGraphField(Field[Nodes, TypedDGLGraphFieldOutput]):
    def __init__(self, name: str, type: str, edge_types: List[str]) -> None:
        super().__init__(name, type)
        self.vocabulary: Vocabulary[str] = Vocabulary()
        self.vocabulary.add_items(edge_types)

    def tensorize(self, sample: Nodes) -> TypedDGLGraphFieldOutput:
        nodes, node_index, token_indexes, _ = sample
        sources, targets, etypes = [], [], []

        for node in nodes:
            if node.parent is not None:
                if "child" in self.vocabulary:
                    sources.append(node_index[id(node.parent)])
                    targets.append(node_index[id(node)])
                    etypes.append(self.vocabulary.get_index("child"))
                if "parent" in self.vocabulary:
                    sources.append(node_index[id(node.parent)])
                    targets.append(node_index[id(node)])
                    etypes.append(self.vocabulary.get_index("parent"))

        for previous_token_index, next_token_index in zip(
            islice(token_indexes, 0, None), islice(token_indexes, 1, None)
        ):
            if "next_token" in self.vocabulary:
                sources.append(previous_token_index)
                targets.append(next_token_index)
                etypes.append(self.vocabulary.get_index("next_token"))

            if "previous_token" in self.vocabulary:
                sources.append(next_token_index)
                targets.append(previous_token_index)
                etypes.append(self.vocabulary.get_index("previous_token"))

        graph = DGLGraph()
        graph.add_nodes(len(nodes))
        graph.add_edges(sources, targets)

        return TypedDGLGraphFieldOutput(
            graph=graph, etypes=tensor(etypes, dtype=torch_long)
        )

    def collate(
        self, tensors: Iterable[TypedDGLGraphFieldOutput]
    ) -> TypedDGLGraphFieldOutput:
        tensors = list(tensors)
        return TypedDGLGraphFieldOutput(
            graph=dgl_batch([tensor.graph for tensor in tensors]),
            etypes=cat([tensor.etypes for tensor in tensors], dim=0),
        )

    def to(
        self, tensor: TypedDGLGraphFieldOutput, device: torch_device
    ) -> TypedDGLGraphFieldOutput:
        tensor.graph.to(device)
        return TypedDGLGraphFieldOutput(
            graph=tensor.graph, etypes=tensor.etypes.to(device)
        )
