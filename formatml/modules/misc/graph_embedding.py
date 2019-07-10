from logging import getLogger
from typing import Any, List

from dgl import DGLGraph
from torch import cat
from torch.nn import Embedding, EmbeddingBag, Module, ModuleList

from formatml.data.vocabulary import Vocabulary


class GraphEmbedding(Module):
    """Embed features to initialize node and edge representations in a graph."""

    _logger = getLogger(__name__)

    def __init__(self, dimensions: List[int], vocabularies: List[Vocabulary]) -> None:
        """Construct a graph embedding layer."""
        super().__init__()
        self.dimensions = dimensions
        self.vocabularies = vocabularies
        self.embeddings = ModuleList()
        self._configured = False

    def forward(self, graph: DGLGraph, features: List[Any]) -> DGLGraph:  # type: ignore
        """Embed features to initialize node representations of the graph."""
        if not self._configured:
            self._configure(features)
        graph.ndata["x"] = cat(
            tensors=[
                emb(*feature) if isinstance(feature, tuple) else emb(feature)
                for emb, feature in zip(self.embeddings, features)
            ],
            dim=1,
        )
        return graph

    def _configure(self, example_features: List[Any]) -> None:
        for dimension, vocabulary, feature in zip(
            self.dimensions, self.vocabularies, example_features
        ):
            embedding_class = EmbeddingBag if isinstance(feature, tuple) else Embedding
            self.embeddings.append(
                embedding_class(num_embeddings=len(vocabulary), embedding_dim=dimension)
            )
            self._configured = True
