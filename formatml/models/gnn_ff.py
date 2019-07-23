from logging import getLogger
from typing import Any, Dict, List

from torch import float as torch_float, tensor
from torch.nn import LogSoftmax, Module, NLLLoss

from formatml.models.model import Model, ModelOutput
from formatml.modules.graph_encoders.graph_encoder import GraphEncoder
from formatml.modules.misc.graph_embedding import GraphEmbedding
from formatml.modules.misc.selector import Selector


class GNNFFModel(Model):
    """GNN encoder followed by a feed-forward output projector."""

    _logger = getLogger(__name__)

    def __init__(
        self,
        graph_embedder: GraphEmbedding,
        graph_encoder: GraphEncoder,
        class_projection: Module,
        graph_field_name: str,
        feature_field_names: List[str],
        indexes_field_name: str,
        label_field_name: str,
    ) -> None:
        """Construct a complete model."""
        super().__init__()
        self.graph_embedder = graph_embedder
        self.graph_encoder = graph_encoder
        self.selector = Selector()
        self.class_projection = class_projection
        self.graph_field_name = graph_field_name
        self.feature_field_names = feature_field_names
        self.indexes_field_name = indexes_field_name
        self.label_field_name = label_field_name
        self.softmax = LogSoftmax(dim=1)
        self.nll = NLLLoss(weight=tensor([1, 2000], dtype=torch_float))

    def forward(self, instance: Dict[str, Any]) -> ModelOutput:  # type: ignore
        """Forward pass of an embedder, encoder and decoder."""
        graph, edge_types = instance[self.graph_field_name]
        features = [instance[field_name] for field_name in self.feature_field_names]
        formatting_indexes = instance[self.indexes_field_name].indexes
        graph = self.graph_embedder(graph=graph, features=features)
        encodings = self.graph_encoder(graph=graph, edge_types=edge_types)
        label_encodings = self.selector(tensor=encodings, indexes=formatting_indexes)
        projections = self.class_projection(label_encodings)
        softmaxed = self.softmax(projections)
        if self.label_field_name in instance:
            loss = self.nll(softmaxed, instance[self.label_field_name])
        else:
            loss = None
        return ModelOutput(output=softmaxed, loss=loss)
