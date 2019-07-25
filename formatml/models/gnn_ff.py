from bisect import bisect_right
from logging import getLogger
from typing import Any, Dict, List

from dgl import unbatch
from torch import float as torch_float, tensor
from torch.nn import LogSoftmax, Module, NLLLoss

from formatml.models.model import Model
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

    def forward(self, sample: Dict[str, Any]) -> Dict[str, Any]:  # type: ignore
        """Forward pass of an embedder, encoder and decoder."""
        if "forward" in sample:
            raise RuntimeError("Forward already computed.")
        if "loss" in sample:
            raise RuntimeError("Loss already computed.")
        graph, edge_types = sample[self.graph_field_name]
        features = [sample[field_name] for field_name in self.feature_field_names]
        formatting_indexes = sample[self.indexes_field_name].indexes
        graph = self.graph_embedder(graph=graph, features=features)
        encodings = self.graph_encoder(graph=graph, edge_types=edge_types)
        label_encodings = self.selector(tensor=encodings, indexes=formatting_indexes)
        projections = self.class_projection(label_encodings)
        softmaxed = self.softmax(projections)
        labels = sample[self.label_field_name]
        sample["forward"] = softmaxed
        if labels is not None:
            sample["loss"] = NLLLoss(
                weight=softmaxed.new(
                    [graph.batch_size, formatting_indexes.numel() - graph.batch_size]
                )
            )(softmaxed, labels)
        return sample

    def decode(self, sample: Dict[str, Any], prefix: str = "") -> None:
        batched_graph = sample["typed_dgl_graph"].graph
        graphs = unbatch(batched_graph)
        start = 0
        total_number_of_nodes = 0
        bounds = []
        numpy_indexes = sample["indexes"].indexes.cpu().numpy()
        for graph in graphs:
            total_number_of_nodes += graph.number_of_nodes()
            end = bisect_right(numpy_indexes, total_number_of_nodes - 1)
            bounds.append((start, end))
            start = end
        for (start, end), path in zip(bounds, sample["metadata"]):
            predictions = sample["indexes"].offsets[start:end][
                sample["forward"][start:end, 1].argsort(descending=True)
            ]
            predictions += 1
            print("%s%s %s" % (prefix, path, " ".join(map(str, predictions.numpy()))))
