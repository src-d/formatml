from logging import getLogger
from typing import Any, Dict, List

from torch.nn import LogSoftmax, NLLLoss
from torch.nn.utils.rnn import PackedSequence

from formatml.models.model import Model
from formatml.modules.decoders.decoder import Decoder
from formatml.modules.graph_encoders.graph_encoder import GraphEncoder
from formatml.modules.misc.graph_embedding import GraphEmbedding
from formatml.modules.misc.packed_embedding import PackedEmbedding
from formatml.modules.misc.selector import Selector
from formatml.modules.misc.vocabulary_linear import VocabularyLinear


class GNNRNNModel(Model):
    """GNN encoder followed by RNN decoder module."""

    _logger = getLogger(__name__)

    def __init__(
        self,
        graph_embedder: GraphEmbedding,
        graph_encoder: GraphEncoder,
        output_embedder: PackedEmbedding,
        decoder: Decoder,
        class_projection: VocabularyLinear,
        graph_field_name: str,
        feature_field_names: List[str],
        label_field_name: str,
    ) -> None:
        """Construct a complete model."""
        super().__init__()
        self.graph_embedder = graph_embedder
        self.graph_encoder = graph_encoder
        self.selector = Selector()
        self.output_embedder = output_embedder
        self.decoder = decoder
        self.class_projection = class_projection
        self.graph_field_name = graph_field_name
        self.feature_field_names = feature_field_names
        self.label_field_name = label_field_name
        self.softmax = LogSoftmax(dim=1)
        self.nll = NLLLoss()

    def forward(self, sample: Dict[str, Any]) -> Dict[str, Any]:  # type: ignore
        """Forward pass of an embedder, encoder and decoder."""
        graph, edge_types = sample[self.graph_field_name]
        features = [sample[field_name] for field_name in self.feature_field_names]
        label_indexes, decoder_inputs, labels, _ = sample[self.label_field_name]
        graph = self.graph_embedder(graph=graph, features=features)
        encodings = self.graph_encoder(graph=graph, edge_types=edge_types)
        label_encodings = self.selector(tensor=encodings, indexes=label_indexes)
        decoder_embeddings = self.output_embedder(decoder_inputs)
        outputs, (h_n, c_n) = self.decoder(
            inputs=decoder_embeddings, conditions=label_encodings
        )
        projections = self.class_projection(outputs.data)
        softmaxed = self.softmax(projections)
        output = PackedSequence(softmaxed, outputs.batch_sizes)
        sample["forward"] = output
        if labels is not None:
            sample["loss"] = self.nll(output.data, labels.data)
        return sample
