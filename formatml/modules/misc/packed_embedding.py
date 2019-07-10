from torch.nn import Embedding, Module
from torch.nn.utils.rnn import PackedSequence

from formatml.data.vocabulary import Vocabulary


class PackedEmbedding(Module):
    def __init__(self, dimension: int, vocabulary: Vocabulary):
        super().__init__()
        self.embedding = Embedding(
            num_embeddings=len(vocabulary), embedding_dim=dimension
        )

    def forward(self, inputs: PackedSequence) -> PackedSequence:  # type: ignore
        """Embed the packed sequence given as input."""
        return PackedSequence(self.embedding.forward(inputs.data), inputs.batch_sizes)
