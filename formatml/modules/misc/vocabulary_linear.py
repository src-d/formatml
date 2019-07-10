from torch.nn import Linear

from formatml.data.vocabulary import Vocabulary


class VocabularyLinear(Linear):
    def __init__(
        self, *, in_features: int, vocabulary: Vocabulary, bias: bool = True
    ) -> None:
        super().__init__(
            in_features=in_features, out_features=len(vocabulary), bias=bias
        )
