from torch.nn import Linear

from formatml.data.vocabulary import Vocabulary
from formatml.utils.from_params import from_params


@from_params
class VocabularyLinear(Linear):
    def __init__(
        self, *, in_features: int, vocabulary: Vocabulary, bias: bool = True
    ) -> None:
        super().__init__(
            in_features=in_features, out_features=len(vocabulary), bias=bias
        )
