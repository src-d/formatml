from typing import Generic, Iterable, TypeVar


_TInputs = TypeVar("_TInputs")
_TOutputs = TypeVar("_TOutputs")


class Field(Generic[_TInputs, _TOutputs]):
    """Field of a sample."""

    def index(self, sample: _TInputs) -> None:
        """
        Index things before the tensorization itself.

        For example, fill a vocabulary object. Does nothing by default.

        :param sample: Sample to use for the pre-tensorization.
        """
        pass

    def tensorize(self, sample: _TInputs) -> _TOutputs:
        """
        Transform a sample into a tensor, or any object that will be fed to the model.

        :param sample: Sample to tensorize.
        :return: A tensor, or any object that will be directly fed to the model.
        """
        raise NotImplementedError()

    def collate(self, tensors: Iterable[_TOutputs]) -> _TOutputs:
        """
        Collate a list of tensorized samples into a batched tensorized sample.

        :param tensors: Tensorized samples to collate.
        :return: Batched tensorized sample.
        """
        raise NotImplementedError()
