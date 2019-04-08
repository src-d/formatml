from typing import Any, Dict, Generic, List, TypeVar

from formatml.data.fields.field import Field
from formatml.utils.from_params import from_params


_T = TypeVar("_T")
_TInstance = TypeVar("_TInstance", bound="Instance")


@from_params
class Instance(Generic[_T]):
    """Describe how a sample is transformed into a instance feedable to a model."""

    def __init__(self, fields: Dict[str, Field]) -> None:
        """Construct an instance."""
        self.fields = fields

    def pre_tensorize(self, inputs: _T) -> None:
        """
        Compute things before the tensorization itself.

        For example, fill a vocabulary object. Does nothing by default.

        :param inputs: Sample to use for the pre-tensorization.
        """
        for field in self.fields.values():
            field.pre_tensorize(inputs)

    def tensorize(self, inputs: _T) -> Dict[str, Any]:
        """
        Transform a sample into a tensor, or any object that will be fed to the model.

        :param inputs: Sample to tensorize.
        :return: A tensor, or any object that will be directly fed to the model.
        """
        return {
            field_name: field.tensorize(inputs)
            for field_name, field in self.fields.items()
        }

    def collate(self, tensors: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Collate a list of tensorized samples into a batched tensorized sample.

        :param tensors: Tensorized samples to collate.
        :return: Batched tensorized sample.
        """
        return {
            field_name: field.collate(tensor[field_name] for tensor in tensors)
            for field_name, field in self.fields.items()
        }
