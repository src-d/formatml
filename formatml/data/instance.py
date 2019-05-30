from typing import Any, Dict, Generic, List, TypeVar

from formatml.data.fields.field import Field
from formatml.utils.from_params import from_params
from formatml.utils.helpers import get_generic_arguments


_T = TypeVar("_T")


@from_params
class Instance(Generic[_T]):
    """Describe how a sample is transformed into a instance feedable to a model."""

    def __init__(self, fields: Dict[str, Field]) -> None:
        """Construct an instance."""
        self.fields = fields
        self._field_input_types = {
            field: get_generic_arguments(Field, field.__class__)[0]
            for field in fields.values()
        }

    def pre_tensorize(self, inputs: _T) -> None:
        """
        Compute things before the tensorization itself.

        For example, fill a vocabulary object. Does nothing by default.

        :param inputs: Sample to use for the pre-tensorization.
        """
        for field in self.fields.values():
            field.pre_tensorize(self._select_input(field, inputs))

    def tensorize(self, inputs: _T) -> Dict[str, Any]:
        """
        Transform a sample into a tensor, or any object that will be fed to the model.

        :param inputs: Sample to tensorize.
        :return: A tensor, or any object that will be directly fed to the model.
        """
        return {
            field_name: field.tensorize(self._select_input(field, inputs))
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

    def _select_input(self, field: Field, inputs: _T) -> Any:
        field_inputs_cls = self._field_input_types[field]
        if isinstance(inputs, field_inputs_cls):
            return inputs
        return inputs[field_inputs_cls]  # type: ignore
