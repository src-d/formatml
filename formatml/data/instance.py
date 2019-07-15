from bz2 import open as bz2_open
from pathlib import Path
from pickle import dump as pickle_dump
from typing import Any, Dict, List, Tuple

from torch import device as torch_device

from formatml.data.fields.field import Field
from formatml.utils.helpers import get_generic_arguments


class Instance:
    """Describe how a sample is transformed into a instance feedable to a model."""

    def __init__(self, fields: List[Tuple[str, Field]]) -> None:
        """Construct an instance."""
        self.fields = fields
        self._field_input_types = {
            field: get_generic_arguments(Field, field.__class__)[0]
            for _, field in fields
        }

    def index(self, inputs: Any) -> None:
        """
        Index things before the tensorization itself.

        For example, fill a vocabulary object. Does nothing by default.

        :param inputs: Sample to use for the pre-tensorization.
        """
        for _, field in self.fields:
            field.index(self._select_input(field, inputs))

    def tensorize(self, inputs: Any) -> Dict[str, Any]:
        """
        Transform a sample into a tensor, or any object that will be fed to the model.

        :param inputs: Sample to tensorize.
        :return: A tensor, or any object that will be directly fed to the model.
        """
        return {
            field_name: field.tensorize(self._select_input(field, inputs))
            for field_name, field in self.fields
        }

    def collate(self, tensors: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Collate a list of tensorized samples into a batched tensorized sample.

        :param tensors: Tensorized samples to collate.
        :return: Batched tensorized sample.
        """
        return {
            field_name: field.collate(tensor[field_name] for tensor in tensors)
            for field_name, field in self.fields
        }

    def to(self, tensor: Dict[str, Any], device: torch_device) -> Dict[str, Any]:
        return {
            field_name: field.to(tensor[field_name], device)
            for field_name, field in self.fields
        }

    def save(self, file_path: Path) -> None:
        with bz2_open(file_path, "wb") as fh:
            pickle_dump(self, fh)

    def _select_input(self, field: Field, inputs: Any) -> Any:
        field_inputs_cls = self._field_input_types[field]
        if isinstance(inputs, field_inputs_cls):
            return inputs
        return inputs[field_inputs_cls]  # type: ignore
