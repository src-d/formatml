from bz2 import open as bz2_open
from pathlib import Path
from pickle import dump as pickle_dump, load as pickle_load
from typing import Any, Dict, Generic, List, Optional, Tuple, TypeVar

from formatml.data.fields.field import Field
from formatml.utils.helpers import get_generic_arguments


_T = TypeVar("_T")


class Instance(Generic[_T]):
    """Describe how a sample is transformed into a instance feedable to a model."""

    _pickle_filename = "instance.pickle.bz2"

    def __init__(
        self, fields: List[Tuple[str, Field]], cache_dir: Optional[Path] = None
    ) -> None:
        """Construct an instance."""
        self.fields = fields
        self._cache_dir = cache_dir
        self._field_input_types = {
            field: get_generic_arguments(Field, field.__class__)[0]
            for _, field in fields
        }

    def pre_tensorize(self, inputs: _T) -> None:
        """
        Compute things before the tensorization itself.

        For example, fill a vocabulary object. Does nothing by default.

        :param inputs: Sample to use for the pre-tensorization.
        """
        for _, field in self.fields:
            field.pre_tensorize(self._select_input(field, inputs))

    def tensorize(self, inputs: _T) -> Dict[str, Any]:
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

    def save(self) -> None:
        if not self._cache_dir:
            raise RuntimeError(
                f"Trying to save a resource but the cache dir was not set."
            )
        self._cache_dir.mkdir(parents=True, exist_ok=True)
        with bz2_open(self._cache_dir / self._pickle_filename, "wb") as fh:
            pickle_dump(self.fields, fh)

    @classmethod
    def load_or_create(
        cls, fields: List[Tuple[str, Field]], cache_dir: Path
    ) -> "Instance":
        cache_path = cache_dir / cls._pickle_filename
        if cache_path.is_file():
            with bz2_open(cache_path, "rb") as fh:
                return cls(fields=pickle_load(fh), cache_dir=cache_dir)
        else:
            return Instance(fields=fields, cache_dir=cache_dir)

    def _select_input(self, field: Field, inputs: _T) -> Any:
        field_inputs_cls = self._field_input_types[field]
        if isinstance(inputs, field_inputs_cls):
            return inputs
        return inputs[field_inputs_cls]  # type: ignore
