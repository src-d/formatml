from typing import TypeVar

from formatml.data.fields.field import Field
from formatml.parser import NodesSample


_T = TypeVar("_T")


class GraphField(Field[NodesSample, _T]):
    pass
