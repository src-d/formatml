from typing import TypeVar

from formatml.data.fields.field import Field
from formatml.parsing.parser import Nodes


_T = TypeVar("_T")


class GraphField(Field[Nodes, _T]):
    pass
