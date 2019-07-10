from typing import Any, Dict, List, NamedTuple, Optional

from numpy import array, uint32


class CodRepLabel(NamedTuple):
    formatting_indexes: List[int]
    error_index: Optional[int]
    n_nodes: int

    def to_tree(self) -> Dict[str, Any]:
        return dict(
            formatting_indexes=array(self.formatting_indexes, dtype=uint32),
            error_index=self.error_index,
            n_nodes=self.n_nodes,
        )

    @staticmethod
    def from_tree(tree: Dict[str, Any]) -> "CodRepLabel":
        return CodRepLabel(
            formatting_indexes=tree["formatting_indexes"].tolist(),
            error_index=tree["error_index"],
            n_nodes=tree["n_nodes"],
        )
