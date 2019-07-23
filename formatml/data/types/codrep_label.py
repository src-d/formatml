from typing import Any, Dict, NamedTuple, Optional


class CodRepLabel(NamedTuple):
    error_index: Optional[int]
    n_formatting_nodes: int

    def to_tree(self) -> Dict[str, Any]:
        return dict(
            error_index=self.error_index, n_formatting_nodes=self.n_formatting_nodes
        )

    @staticmethod
    def from_tree(tree: Dict[str, Any]) -> "CodRepLabel":
        return CodRepLabel(
            error_index=tree["error_index"],
            n_formatting_nodes=tree["n_formatting_nodes"],
        )
