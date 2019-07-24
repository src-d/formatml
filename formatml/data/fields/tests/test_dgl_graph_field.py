from typing import List

from formatml.data.fields.typed_dgl_graph_field import TypedDGLGraphField
from formatml.parsing.parser import Nodes


def _make_field(edge_types: List[str]) -> TypedDGLGraphField:
    return TypedDGLGraphField("dgl_typed_graph", "graph", edge_types)


def test_graph_field(nodes: Nodes) -> None:
    dgl_graph_field = _make_field(["parent"])
    output = dgl_graph_field.tensorize(nodes)
    assert len(output.edges_by_type) == 1


def test_no_edge_types(nodes: Nodes) -> None:
    dgl_graph_field = _make_field([])
    output = dgl_graph_field.tensorize(nodes)
    assert len(output.edges_by_type) == 0
    assert output.graph.number_of_edges() == 0


def test_symmetric_edge_types(nodes: Nodes) -> None:
    dgl_graph_field_parent = _make_field(["parent"])
    dgl_graph_field_child = _make_field(["child"])
    dgl_graph_field = _make_field(["child", "parent"])
    output_parent = dgl_graph_field_parent.tensorize(nodes)
    output_child = dgl_graph_field_child.tensorize(nodes)
    output = dgl_graph_field.tensorize(nodes)
    assert output.graph.number_of_edges() > 0
    assert output_parent.graph.number_of_edges() == output_child.graph.number_of_edges()
    assert output.graph.number_of_edges() == output_child.graph.number_of_edges() * 2


def test_collate(nodes: Nodes, other_nodes: Nodes) -> None:
    dgl_graph_field = _make_field(["child", "parent", "next_token", "previous_token"])
    output_1 = dgl_graph_field.tensorize(nodes)
    output_2 = dgl_graph_field.tensorize(other_nodes)
    collated = dgl_graph_field.collate([output_1, output_2])
    assert (
        collated.graph.number_of_edges()
        == output_1.graph.number_of_edges() + output_2.graph.number_of_edges()
    )
    assert (
        collated.graph.number_of_nodes()
        == output_1.graph.number_of_nodes() + output_2.graph.number_of_nodes()
    )
    assert sum(t.numel() for t in collated.edges_by_type) == sum(
        t.numel() for t in output_1.edges_by_type
    ) + sum(t.numel() for t in output_2.edges_by_type)
    assert all(
        (ec[e1.numel() :] >= output_1.graph.number_of_edges()).all()
        for e1, ec in zip(output_1.edges_by_type, collated.edges_by_type)
    )
