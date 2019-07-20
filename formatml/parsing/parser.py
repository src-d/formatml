from difflib import context_diff as unified_diff
from itertools import islice
from logging import getLogger
from os import environ
from pathlib import Path
from re import compile as re_compile, escape as re_escape
from typing import Any, Callable, Dict, List, NamedTuple, Optional, Set

from bblfsh import BblfshClient, Node as BblfshNode, role_name
from numpy import array, int32, uint32, unicode_


FORMATTING_INTERNAL_TYPE = "Formatting"
FORMATTING_ROLE = "FORMATTING"


class ParsingException(Exception):
    """Exception thrown in case of parsing failure."""

    pass


class Node:
    """Replacement for bblfsh.Node to ease parent navigation."""

    def __init__(
        self,
        *,
        token: Optional[str],
        internal_type: str,
        roles: List[str],
        parent: Optional["Node"],
        start: Optional[int],
        end: Optional[int],
    ):
        """
        Construct a Node (bblfsh.Node wrapper).

        :param token: Token that the Node represents.
        :param internal_type: Native type of the node. Formatting for formatting nodes.
        :param roles: List of Babelfish roles.
        :param parent: Parent of the node.
        :param start: Starting offset of the node in the parsed file.
        :param end: Ending offset of the node in the parsed file.
        """
        self.token = token
        self.internal_type = internal_type
        self.roles = roles
        self.parent = parent
        self.start = start
        self.end = end

    def __repr__(self) -> str:
        return (
            f"Node(token={self.token}, "
            f"internal_type={self.internal_type}, "
            f"roles={self.roles}, "
            f"parent={id(self.parent)}, "
            f"start={self.start}, "
            f"end={self.end})"
        )


class Nodes(NamedTuple):
    """Utilities for lists of nodes."""

    nodes: List[Node]
    node_index: Dict[int, int]
    token_indexes: List[int]

    def to_tree(self, file_content: str) -> Dict[str, Any]:
        """
        Convert a list of nodes into a tree serializable by asdf.

        :param file_content: Content of the file from which the nodes were extracted.
        :return: Dictionary serializable by asdf.
        """
        roles_offset = 0
        roles_offsets = []
        for node in self.nodes:
            roles_offsets.append(roles_offset)
            roles_offset += len(node.roles)
        for node in self.nodes:
            if node.internal_type == "WhiteSpace":
                assert node.token.isspace()

        return {
            "file_content": array([file_content], dtype=unicode_),
            "internal_types": array(
                [node.internal_type for node in self.nodes], dtype=unicode_
            ),
            "roles_list": array(
                [role for node in self.nodes for role in node.roles], dtype=unicode_
            ),
            "roles_offsets": array(roles_offsets, dtype=uint32),
            "parents": array(
                [self.node_index.get(id(node.parent), -1) for node in self.nodes],
                dtype=int32,
            ),
            "starts": array([node.start for node in self.nodes], dtype=uint32),
            "ends": array([node.end for node in self.nodes], dtype=uint32),
            "token_node_indexes": array(self.token_indexes, dtype=uint32),
        }

    @staticmethod
    def from_token_nodes(token_nodes: List[Node]) -> "Nodes":
        all_nodes = []
        seen: Set[int] = set()
        for node in token_nodes:
            current = node
            while current is not None:
                if id(current) not in seen:
                    seen.add(id(current))
                    all_nodes.append(current)
                current = current.parent
        node_to_index = {id(node): i for i, node in enumerate(all_nodes)}
        token_node_indexes = [
            node_to_index[id(token_node)] for token_node in token_nodes
        ]
        return Nodes(
            nodes=all_nodes, node_index=node_to_index, token_indexes=token_node_indexes
        )

    @staticmethod
    def from_tree(tree: Dict[str, Any]) -> "Nodes":
        """
        Convert an asdf tree into a list of nodes.

        :param tree: Tree to convert.
        :return: Nodes corresponding to the tree and the indexes of formatting nodes.
        """
        file_content = tree["file_content"][0]
        roles = []
        previous_roles_offset = 0
        for roles_offset in tree["roles_offsets"][1:]:
            roles.append(tree["roles_list"][previous_roles_offset:roles_offset])
            previous_roles_offset = roles_offset
        if tree["roles_offsets"].shape[0]:
            roles.append(tree["roles_list"][previous_roles_offset:])
        all_nodes = []
        token_node_indexes = frozenset(tree["token_node_indexes"])
        for i, (start, end, internal_type, roles) in enumerate(
            zip(tree["starts"], tree["ends"], tree["internal_types"], roles)
        ):
            all_nodes.append(
                Node(
                    start=int(start),
                    end=int(end),
                    roles=roles,
                    parent=None,
                    internal_type=internal_type,
                    token=file_content[start:end] if i in token_node_indexes else "",
                )
            )
        for node, parent_index in zip(all_nodes, map(int, tree["parents"])):
            node.parent = all_nodes[parent_index] if parent_index >= 0 else None
        for node in all_nodes:
            if node.internal_type == "WhiteSpace":
                assert node.token.isspace()
        node_index = {id(node): i for i, node in enumerate(all_nodes)}
        return Nodes(
            all_nodes, node_index, [i for i in map(int, tree["token_node_indexes"])]
        )


class BblfshNodeConverter:
    """Convert `BblfshNode`-s to `Node`-s (and handle bytes-unicode conversion)."""

    def __init__(self, file_content: str, convert_to_utf8: bool):
        """Contruct a converter."""
        self.file_content = file_content
        self.convert_to_utf8 = convert_to_utf8
        self.binary_to_str: Dict[int, int] = {}
        current_offset = 0
        for i, char in enumerate(self.file_content):
            self.binary_to_str[current_offset] = i
            current_offset += len(char.encode("utf-8", errors="replace"))
        self.binary_to_str[current_offset] = len(self.file_content)

    def bblfsh_node_to_node(
        self, bblfsh_node: BblfshNode, parent: Optional[Node]
    ) -> Node:
        """Create a `Node` given a `BblfshNode` and an optional parent."""
        position = bool(
            bblfsh_node.start_position.offset or bblfsh_node.end_position.offset
        )
        if position:
            start = bblfsh_node.start_position.offset
            end = bblfsh_node.end_position.offset
            if self.convert_to_utf8:
                start = self.binary_to_str[start]
                end = self.binary_to_str[end]
            token = self.file_content[start:end]
        else:
            start = None
            end = None
            token = bblfsh_node.token
            # Workaround https://github.com/bblfsh/javascript-driver/issues/65
            if not token and bblfsh_node.internal_type == "StringLiteralTypeAnnotation":
                token = bblfsh_node.properties["value"]
        return Node(
            token=token,
            internal_type=bblfsh_node.internal_type,
            roles=[role_name(role_id) for role_id in bblfsh_node.roles],
            parent=parent,
            start=start,
            end=end,
        )


class Parser:
    """Parse files into list of nodes."""

    def __init_subclass__(
        cls,
        bblfsh_language: str,
        reserved: List[str],
        uast_fixers: Optional[Dict[str, Callable[[BblfshNode], None]]] = None,
        convert_to_utf8: bool = True,
    ) -> None:
        cls._bblfsh_language = bblfsh_language
        cls._parser_reserved = re_compile(
            "|".join(re_escape(i) for i in sorted(reserved, reverse=True))
        )
        cls._parser_space = re_compile(r"\s+")
        cls._uast_fixers = uast_fixers if uast_fixers else {}
        cls._convert_to_utf8 = convert_to_utf8
        cls._logger = getLogger(cls.__name__)

    def __init__(
        self,
        bblfshd_endpoint: str = environ.get("BBLFSHD_ENDPOINT", "0.0.0.0:9432"),
        split_formatting: bool = False,
    ) -> None:
        """Construct a parser."""
        for attr in [
            "_bblfsh_language",
            "_parser_reserved",
            "_parser_space",
            "_uast_fixers",
        ]:
            if not hasattr(self, attr):

                raise NotImplementedError(
                    f"The {self.__class__.__name__} is a base class and should not be "
                    "used directly."
                )
        self._bblfsh_client = BblfshClient(bblfshd_endpoint)
        self._split_formatting = split_formatting

    @property
    def split_formatting(self) -> bool:
        return self._split_formatting

    def parse(self, repository_path: Path, file_path: Path) -> Nodes:
        """
        Parse a file into a list of `Node`s.

        :param repository_path: Path of the folder that contains the file to parse.
        :param file_path: Path of the file to parse.
        :return: List of parsed `Node`s.
        """
        response = self._bblfsh_client.parse(
            str(repository_path / file_path), language=self._bblfsh_language
        )
        if response.status != 0:
            self._logger.warn(
                "Could not process file %s, errors: %s",
                file_path,
                "; ".join(response.errors),
            )
            raise ParsingException(
                f"Could not process file {file_path}, "
                f"errors: {'; '.join(response.errors)}"
            )
        file_content = (repository_path / file_path).read_text(
            encoding="utf-8", errors="replace"
        )
        bblfsh_node_converter = BblfshNodeConverter(
            file_content, convert_to_utf8=self._convert_to_utf8
        )
        root_node = bblfsh_node_converter.bblfsh_node_to_node(response.uast, None)
        to_visit = [(response.uast, root_node)]
        non_formatting_tokens = []
        while to_visit:
            current_bblfsh_node, current_node = to_visit.pop()
            if current_bblfsh_node.internal_type in self._uast_fixers:
                current_bblfsh_node = self._uast_fixers[
                    current_bblfsh_node.internal_type
                ](current_bblfsh_node)
                if current_bblfsh_node is None:
                    continue
            to_visit.extend(
                (
                    bblfsh_child,
                    bblfsh_node_converter.bblfsh_node_to_node(
                        bblfsh_child, current_node
                    ),
                )
                for bblfsh_child in current_bblfsh_node.children
            )
            if (
                current_node.token
                and not current_bblfsh_node.children
                and (current_node.start is not None and current_node.end is not None)
            ):
                non_formatting_tokens.append(current_node)
        sentinel = Node(
            token=None,
            internal_type="Sentinel",
            roles=[],
            parent=None,
            start=len(file_content),
            end=len(file_content),
        )
        non_formatting_tokens.append(sentinel)

        pos = 0
        tokens = []
        for node in sorted(non_formatting_tokens, key=lambda n: n.start):
            if node.start < pos:
                continue
            if node.start > pos:
                sumlen = 0
                diff = file_content[pos : node.start]
                additional_nodes = []
                for match in self._parser_reserved.finditer(diff):
                    token = match.group()
                    additional_nodes.append(
                        Node(
                            start=match.start() + pos,
                            end=match.end() + pos,
                            token=token,
                            parent=None,
                            internal_type=token.title(),
                            roles=[match.group().upper()],
                        )
                    )
                    sumlen += len(token)
                for match in self._parser_space.finditer(diff):
                    token = match.group()
                    assert token.isspace()
                    additional_nodes.append(
                        Node(
                            start=match.start() + pos,
                            end=match.end() + pos,
                            token=token,
                            parent=None,
                            internal_type=FORMATTING_INTERNAL_TYPE,
                            roles=[FORMATTING_ROLE],
                        )
                    )
                    sumlen += len(token)
                if sumlen != node.start - pos:
                    self._logger.warn(f"missed some imaginary tokens: {diff}")
                    raise ParsingException(f"missed some imaginary tokens: {diff}")
                tokens.extend(sorted(additional_nodes, key=lambda n: n.start))
            if node is sentinel:
                break
            tokens.append(node)
            pos = node.end

        tokens = self._augment_tokens(tokens)

        closest_left_node = None
        for i, token_node in enumerate(tokens):
            if token_node.parent is not None:
                closest_left_node = token_node
            else:
                found_parent = self._find_parent(i, tokens, closest_left_node)
                token_node.parent = (
                    found_parent if found_parent is not None else root_node
                )

        if self._split_formatting:
            tokens = self._perform_split_formatting(tokens)

        reconstructed_file_content = "".join(node.token for node in tokens)

        if file_content != reconstructed_file_content:
            diff = "".join(
                unified_diff(
                    file_content.splitlines(keepends=True),
                    reconstructed_file_content.splitlines(keepends=True),
                    fromfile="original",
                    tofile="reconstructed",
                )
            )
            self._logger.warn("reconstructed file is not equal to original:\n%s", diff)
        return Nodes.from_token_nodes(tokens)

    def _augment_tokens(self, tokens: List[Node]) -> List[Node]:
        augmented_tokens = []

        if not tokens or tokens[0].internal_type != FORMATTING_INTERNAL_TYPE:
            augmented_tokens.append(
                Node(
                    start=0,
                    end=0,
                    token="",
                    parent=None,
                    internal_type=FORMATTING_INTERNAL_TYPE,
                    roles=[FORMATTING_ROLE],
                )
            )
        if tokens:
            augmented_tokens.append(tokens[0])

        for previous_token, next_token in zip(
            islice(tokens, 0, None), islice(tokens, 1, None)
        ):
            assert previous_token.end == next_token.start
            if (
                previous_token.internal_type != FORMATTING_INTERNAL_TYPE
                and next_token.internal_type != FORMATTING_INTERNAL_TYPE
            ):
                augmented_tokens.append(
                    Node(
                        start=previous_token.end,
                        end=previous_token.end,
                        token="",
                        parent=None,
                        internal_type=FORMATTING_INTERNAL_TYPE,
                        roles=[FORMATTING_ROLE],
                    )
                )
            augmented_tokens.append(next_token)

        if tokens and tokens[-1].internal_type != FORMATTING_INTERNAL_TYPE:
            augmented_tokens.append(
                Node(
                    start=tokens[-1].end,
                    end=tokens[-1].end,
                    token="",
                    parent=None,
                    internal_type=FORMATTING_INTERNAL_TYPE,
                    roles=[FORMATTING_ROLE],
                )
            )
        return augmented_tokens

    @staticmethod
    def _find_parent(
        node_index: int, nodes: List[Node], closest_left_node: Optional[Node]
    ) -> Optional[Node]:
        """
        Compute a node's parent as the LCA of the closest left and right nodes.

        :param node_index: Index of the node for which to find a parent.
        :param nodes: Sequence of token `Node`-s.
        :param closest_left_node: Closest node on the left with a true parent.
        :return: The Node of the found parent or None if no parent was found.
        """
        if closest_left_node is None:
            return None
        left_ancestor_ids = set()
        current_left_ancestor = closest_left_node.parent
        while current_left_ancestor is not None:
            left_ancestor_ids.add(id(current_left_ancestor))
            current_left_ancestor = current_left_ancestor.parent

        for future_node in nodes[node_index + 1 :]:
            if future_node.parent is not None:
                break
        else:
            return None
        current_right_ancestor = future_node.parent
        while current_right_ancestor is not None:
            if id(current_right_ancestor) in left_ancestor_ids:
                return current_right_ancestor
            current_right_ancestor = current_right_ancestor.parent
        return None

    def _perform_split_formatting(self, nodes: List[Node]) -> List[Node]:
        """
        Split each formatting node into a list of one node per character.

        :param nodes: Sequence of token `Node`-s.
        :return: The new sequence, with split formatting nodes.
        """
        new_nodes = []
        for node in nodes:
            if node.internal_type == FORMATTING_INTERNAL_TYPE and node.token:
                for i, char in enumerate(node.token):
                    new_nodes.append(
                        Node(
                            token=char,
                            internal_type=node.internal_type,
                            roles=node.roles,
                            parent=node.parent,
                            start=node.start + i,
                            end=node.start + i + 1,
                        )
                    )
            else:
                new_nodes.append(node)
        return new_nodes

    def __del__(self) -> None:
        if self._bblfsh_client:
            self._bblfsh_client._channel.close()
            self._bblfsh_client._channel = self._bblfsh_client._stub = None
