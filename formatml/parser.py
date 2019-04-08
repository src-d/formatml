from difflib import unified_diff
from itertools import islice
from logging import getLogger
from pathlib import Path
from re import compile as re_compile, escape as re_escape
from typing import Any, Dict, List, NamedTuple, Optional, Set

from bblfsh import BblfshClient, Node as BblfshNode, role_name
from numpy import array, int32, uint32, unicode_


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


class NodesSample(NamedTuple):
    """Utilities for lists of nodes."""

    nodes: List[Node]
    node_index: Dict[int, int]
    token_indexes: List[int]

    @staticmethod
    def to_tree(token_nodes: List[Node], file_content: str) -> Dict[str, Any]:
        """
        Convert a list of nodes into a tree serializable by asdf.

        :param token_nodes: List of nodes to convert.
        :param file_content: Content of the file from which the nodes were extracted.
        :return: Dictionary serializable by asdf.
        """
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
        roles_offset = 0
        roles_offsets = []
        for node in all_nodes:
            roles_offsets.append(roles_offset)
            roles_offset += len(node.roles)
        for node in all_nodes:
            if node.internal_type == "WhiteSpace":
                assert node.token.isspace()

        return {
            "file_content": array([file_content], dtype=unicode_),
            "internal_types": array(
                [node.internal_type for node in all_nodes], dtype=unicode_
            ),
            "roles_list": array(
                [role for node in all_nodes for role in node.roles], dtype=unicode_
            ),
            "roles_offsets": array(roles_offsets, dtype=uint32),
            "parents": array(
                [node_to_index.get(id(node.parent), -1) for node in all_nodes],
                dtype=int32,
            ),
            "starts": array([node.start for node in all_nodes], dtype=uint32),
            "ends": array([node.end for node in all_nodes], dtype=uint32),
            "token_node_indexes": array(token_node_indexes, dtype=uint32),
        }

    @staticmethod
    def from_tree(tree: Dict[str, Any]) -> "NodesSample":
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
        return NodesSample(
            all_nodes, node_index, [i for i in map(int, tree["token_node_indexes"])]
        )


class BblfshNodeConverter:
    """Convert `BblfshNode`-s to `Node`-s (and handle bytes-unicode conversion)."""

    def __init__(self, file_content: str):
        """Contruct a converter."""
        self.file_content = file_content
        self.binary_to_str: Dict[int, int] = {}
        current_offset = 0
        for i, char in enumerate(self.file_content):
            self.binary_to_str[current_offset] = i
            current_offset += len(char.encode("utf-8", errors="strict"))
        self.binary_to_str[current_offset] = len(self.file_content)

    def bblfsh_node_to_node(
        self, bblfsh_node: BblfshNode, parent: Optional[Node]
    ) -> Node:
        """Create a `Node` given a `BblfshNode` and an optional parent."""
        position = bool(
            bblfsh_node.start_position.offset or bblfsh_node.end_position.offset
        )
        if position:
            start = self.binary_to_str[bblfsh_node.start_position.offset]
            end = self.binary_to_str[bblfsh_node.end_position.offset]
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

    _logger = getLogger(__name__)

    def __init__(self) -> None:
        """Construct a parser."""
        self.reserved = [
            "abstract",
            "any",
            "as",
            "async",
            "await",
            "boolean",
            "break",
            "byte",
            "case",
            "catch",
            "char",
            "class",
            "const",
            "continue",
            "debugger",
            "declare",
            "default",
            "delete",
            "do",
            "double",
            "else",
            "enum",
            "export",
            "exports",
            "extends",
            "false",
            "final",
            "finally",
            "float",
            "for",
            "from",
            "function",
            "get",
            "goto",
            "of",
            "opaque",
            "if",
            "implements",
            "import",
            "in",
            "instanceof",
            "int",
            "interface",
            "let",
            "long",
            "mixed",
            "module",
            "native",
            "new",
            "number",
            "null",
            "package",
            "private",
            "protected",
            "public",
            "return",
            "set",
            "short",
            "static",
            "string",
            "super",
            "switch",
            "synchronized",
            "this",
            "throw",
            "throws",
            "transient",
            "true",
            "try",
            "type",
            "typeof",
            "yield",
            "var",
            "void",
            "volatile",
            "while",
            "with",
            "+",
            "-",
            "*",
            "/",
            "%",
            "++",
            "--",
            "=",
            "+=",
            "-=",
            "/=",
            "%=",
            "==",
            "===",
            "!=",
            "!==",
            ">",
            "<",
            ">=",
            "<=",
            "?",
            ":",
            "&&",
            "||",
            "!",
            "&",
            "|",
            "~",
            "^",
            ">>",
            "<<",
            "(",
            ")",
            "{",
            "}",
            ".",
            "...",
            "[",
            "]",
            ">>>",
            ",",
            ";",
            "'",
            '"',
            "`",
            "${",
            "\\",
        ]
        # The longest keywords should come first for the regex below to be usable with
        # finditer
        self.reserved.sort(reverse=True)
        self.parser_reserved = re_compile("|".join(re_escape(i) for i in self.reserved))
        self.parser_space = re_compile(r"\s+")

    def parse(
        self,
        repository_path: Path,
        file_path: Path,
        bblfsh_client: BblfshClient,
        formatting_internal_type: str,
    ) -> List[Node]:
        """
        Parse a file into a list of `Node`s.

        :param repository_path: Path of the folder that contains the file to parse.
        :param file_path: Path of the file to parse.
        :param bblfsh_client: Babelfish client to use for parsing.
        :param formatting_internal_type: New internal type to use for formatting nodes.
        :return: List of parsed `Node`s.
        """
        response = bblfsh_client.parse(
            str(repository_path / file_path), language="javascript"
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
        bblfsh_node_converter = BblfshNodeConverter(file_content)
        root_node = bblfsh_node_converter.bblfsh_node_to_node(response.uast, None)
        to_visit = [(response.uast, root_node)]
        non_formatting_tokens = []
        while to_visit:
            current_bblfsh_node, current_node = to_visit.pop()
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
                for match in self.parser_reserved.finditer(diff):
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
                for match in self.parser_space.finditer(diff):
                    token = match.group()
                    assert token.isspace()
                    additional_nodes.append(
                        Node(
                            start=match.start() + pos,
                            end=match.end() + pos,
                            token=token,
                            parent=None,
                            internal_type=formatting_internal_type,
                            roles=["FORMATTING"],
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

        tokens = self._augment_tokens(tokens, formatting_internal_type)

        closest_left_node = None
        for i, token_node in enumerate(tokens):
            if token_node.parent is not None:
                closest_left_node = token_node
            else:
                found_parent = self._find_parent(i, tokens, closest_left_node)
                token_node.parent = (
                    found_parent if found_parent is not None else root_node
                )

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
        return tokens

    @staticmethod
    def _augment_tokens(
        tokens: List[Node], formatting_internal_type: str
    ) -> List[Node]:
        augmented_tokens = []

        if tokens and tokens[0].internal_type != formatting_internal_type:
            augmented_tokens.append(
                Node(
                    start=0,
                    end=0,
                    token="",
                    parent=None,
                    internal_type=formatting_internal_type,
                    roles=["FORMATTING"],
                )
            )
            augmented_tokens.append(tokens[0])

        for previous_token, next_token in zip(
            islice(tokens, 0, None), islice(tokens, 1, None)
        ):
            assert previous_token.end == next_token.start
            if (
                previous_token.internal_type != formatting_internal_type
                and next_token.internal_type != formatting_internal_type
            ):
                augmented_tokens.append(
                    Node(
                        start=previous_token.end,
                        end=previous_token.end,
                        token="",
                        parent=None,
                        internal_type=formatting_internal_type,
                        roles=["FORMATTING"],
                    )
                )
            augmented_tokens.append(next_token)

        if tokens and tokens[-1].internal_type != formatting_internal_type:
            augmented_tokens.append(tokens[-1])
            augmented_tokens.append(
                Node(
                    start=tokens[-1].end,
                    end=tokens[-1].end,
                    token="",
                    parent=None,
                    internal_type=formatting_internal_type,
                    roles=["FORMATTING"],
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
