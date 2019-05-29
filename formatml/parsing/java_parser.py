from bblfsh import Node as BblfshNode

from formatml.parsing.parser import Parser
from formatml.utils.registrable import register


_reserved = [
    "~",
    "}",
    "||",
    "|=",
    "|",
    "{",
    "while",
    "volatile",
    "void",
    "try",
    "true",
    "transient",
    "throws",
    "throw",
    "this",
    "synchronized",
    "switch",
    "super",
    "strictfp",
    "static",
    "short",
    "return",
    "public",
    "protected",
    "private",
    "package",
    "null",
    "new",
    "native",
    "long",
    "interface",
    "int",
    "instanceof",
    "import",
    "implements",
    "if",
    "goto",
    "for",
    "float",
    "finally",
    "final",
    "false",
    "extends",
    "enum",
    "else",
    "double",
    "do",
    "default",
    "continue",
    "const",
    "class",
    "char",
    "catch",
    "case",
    "byte",
    "break",
    "boolean",
    "assert",
    "abstract",
    "^=",
    "^",
    "]",
    "[",
    "@",
    "?",
    ">>>=",
    ">>>",
    ">>=",
    ">>",
    ">=",
    ">",
    "==",
    "=",
    "<=",
    "<<=",
    "<<",
    "<",
    ";",
    ":",
    "/=",
    "//",
    "/**",
    "/*",
    "/",
    ".",
    "-=",
    "--",
    "-",
    ",",
    "+=",
    "++",
    "+",
    "*=",
    "*/",
    "*",
    ")",
    "(",
    "&=",
    "&&",
    "&",
    "%=",
    "%",
    "!=",
    "!",
]


def _remove_children(bblfsh_node: BblfshNode) -> None:
    try:
        while True:
            bblfsh_node.children.pop()
    except IndexError:
        return bblfsh_node


def _exclude_if_empty(bblfsh_node: BblfshNode) -> None:
    if not bblfsh_node.children:
        return None
    return bblfsh_node


_uast_fixers = dict(
    BlockComment=_remove_children,
    Javadoc=_remove_children,
    LineComment=_remove_children,
    Dimension=_exclude_if_empty,
    Block=_exclude_if_empty,
    AnonymousClassDeclaration=_exclude_if_empty,
)


@register(cls=Parser, name="java")
class JavaParser(
    Parser,
    bblfsh_language="java",
    reserved=_reserved,
    uast_fixers=_uast_fixers,
    convert_to_utf8=False,
):
    """Parse files into list of nodes."""

    pass
