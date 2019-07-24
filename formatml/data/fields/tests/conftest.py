from pathlib import Path

from pytest import fixture

from formatml.parsing.java_parser import JavaParser
from formatml.parsing.parser import Nodes


@fixture(scope="session")
def nodes() -> Nodes:
    parser = JavaParser()
    return parser.parse(Path(__file__).parent / "data", Path("Test.java"))


@fixture(scope="session")
def other_nodes() -> Nodes:
    parser = JavaParser()
    return parser.parse(Path(__file__).parent / "data", Path("OtherTest.java"))
