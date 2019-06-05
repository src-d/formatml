from os import environ
from pathlib import Path

from pytest import fixture

from formatml.parsing.java_parser import JavaParser
from formatml.parsing.parser import Nodes


@fixture(scope="session")
def nodes() -> Nodes:
    parser = JavaParser(bblfsh_endpoint=environ.get("BBLFSH_ENDPOINT", "0.0.0.0:9999"))
    return parser.parse(Path(__file__).parent / "data", Path("Test.java"))


@fixture(scope="session")
def other_nodes() -> Nodes:
    parser = JavaParser(bblfsh_endpoint=environ.get("BBLFSH_ENDPOINT", "0.0.0.0:9999"))
    return parser.parse(Path(__file__).parent / "data", Path("OtherTest.java"))
