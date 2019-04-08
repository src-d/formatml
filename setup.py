from importlib.machinery import SourceFileLoader
from pathlib import Path
from sys import stderr
from types import ModuleType

from setuptools import find_packages, setup

try:
    import torch  # noqa: F401
except ImportError:
    print(
        "PyTorch should be installed. "
        "Please visit https://pytorch.org/ for instructions.",
        file=stderr,
    )

loader = SourceFileLoader("formatml", "./formatml/__init__.py")
formatml = ModuleType(loader.name)
loader.exec_module(formatml)

setup(
    name="formatml",
    version=formatml.__version__,  # type: ignore
    description="Formatting with meta-learning experiments.",
    long_description=(Path(__file__).parent / "README.md").read_text(encoding="utf-8"),
    long_description_content_type="text/markdown",
    author="source{d}",
    author_email="machine-learning@sourced.tech",
    python_requires=">=3.7.0",
    url="https://github.com/src-d/formatml",
    packages=find_packages(exclude=["tests"]),
    entry_points={"console_scripts": ["formatml=formatml.__main__:main"]},
    install_requires=[
        "coloredlogs",
        "jsonnet",
        "dgl",
        "bblfsh <3.0",
        "asdf",
        "dulwich",
        "tensorboardX",
        "tensorflow",
    ],
    include_package_data=True,
    license="Apache-2.0",
    classifiers=[
        "Development Status :: 2 - Pre-Alpha"
        "License :: OSI Approved :: Apache Software License",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Developers",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Code Generators",
        "Topic :: Software Development :: Pre-processors",
        "Topic :: Software Development :: Quality Assurance",
    ],
)
