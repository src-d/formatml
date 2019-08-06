from argparse import ArgumentParser
from collections import defaultdict
from json import load as json_load
from pathlib import Path
from sys import stderr
from typing import Any, DefaultDict, Dict, List, Optional, Tuple

from flask import Flask, jsonify, render_template, request, Response
from flask_cors import CORS
from waitress import serve


def _list_tasks(path: Path) -> Tuple[List[str], List[int]]:
    if not path.is_dir():
        raise RuntimeError("Dataset path %s does not point to a directory" % path)
    labels_path = path / "out.txt"
    if not labels_path.is_file():
        raise RuntimeError("Did not find the ground truth file %s" % labels_path)
    with labels_path.open(mode="r", encoding="utf8") as fh:
        labels = [int(line) - 1 for line in fh if line.strip()]
    file_names = set(p.name for p in path.iterdir())
    task_names = ["%d.txt" % i for i in range(len(labels))]
    if not file_names.issuperset(task_names):
        raise RuntimeError(
            "Did not find all tasks in the given dataset. Missing: %s"
            % ", ".join(set(task_names) - file_names)
        )
    return task_names, labels


def _parse_output(path: Path, task_names: List[str]) -> Dict[str, List[int]]:
    parsed_output: DefaultDict[str, List[int]] = defaultdict(list)
    task_names_set = set(task_names)
    with path.open("r", encoding="utf8") as fh:
        for line in fh:
            fields = line.split(" ")
            try:
                if not fields:
                    continue
                task_name = Path(fields[0]).name
                if task_name not in task_names_set:
                    print("Warning: task %s was ignored" % task_name, file=stderr)
                    continue
                parsed_output[task_name] = [int(v) - 1 for v in fields[1:]]
            except ValueError:
                print(
                    "Warning: task %s output was not parsed correctly" % task_name,
                    file=stderr,
                )
                continue
    return parsed_output


def _parse_metadata(path: Optional[Path]) -> Dict[str, Any]:
    empty: Dict[str, Any] = dict(columns=[], metadata={})
    if path is None:
        return empty
    try:
        with path.open("r", encoding="utf8") as fh:
            metadata = json_load(fh)
    except Exception:
        print("Metadata file %s is not used" % path, file=stderr)
        return empty
    return dict(
        columns=metadata["columns"],
        metadata={
            Path(k).name: {o: list(map(str, w)) for o, w in v.items()}
            for k, v in metadata["metadata"].items()
        },
    )


def _create_app(*, data: Path, results: Path, metadata: Optional[Path]) -> Flask:
    app = Flask(__name__, static_folder="build/static", template_folder="build")
    CORS(app)

    task_names, error_offsets = _list_tasks(data)
    parsed_output = _parse_output(results, task_names)
    parsed_metadata = _parse_metadata(metadata)

    @app.route("/")
    def index() -> Any:
        return render_template("index.html")

    @app.route("/api/tasks", methods=["GET"])
    def tasks() -> Response:
        return jsonify(
            dict(
                dataset=str(data),
                numberOfTasks=len(task_names),
                metadata=parsed_metadata,
            )
        )

    @app.route("/api/task", methods=["POST"])
    def task() -> Response:
        task_index = request.get_json()["task"]
        task_name = "%d.txt" % task_index
        return jsonify(
            dict(
                content=(data / task_name).read_text(encoding="utf8"),
                ranking=parsed_output[task_name],
                errorOffset=error_offsets[task_index],
            )
        )

    return app


def main() -> None:
    parser = ArgumentParser()
    parser.add_argument(
        "--data",
        required=True,
        type=Path,
        help="Path to the CodRep 2019 results file for the given dataset.",
    )
    parser.add_argument(
        "--results",
        required=True,
        type=Path,
        help="Path to the CodRep 2019 results file for the given dataset.",
    )
    parser.add_argument(
        "--metadata",
        type=Path,
        help="Path to the CodRep 2019 metadata json for the given dataset.",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=5001,
        help="Port to use to run the server. The React app expects it to be 5001.",
    )
    args = parser.parse_args()
    app = _create_app(data=args.data, results=args.results, metadata=args.metadata)

    serve(app, host="0.0.0.0", port=args.port)


if __name__ == "__main__":
    main()
