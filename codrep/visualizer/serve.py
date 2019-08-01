from argparse import ArgumentParser
from pathlib import Path
from typing import Any, Dict, List, Tuple

from flask import Flask, jsonify, render_template, request, Response
from flask_cors import CORS
from waitress import serve


def _list_tasks(path: Path) -> Tuple[List[str], List[int]]:
    if not path.is_dir():
        return [], []
    labels_path = path / "out.txt"
    if not labels_path.is_file():
        return [], []
    with labels_path.open(mode="r", encoding="utf8") as fh:
        labels = [int(line) - 1 for line in fh if line.strip()]
    file_names = set(p.name for p in path.iterdir())
    task_names = ["%d.txt" % i for i in range(len(labels))]
    if not file_names.issuperset(task_names):
        return [], []
    return task_names, labels


def _parse_output(path: Path, task_names: List[str]) -> Dict[str, List[int]]:
    parsed_output = {}
    task_names_set = set(task_names)
    with path.open("r", encoding="utf8") as fh:
        for line in fh:
            fields = line.split(" ")
            try:
                if not fields:
                    continue
                task_name = Path(fields[0]).name
                if task_name not in task_names_set:
                    continue
                parsed_output[task_name] = [int(v) - 1 for v in fields[1:]]
            except ValueError:
                continue
    return parsed_output


def _create_app(*, data: Path, results: Path) -> Flask:
    app = Flask(__name__, static_folder="build/static", template_folder="build")
    CORS(app)

    task_names, error_offsets = _list_tasks(data)
    parsed_output = _parse_output(results, task_names)

    @app.route("/")
    def index() -> Any:
        return render_template("index.html")

    @app.route("/api/tasks", methods=["GET"])
    def tasks() -> Response:
        return jsonify(dict(dataset=str(data), tasks=task_names))

    @app.route("/api/task", methods=["POST"])
    def task() -> Response:
        task_index = request.get_json()["task"]
        task_name = "%d.txt" % task_index
        return jsonify(
            dict(
                content=(data / task_name).read_text(encoding="utf8"),
                ranking=parsed_output[task_name],
                error_offset=error_offsets[task_index],
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
        "--port",
        type=int,
        default=5001,
        help="Port to use to run the server. The React app expects it to be 5001.",
    )
    args = parser.parse_args()

    app = _create_app(data=args.data, results=args.results)

    serve(app, host="0.0.0.0", port=args.port)


if __name__ == "__main__":
    main()
