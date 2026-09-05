"""Reading and writing the JSON files."""

from pydantic import BaseModel, ValidationError
from typing import Any
import json
import os


def read_json_list(path: str, model: type[BaseModel]) -> list[Any]:
    """Read a JSON array from `path` and validate each item with `model`."""

    try:
        with open(path, "r", encoding="utf-8") as file:
            raw = file.read()
    except FileNotFoundError:
        raise ValueError(f"missing input file: {path}")
    except OSError as e:
        raise ValueError(f"cannot read {path}: {e}")

    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        raise ValueError("invalid JSON")

    if not isinstance(data, list):
        raise ValueError(f"{path} must contain a JSON array")

    try:
        return [model(**item) for item in data]
    except ValidationError as e:
        raise ValueError(f"invalid data in {path}: " + e.errors()[0]["msg"])


def write_json(path: str, data: Any) -> None:
    """Write `data` as pretty JSON to `path`, creating folders as needed."""

    try:
        folder = os.path.dirname(path)
        if folder:
            os.makedirs(folder, exist_ok=True)
        with open(path, "w", encoding="utf-8") as file:
            json.dump(data, file, ensure_ascii=False, indent=2)
            file.write("\n")

    except OSError as e:
        raise ValueError(f"cannot write {path}: {e}")
