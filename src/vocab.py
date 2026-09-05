"""Loading the model vocabulary and finding special-purpose tokens."""

import json

NUMERIC_CHARS = set("0123456789.-")
BOUNDARY_TEXTS = {",", "}", " ", "\n"}


def load_vocab(path: str) -> dict[int, str]:
    """Load a vocab file into {token_id: token_text}."""

    with open(path, "r", encoding="utf-8") as file:
        data = json.load(file)

    return {int(token_id): str(token) for token, token_id in data.items()}


def token_sets(id_to_text: dict[int, str]) -> tuple:
    """Return (numeric token ids, boundary token ids, {'true'/'false': id})."""

    numeric_ids: set[int] = set()
    boundary_ids: set[int] = set()
    bool_ids: dict[str, int] = {}

    for token_id, token in id_to_text.items():
        if token and all(char in NUMERIC_CHARS for char in token):
            numeric_ids.add(token_id)
        if token in BOUNDARY_TEXTS:
            boundary_ids.add(token_id)
        if token in ("true", "false") and token not in bool_ids:
            bool_ids[token] = token_id

    return numeric_ids, boundary_ids, bool_ids
