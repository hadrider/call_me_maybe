"""Constrained decoding for function calling."""

from llm_sdk import Small_LLM_Model  # type: ignore
from src.models import FunctionDef
import numpy as np
from typing import Any


def encode(model: Small_LLM_Model, text: str) -> list[int]:
    """Convert text into token IDs."""

    tokens = model.encode(text)
    return tokens.tolist()[0]


def decode_ids(model: Small_LLM_Model, ids: list[int]) -> str:
    """Convert token IDs back into text."""

    if not ids:
        return ""
    return str(model.decode(ids))


def next_logits(model: Small_LLM_Model, input_ids: list[int]) -> np.ndarray:
    """Get the model's scores for the next token."""

    return np.asarray(model.get_logits_from_input_ids(input_ids))


def masked_argmax(logits: np.ndarray, valid_ids: set[int]) -> int:
    """Return the best token among the allowed tokens."""

    mask = np.full(logits.shape, -np.inf)
    ids = [token_id for token_id in valid_ids if 0 <= token_id < len(logits)]

    if not ids:
        raise RuntimeError("no valid token was found")

    mask[ids] = logits[ids]
    return int(np.argmax(mask))


def select_function(model: Small_LLM_Model, prompt: str,
                    functions: list[FunctionDef]) -> FunctionDef:
    """Use the LLM to choose the correct function."""

    if len(functions) == 1:
        return functions[0]

    descriptions = "\n".join(f"- {function.name}: {function.description}"
                             for function in functions)

    context = ("Pick the function that answers the request.\n"
               f"{descriptions}\n"
               f"Request: {prompt}\n"
               '{"name": "')

    input_ids = encode(model, context)
    function_tokens = {function.name: encode(model, function.name)
                       for function in functions}
    candidates = functions
    position = 0

    while len(candidates) > 1:
        valid_ids = {function_tokens[function.name][position]
                     for function in candidates
                     if position < len(function_tokens[function.name])}

        token_id = masked_argmax(next_logits(model, input_ids), valid_ids)
        input_ids.append(token_id)

        candidates = [function for function in candidates
                      if (position < len(function_tokens[function.name])
                          and function_tokens[
                              function.name][position] == token_id)]
        position += 1
    return candidates[0]


def generate_number(model: Small_LLM_Model, input_ids: list[int],
                    numeric_ids: set[int], boundary_ids: set[int]) -> str:
    """Generate a number until a boundary token is produced."""

    valid_ids = numeric_ids | boundary_ids
    collected = []

    while True:
        token_id = masked_argmax(next_logits(model, input_ids), valid_ids)
        if token_id not in numeric_ids:
            break

        collected.append(token_id)
        input_ids.append(token_id)

    result = decode_ids(model, collected).strip()
    return result


def generate_boolean(model: Small_LLM_Model, input_ids: list[int],
                     bool_ids: dict[str, int]) -> str:
    """Generate either true or false."""

    if not bool_ids:
        return "false"

    token_id = masked_argmax(next_logits(model, input_ids),
                             set(bool_ids.values()))
    input_ids.append(token_id)
    if token_id == bool_ids.get("true"):
        return "true"

    return "false"


def generate_string(model: Small_LLM_Model, input_ids: list[int],
                    quote_id: int) -> str:
    """Generate a string until the closing quote is reached."""

    input_ids.append(quote_id)
    collected: list = []

    while True:
        token_id = int(np.argmax(next_logits(model, input_ids)))
        token_text = decode_ids(model, [token_id])

        if '"' in token_text:
            before_quote = token_text.split('"', 1)[0]
            return decode_ids(model, collected) + before_quote

        collected.append(token_id)
        input_ids.append(token_id)


def fill_parameters(model: Small_LLM_Model, prompt: str,
                    function: FunctionDef, numeric_ids: set[int],
                    boundary_ids: set[int], bool_ids: dict[str, int],
                    quote_id: int) -> dict[str, Any]:
    """Generate all parameters required by a function."""

    context = (f"Request: {prompt}\n"
               "Answer with a single JSON object, nothing else.\n"
               f'{{"name": "{function.name}", "parameters": {{')

    input_ids = encode(model, context)
    values: dict[str, Any] = {}
    parameters = list(function.parameters.items())

    for index, (name, parameter) in enumerate(parameters):
        input_ids += encode(model, f'"{name}": ')

        if parameter.type in ("number", "integer"):
            text = generate_number(model, input_ids, numeric_ids, boundary_ids)

            if parameter.type == "number":
                values[name] = float(text)
            else:
                values[name] = int(float(text))

        elif parameter.type == "boolean":
            text = generate_boolean(model, input_ids, bool_ids)
            values[name] = text == "true"

        else:
            values[name] = generate_string(model, input_ids, quote_id)

        if index < len(parameters) - 1:
            input_ids += encode(model, ", ")

    return values
