"""Pydantic models for the input/output JSON."""

from pydantic import BaseModel
from typing import Any


class Parameter(BaseModel):
    """One function parameter: just its JSON type."""

    type: str


class FunctionDef(BaseModel):
    """One callable function the model can pick between."""

    name: str
    description: str
    parameters: dict[str, Parameter]
    returns: dict[str, str] = {}


class Prompt(BaseModel):
    """One natural-language prompt to process."""

    prompt: str


class OutputEntry(BaseModel):
    """One generated function call, ready to write to the output file."""

    prompt: str
    name: str
    parameters: dict[str, Any]
