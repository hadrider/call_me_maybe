"""call me maybe - turn natural language prompts into function calls."""

from llm_sdk import Small_LLM_Model  # type: ignore
from src.decode import encode, fill_parameters, select_function
from src.io_utils import read_json_list, write_json
from src.models import FunctionDef, Prompt
from src.vocab import load_vocab, token_sets
import argparse
import time


def main() -> None:
    """Read the input files, run constrained decoding, write the output
    file."""

    parser = argparse.ArgumentParser()
    parser.add_argument("--functions_definition",
                        default="data/input/functions_definition.json")
    parser.add_argument("--input",
                        default="data/input/function_calling_tests.json")
    parser.add_argument("--output",
                        default="data/output/function_calling_results.json")
    args = parser.parse_args()

    try:
        functions = read_json_list(args.functions_definition, FunctionDef)
        prompts = read_json_list(args.input, Prompt)
        if not functions or not prompts:
            raise ValueError("the file can't be empty")

        model = Small_LLM_Model()
        id_to_text = load_vocab(model.get_path_to_vocab_file())
        numeric_ids, boundary_ids, bool_ids = token_sets(id_to_text)
        quote_id = encode(model, '"')[0]

        results = []
        for item in prompts:
            func = select_function(model, item.prompt, functions)
            parameters = fill_parameters(model, item.prompt, func,
                                         numeric_ids, boundary_ids,
                                         bool_ids, quote_id)
            results.append({"prompt": item.prompt, "name": func.name,
                            "parameters": parameters})

        write_json(args.output, results)

    except (ValueError, RuntimeError, OSError) as e:
        print(f"Error: {e}")
        exit()


if __name__ == "__main__":
    time67 = time.time()
    main()
    print(time.time() - time67)
