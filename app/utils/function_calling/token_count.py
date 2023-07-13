from tiktoken import encoding_for_model, get_encoding

from app.models.function_calling.base import FunctionCall


def get_num_tokens_from_functions(
    functions: list[FunctionCall], model: str = "gpt-3.5-turbo"
) -> int:
    """Return the number of tokens used by a list of functions."""
    try:
        encoding = encoding_for_model(model)
    except KeyError:
        print("Warning: model not found. Using cl100k_base encoding.")
        encoding = get_encoding("cl100k_base")

    num_tokens = 0
    for _function in functions:
        function = _function.to_dict()
        function_tokens = len(encoding.encode(function["name"]))
        if "description" in function:
            function_tokens += len(encoding.encode(function["description"]))

        if "parameters" in function:
            parameters = function["parameters"]
            if "properties" in parameters:
                for name, property in parameters["properties"].items():
                    function_tokens += len(encoding.encode(name))
                    for field in property:
                        if field == "type":
                            function_tokens += 2
                            function_tokens += len(
                                encoding.encode(property["type"])
                            )
                        elif (
                            field == "description"
                            and "description" in property
                        ):
                            function_tokens += 2
                            function_tokens += len(
                                encoding.encode(property["description"])
                            )
                        elif field == "enum" and "enum" in property:
                            function_tokens -= 3
                            for o in property["enum"]:
                                function_tokens += 3
                                function_tokens += len(encoding.encode(str(o)))
                        else:
                            print(f"Warning: not supported field {field}")
                function_tokens += 11

        num_tokens += function_tokens

    num_tokens += 12
    return num_tokens
