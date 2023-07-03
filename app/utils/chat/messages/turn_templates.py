"""This module contains functions to extract information from chat turn templates."""

from re import DOTALL, compile
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from langchain import PromptTemplate


def shatter_chat_turn_prompt(
    *keys: str, chat_turn_prompt: "PromptTemplate"
) -> tuple[str, ...]:
    """Identify the chat turn template and return the shatter result.
    e.g. If template of chat_turn_prompt is "### {role}: {content} </s>"
    and keys are "role" and "content",
    then the result will be ('### ', "{role}", ': ', "{content}", ' </s>')."""
    pattern: str = "(.*)"
    kwargs: dict[str, str] = {}
    for key in keys:
        kwargs[key] = "{" + key + "}"
        pattern += f"({kwargs[key]})(.*)"
    search_result = compile(pattern, flags=DOTALL).match(
        chat_turn_prompt.format(**kwargs)
    )
    if search_result is None:
        raise ValueError(
            f"Invalid chat turn prompt: {chat_turn_prompt.format(**kwargs)}"
        )
    return search_result.groups()


def identify_end_of_string(*keys, chat_turn_prompt: "PromptTemplate") -> Optional[str]:
    """Identify the end of string in the chat turn prompt.
    e.g. If template of chat_turn_prompt is "### {role}: {content} </s>"
    then the result will be "</s>".
    If there is no end of string, then the result will be None."""
    return shatter_chat_turn_prompt(*keys, chat_turn_prompt=chat_turn_prompt)[
        -1
    ].strip()


if __name__ == "__main__":
    from langchain import PromptTemplate

    input_variables = ["role", "content"]
    for template, template_format in (
        ("### {role}: {content} </s>", "f-string"),
        ("### {{role}}: {{content}} </s>", "jinja2"),
    ):
        chat_turn_prompt = PromptTemplate(
            template=template,
            input_variables=input_variables,
            template_format=template_format,
        )
        print(
            "Shattered String:",
            shatter_chat_turn_prompt(
                *input_variables, chat_turn_prompt=chat_turn_prompt
            ),
        )  # ('### ', '{role}', ': ', '{content}', ' </s>')
        print(
            "End-of-String:",
            identify_end_of_string(*input_variables, chat_turn_prompt=chat_turn_prompt),
        )  # </s>
