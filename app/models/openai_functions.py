from app.models.base_models import OpenAIFunction, OpenAIFunctionParameter


class OpenAIFunctions:
    CLICK_LINK: dict = OpenAIFunction(
        name="click_link",
        description="Click on the link provided to read more about it.",
        parameters=[
            OpenAIFunctionParameter(
                name="link",
                type=str,
            ),
        ],
        required=["link"],
    ).to_dict()
    ANSWERABLE_OR_NOT: dict = OpenAIFunction(
        name="answerable_or_not",
        description="Determine if you can answer the user's question.",
        parameters=[
            OpenAIFunctionParameter(
                name="answerable",
                type=bool,
                enum=[True, False],
            ),
        ],
        required=["answerable"],
    ).to_dict()
    WEB_BROWSING: dict = OpenAIFunction(
        name="web_browsing",
        description="Perform web browsing to answer the user's question.",
        parameters=[
            OpenAIFunctionParameter(
                name="action",
                type=str,
                description=(
                    "Your action to take. Select `finish_browsing` if you can answer the user's quest"
                    "ion or there's no relevant information. Select `click_link` only if you need to "
                    "click on a link to gather more information."
                ),
                enum=["finish_browsing", "click_link"],
            ),
            OpenAIFunctionParameter(
                name="link_to_click",
                description="The link to click on if you selected `click_link` as your action.",
                type=str,
            ),
        ],
        required=["action", "link_to_click"],
    ).to_dict()
    WEB_SEARCH: dict = OpenAIFunction(
        name="web_search",
        description="Perform a web search for a user's question.",
        parameters=[
            OpenAIFunctionParameter(
                name="query_to_search",
                type=str,
                description="A generalized query to return sufficiently relevant results when searching the web.",
            ),
        ],
        required=["query_to_search"],
    ).to_dict()
    VECTORSTORE_SEARCH: dict = OpenAIFunction(
        name="web_search",
        description="Performs a vector similarity-based search for a user's question.",
        parameters=[
            OpenAIFunctionParameter(
                name="query_to_search",
                type=str,
                description="hypothetical answer to facilitate searching in the Vector database.",
            ),
        ],
        required=["query_to_search"],
    ).to_dict()
