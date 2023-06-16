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
    CONTROL_WEB_PAGE: dict = OpenAIFunction(
        name="control_web_page",
        description=(
            "Control the web page to read more information or stop reading. You have to evaluate "
            "the relevance of the information you read and decide whether to scroll down, go back, "
            "or pick."
        ),
        parameters=[
            OpenAIFunctionParameter(
                name="action",
                type=str,
                description=(
                    "Whether to scroll down, go back, or pick the result. Select `scroll_down` if you"
                    "must scroll down to read more information. Select `go_back` if you need to g"
                    "o back to the previous page to read other information. Select `pick` "
                    "only if you can provide a satisfactory answer to the user from the given information."
                ),
                enum=["scroll_down", "go_back", "pick"],
            ),
            OpenAIFunctionParameter(
                name="relevance_score",
                type=int,
                description=(
                    "A score that indicates how helpful the given context is in answering the user's "
                    "question. If the information is very relevant and sufficient to answer the user's"
                    " question, give it a score of 10; if the information is very irrelevant, give it a score of 0."
                ),
                enum=[score for score in range(11)],
            ),
        ],
        required=["action", "relevance_score"],
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
