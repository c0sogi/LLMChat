import json
from typing import Any, Dict, List, Optional

import aiohttp
import requests
from langchain.utils import get_from_dict_or_env
from pydantic import (
    BaseModel,
    Extra,
    Field,
    PrivateAttr,
    root_validator,
    validator,
)

from app.utils.api.duckduckgo import DDGS


def _get_default_params() -> dict:
    return {"language": "en", "format": "json"}


class SearxResults(dict):
    """Dict like wrapper around search api results."""

    _data = ""

    def __init__(self, data: str):
        """Take a raw result from Searx and make it into a dict like object."""
        json_data = json.loads(data)
        super().__init__(json_data)
        self.__dict__ = self

    def __str__(self) -> str:
        """Text representation of searx result."""
        return self._data

    @property
    def results(self) -> Any:
        """Silence mypy for accessing this field.

        :meta private:
        """
        return self.get("results")

    @property
    def answers(self) -> Any:
        """Helper accessor on the json result."""
        return self.get("answers")


class SearxSearchWrapper(BaseModel):
    """Wrapper for Searx API.

    To use you need to provide the searx host by passing the named parameter
    ``searx_host`` or exporting the environment variable ``SEARX_HOST``.

    In some situations you might want to disable SSL verification, for example
    if you are running searx locally. You can do this by passing the named parameter
    ``unsecure``. You can also pass the host url scheme as ``http`` to disable SSL.

    Example:
        .. code-block:: python

            from langchain.utilities import SearxSearchWrapper
            searx = SearxSearchWrapper(searx_host="http://localhost:8888")

    Example with SSL disabled:
        .. code-block:: python

            from langchain.utilities import SearxSearchWrapper
            # note the unsecure parameter is not needed if you pass the url scheme as
            # http
            searx = SearxSearchWrapper(searx_host="http://localhost:8888",
                                                    unsecure=True)


    """

    _result: SearxResults = PrivateAttr()
    searx_host: str = ""
    unsecure: bool = False
    params: dict = Field(default_factory=_get_default_params)
    headers: Optional[dict] = None
    engines: Optional[List[str]] = []
    categories: Optional[List[str]] = []
    query_suffix: Optional[str] = ""
    k: int = 10
    aiosession: Optional[Any] = None

    @validator("unsecure")
    def disable_ssl_warnings(cls, v: bool) -> bool:
        """Disable SSL warnings."""
        if v:
            # requests.urllib3.disable_warnings()
            try:
                import urllib3

                urllib3.disable_warnings()
            except ImportError as e:
                print(e)

        return v

    @root_validator()
    def validate_params(cls, values: Dict) -> Dict:
        """Validate that custom searx params are merged with default ones."""
        user_params = values["params"]
        default = _get_default_params()
        values["params"] = {**default, **user_params}

        engines = values.get("engines")
        if engines:
            values["params"]["engines"] = ",".join(engines)

        categories = values.get("categories")
        if categories:
            values["params"]["categories"] = ",".join(categories)

        searx_host = get_from_dict_or_env(values, "searx_host", "SEARX_HOST")
        if not searx_host.startswith("http"):
            print(
                f"Warning: missing the url scheme on host \
                ! assuming secure https://{searx_host} "
            )
            searx_host = "https://" + searx_host
        elif searx_host.startswith("http://"):
            values["unsecure"] = True
            cls.disable_ssl_warnings(True)
        values["searx_host"] = searx_host

        return values

    class Config:
        """Configuration for this pydantic object."""

        extra = Extra.forbid

    def _searx_api_query(self, params: dict) -> SearxResults:
        """Actual request to searx API."""
        raw_result = requests.get(
            self.searx_host,
            headers=self.headers,
            params=params,
            verify=not self.unsecure,
        )
        # test if http result is ok
        if not raw_result.ok:
            raise ValueError("Searx API returned an error: ", raw_result.text)
        res = SearxResults(raw_result.text)
        self._result = res
        return res

    async def _asearx_api_query(self, params: dict) -> SearxResults:
        if not self.aiosession:
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    self.searx_host,
                    headers=self.headers,
                    params=params,
                    ssl=(lambda: False if self.unsecure else None)(),
                ) as response:
                    if not response.ok:
                        raise ValueError(
                            "Searx API returned an error: ", response.text
                        )
                    result = SearxResults(await response.text())
                    self._result = result
        else:
            async with self.aiosession.get(
                self.searx_host,
                headers=self.headers,
                params=params,
                verify=not self.unsecure,
            ) as response:
                if not response.ok:
                    raise ValueError(
                        "Searx API returned an error: ", response.text
                    )
                result = SearxResults(await response.text())
                self._result = result

        return result

    def run(
        self,
        query: str,
        engines: Optional[List[str]] = None,
        categories: Optional[List[str]] = None,
        query_suffix: Optional[str] = "",
        **kwargs: Any,
    ) -> str:
        """Run query through Searx API and parse results.

        You can pass any other params to the searx query API.

        Args:
            query: The query to search for.
            query_suffix: Extra suffix appended to the query.
            engines: List of engines to use for the query.
            categories: List of categories to use for the query.
            **kwargs: extra parameters to pass to the searx API.

        Returns:
            str: The result of the query.

        Raises:
            ValueError: If an error occured with the query.


        Example:
            This will make a query to the qwant engine:

            .. code-block:: python

                from langchain.utilities import SearxSearchWrapper
                searx = SearxSearchWrapper(searx_host="http://my.searx.host")
                searx.run("what is the weather in France ?", engine="qwant")

                # the same result can be achieved using the `!` syntax of searx
                # to select the engine using `query_suffix`
                searx.run("what is the weather in France ?", query_suffix="!qwant")
        """
        _params = {
            "q": query,
        }
        params = {**self.params, **_params, **kwargs}

        if self.query_suffix and len(self.query_suffix) > 0:
            params["q"] += " " + self.query_suffix

        if isinstance(query_suffix, str) and len(query_suffix) > 0:
            params["q"] += " " + query_suffix

        if isinstance(engines, list) and len(engines) > 0:
            params["engines"] = ",".join(engines)

        if isinstance(categories, list) and len(categories) > 0:
            params["categories"] = ",".join(categories)

        res = self._searx_api_query(params)

        if len(res.answers) > 0:
            toret = res.answers[0]

        # only return the content of the results list
        elif len(res.results) > 0:
            toret = "\n\n".join(
                [r.get("content", "") for r in res.results[: self.k]]
            )
        else:
            toret = "No good search result found"

        return toret

    async def arun(
        self,
        query: str,
        engines: Optional[List[str]] = None,
        query_suffix: Optional[str] = "",
        **kwargs: Any,
    ) -> str:
        """Asynchronously version of `run`."""
        _params = {
            "q": query,
        }
        params = {**self.params, **_params, **kwargs}

        if self.query_suffix and len(self.query_suffix) > 0:
            params["q"] += " " + self.query_suffix

        if isinstance(query_suffix, str) and len(query_suffix) > 0:
            params["q"] += " " + query_suffix

        if isinstance(engines, list) and len(engines) > 0:
            params["engines"] = ",".join(engines)

        res = await self._asearx_api_query(params)

        if len(res.answers) > 0:
            toret = res.answers[0]

        # only return the content of the results list
        elif len(res.results) > 0:
            toret = "\n\n".join(
                [r.get("content", "") for r in res.results[: self.k]]
            )
        else:
            toret = "No good search result found"

        return toret

    def results(
        self,
        query: str,
        num_results: int,
        engines: Optional[List[str]] = None,
        categories: Optional[List[str]] = None,
        query_suffix: Optional[str] = "",
        **kwargs: Any,
    ) -> List[Dict]:
        """Run query through Searx API and returns the results with metadata.

        Args:
            query: The query to search for.
            query_suffix: Extra suffix appended to the query.
            num_results: Limit the number of results to return.
            engines: List of engines to use for the query.
            categories: List of categories to use for the query.
            **kwargs: extra parameters to pass to the searx API.

        Returns:
            Dict with the following keys:
            {
                snippet:  The description of the result.
                title:  The title of the result.
                link: The link to the result.
                engines: The engines used for the result.
                category: Searx category of the result.
            }

        """
        _params = {
            "q": query,
        }
        params = {**self.params, **_params, **kwargs}
        if self.query_suffix and len(self.query_suffix) > 0:
            params["q"] += " " + self.query_suffix
        if isinstance(query_suffix, str) and len(query_suffix) > 0:
            params["q"] += " " + query_suffix
        if isinstance(engines, list) and len(engines) > 0:
            params["engines"] = ",".join(engines)
        if isinstance(categories, list) and len(categories) > 0:
            params["categories"] = ",".join(categories)
        results = self._searx_api_query(params).results[:num_results]
        if len(results) == 0:
            return [{"Result": "No good Search Result was found"}]

        return [
            {
                "snippet": result.get("content", ""),
                "title": result["title"],
                "link": result["url"],
                "engines": result["engines"],
                "category": result["category"],
            }
            for result in results
        ]

    async def aresults(
        self,
        query: str,
        num_results: int,
        engines: Optional[List[str]] = None,
        query_suffix: Optional[str] = "",
        **kwargs: Any,
    ) -> List[Dict]:
        """Asynchronously query with json results.

        Uses aiohttp. See `results` for more info.
        """
        _params = {
            "q": query,
        }
        params = {**self.params, **_params, **kwargs}

        if self.query_suffix and len(self.query_suffix) > 0:
            params["q"] += " " + self.query_suffix
        if isinstance(query_suffix, str) and len(query_suffix) > 0:
            params["q"] += " " + query_suffix
        if isinstance(engines, list) and len(engines) > 0:
            params["engines"] = ",".join(engines)
        results = (await self._asearx_api_query(params)).results[:num_results]
        if len(results) == 0:
            return [{"Result": "No good Search Result was found"}]

        return [
            {
                "snippet": result.get("content", ""),
                "title": result["title"],
                "link": result["url"],
                "engines": result["engines"],
                "category": result["category"],
            }
            for result in results
        ]


class DuckDuckGoSearchAPIWrapper(BaseModel):
    """Wrapper for DuckDuckGo Search API.

    Free and does not require any setup
    """

    k: int = 10  # Not used
    region: str = "wt-wt"
    safesearch: str = "moderate"
    time: Optional[str] = "y"
    max_results: int = 10  # Max results to return

    class Config:
        """Configuration for this pydantic object."""

        extra = Extra.forbid

    @staticmethod
    def _ddg(
        keywords: str,
        region: str = "wt-wt",
        safesearch: str = "moderate",
        timelimit: Optional[str] = None,
        backend: str = "api",
        max_results: Optional[int] = None,
        pages: Optional[int] = 1,
        results_per_page: int = 20,
    ) -> List[Dict]:
        results = []
        for result in DDGS().text(
            keywords=keywords,
            region=region,
            safesearch=safesearch,
            timelimit=timelimit,
            backend=backend,
        ):
            results.append(result)
            if (max_results and len(results) >= max_results) or (
                pages and len(results) >= results_per_page * pages
            ):
                break
        return results

    @staticmethod
    def _get_formatted_result(result: Dict[str, str]) -> str:
        return "# [{link}]\n```{title}\n{snippet}\n```".format(
            title=result["title"],
            snippet=result["snippet"],
            link=result["link"],
        )

    def get_snippets(self, query: str) -> List[str]:
        """Run query through DuckDuckGo and return concatenated results."""
        results = self._ddg(
            query,
            region=self.region,
            safesearch=self.safesearch,
            timelimit=self.time,
            max_results=self.max_results,
        )

        if results is None or len(results) == 0:
            return ["No good DuckDuckGo Search Result was found"]
        snippets = [result["body"] for result in results]
        return snippets

    def run(self, query: str) -> str:
        return "\n\n".join(self.formatted_results(query))

    def formatted_results(self, query: str) -> List[str]:
        return [
            self._get_formatted_result(result)
            for result in self.results(query)
        ]

    def formatted_results_with_link(self, query: str) -> Dict[str, str]:
        return {
            result["link"]: self._get_formatted_result(result)
            for result in self.results(query)
        }

    def results(
        self, query: str, num_results: Optional[int] = None
    ) -> List[Dict[str, str]]:
        """Run query through DuckDuckGo and return metadata.

        Args:
            query: The query to search for.
            num_results: The number of results to return.

        Returns:
            A list of dictionaries with the following keys:
                snippet - The description of the result.
                title - The title of the result.
                link - The link to the result.
        """
        results = self._ddg(
            query,
            region=self.region,
            safesearch=self.safesearch,
            timelimit=self.time,
            max_results=self.max_results,
        )

        if results is None or len(results) == 0:
            return [{"Result": "No good DuckDuckGo Search Result was found"}]

        return [
            {
                "snippet": result["body"],
                "title": result["title"],
                "link": result["href"],
            }
            for result in results
        ][:num_results]
