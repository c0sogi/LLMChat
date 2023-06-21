"""Wrapper around embedding API models."""
from __future__ import annotations

import logging
from typing import (
    Any,
    List,
    Optional,
    Tuple,
    Union,
)

import requests

from pydantic import BaseModel, Extra
from langchain.embeddings.base import Embeddings

logger = logging.getLogger(__name__)


class APIEmbeddings(BaseModel, Embeddings):
    """Wrapper around embedding models from OpenAI-style API"""

    client: Any  #: :meta private:
    model: str = "intfloat/e5-large-v2"
    embedding_api_url: str = "http://localhost:8002/v1/embeddings"
    request_timeout: Optional[Union[float, Tuple[float, float]]] = None
    """Timeout in seconds for the API request."""
    headers: Any = None

    class Config:
        """Configuration for this pydantic object."""

        extra = Extra.forbid

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Call out to embedding endpoint for embedding search docs.

        Args:
            texts: The list of texts to embed.

        Returns:
            List of embeddings, one for each text.
        """
        # NOTE: to keep things simple, we assume the list may contain texts longer
        #       than the maximum context and use length-safe embedding function.
        response = requests.post(
            self.embedding_api_url,
            json={
                "model": self.model,
                "input": texts,
            },
        )

        # Check if the request was successful
        if response.status_code == 200:
            # Parse the response
            response_data = response.json()

            # Extract the embeddings and total tokens
            embeddings: list[list[float]] = [
                data["embedding"] for data in response_data["data"]
            ]
            total_tokens = response_data["usage"]["total_tokens"]
            return embeddings
        raise ConnectionError(
            f"Request to {self.embedding_api_url} failed with status code "
            f"{response.status_code}."
        )

    def embed_query(self, text: str) -> List[float]:
        """Call out to OpenAI's embedding endpoint for embedding query text.

        Args:
            text: The text to embed.

        Returns:
            Embedding for the text.
        """
        response = requests.post(
            self.embedding_api_url,
            json={
                "model": self.model,
                "input": text,
            },
        )

        # Check if the request was successful
        if response.status_code == 200:
            # Parse the response
            response_data = response.json()

            # Extract the embeddings and total tokens
            embeddings: list[list[float]] = [
                data["embedding"] for data in response_data["data"]
            ]
            total_tokens = response_data["usage"]["total_tokens"]
            return embeddings[0]
        raise ConnectionError(
            f"Request to {self.embedding_api_url} failed with status code "
            f"{response.status_code}."
        )


if __name__ == "__main__":
    import argparse
    import json

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="intfloat/e5-large-v2")
    parser.add_argument(
        "--embedding_api_url", type=str, default="http://localhost:8002/v1/embeddings"
    )
    parser.add_argument("--request_timeout", type=float, default=None)
    parser.add_argument("--headers", type=str, default=None)
    parser.add_argument("--text", type=str, default="Hello, world!")
    args = parser.parse_args()

    print(args)

    # Create the API embeddings model
    api_embeddings = APIEmbeddings(
        client=None,
        model=args.model,
        embedding_api_url=args.embedding_api_url,
        request_timeout=args.request_timeout,
        headers=args.headers,
    )

    # Embed the query
    query_embedding = api_embeddings.embed_query(args.text)

    # Print the query embedding
    print(json.dumps(query_embedding))
