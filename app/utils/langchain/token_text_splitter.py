from copy import deepcopy
from typing import (
    AbstractSet,
    Any,
    Collection,
    Iterable,
    List,
    Literal,
    Optional,
    Sequence,
    Union,
)

from langchain.docstore.document import Document
from langchain.text_splitter import Tokenizer, TokenTextSplitter, split_text_on_tokens


class CustomTokenTextSplitter(TokenTextSplitter):
    """Implementation of splitting text that looks at tokens."""

    def __init__(
        self,
        encoding_name: str = "gpt2",
        model_name: Optional[str] = None,
        allowed_special: Union[Literal["all"], AbstractSet[str]] = set(),
        disallowed_special: Union[Literal["all"], Collection[str]] = "all",
        **kwargs: Any,
    ):
        super().__init__(
            encoding_name=encoding_name,
            model_name=model_name,
            allowed_special=allowed_special,
            disallowed_special=disallowed_special,
            **kwargs,
        )

    def split_text(
        self,
        text: str,
        tokens_per_chunk: Optional[int] = None,
        chunk_overlap: Optional[int] = None,
    ) -> List[str]:
        def _encode(_text: str) -> List[int]:
            return self._tokenizer.encode(
                _text,
                allowed_special=self._allowed_special,  # type: ignore
                disallowed_special=self._disallowed_special,
            )

        tokenizer = Tokenizer(
            chunk_overlap=self._chunk_overlap
            if chunk_overlap is None
            else chunk_overlap,
            tokens_per_chunk=self._chunk_size
            if tokens_per_chunk is None
            else tokens_per_chunk,
            decode=self._tokenizer.decode,
            encode=_encode,
        )

        return split_text_on_tokens(text=text, tokenizer=tokenizer)

    def create_documents(
        self,
        texts: List[str],
        metadatas: Optional[List[dict]] = None,
        tokens_per_chunk: Optional[int] = None,
        chunk_overlap: Optional[int] = None,
    ) -> List[Document]:
        """Create documents from a list of texts."""
        _metadatas = metadatas or [{}] * len(texts)
        documents = []
        for i, text in enumerate(texts):
            index = -1
            for chunk in self.split_text(
                text,
                tokens_per_chunk=tokens_per_chunk,
                chunk_overlap=chunk_overlap,
            ):
                metadata = deepcopy(_metadatas[i])
                if self._add_start_index:
                    index = text.find(chunk, index + 1)
                    metadata["start_index"] = index
                new_doc = Document(page_content=chunk, metadata=metadata)
                documents.append(new_doc)
        return documents

    def split_documents(
        self,
        documents: Iterable[Document],
        tokens_per_chunk: Optional[int] = None,
        chunk_overlap: Optional[int] = None,
    ) -> List[Document]:
        """Split documents."""
        texts, metadatas = [], []
        for doc in documents:
            texts.append(doc.page_content)
            metadatas.append(doc.metadata)
        return self.create_documents(
            texts,
            metadatas=metadatas,
            tokens_per_chunk=tokens_per_chunk,
            chunk_overlap=chunk_overlap,
        )

    def transform_documents(
        self,
        documents: Sequence[Document],
        tokens_per_chunk: Optional[int] = None,
        chunk_overlap: Optional[int] = None,
        **kwargs: Any,
    ) -> Sequence[Document]:
        """Transform sequence of documents by splitting them."""
        return self.split_documents(
            list(documents),
            tokens_per_chunk=tokens_per_chunk,
            chunk_overlap=chunk_overlap,
        )
