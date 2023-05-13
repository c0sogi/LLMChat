import io
from typing import IO, Any

from langchain.document_loaders.unstructured import UnstructuredBaseLoader
from langchain.docstore.document import Document
from unstructured.partition.auto import partition


class UnstructuredFileIOLoader(UnstructuredBaseLoader):
    """Loader that uses unstructured to load file IO objects."""

    def __init__(self, file: IO, filename: str, mode: str = "single", **unstructured_kwargs: Any):
        """Initialize with file path."""
        self.file = file
        self.filename = filename
        super().__init__(mode=mode, **unstructured_kwargs)

    def _get_elements(self) -> list:
        return partition(file=self.file, file_filename=self.filename, **self.unstructured_kwargs)

    def _get_metadata(self) -> dict:
        return {}


def read_bytes_to_documents(file: bytes, filename: str) -> list[Document]:
    return UnstructuredFileIOLoader(file=io.BytesIO(file), strategy="fast", filename=filename).load()


def read_bytes_to_text(file: bytes, filename: str) -> str:
    return "\n\n".join([doc.page_content for doc in read_bytes_to_documents(file=file, filename=filename)])


if __name__ == "__main__":
    with open(r"test.pdf", "rb") as f:
        file = f.read()
    text = read_bytes_to_text(file, "test.pdf")
    print(text)
