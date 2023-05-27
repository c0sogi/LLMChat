from langchain.docstore.document import Document

from app.common.constants import LONG_PROMPT
from app.shared import Shared

summarize_chain = Shared().map_reduce_summarize_chain
docs = [Document(page_content=LONG_PROMPT)]
result = summarize_chain.run(docs)

print(result)
