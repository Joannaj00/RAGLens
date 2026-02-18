import json
from langchain_core.documents import Document

def load_jsonl(path: str) -> list[Document]:
    docs = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line) # turn json-formatted string -> python dict
            metadata = {
                "id": row.get("id"),
                "title": row.get("title"),
                "source": row.get("source")
            }
            content = row.get("abstract", "")
            doc = Document(page_content=content, metadata=metadata)
            docs.append(doc)
    return docs 