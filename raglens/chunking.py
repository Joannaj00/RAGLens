from langchain_text_splitters import CharacterTextSplitter, RecursiveCharacterTextSplitter

# fixed length chunking
def chunk_docs_fixed(docs, chunk_size=1000, chunk_overlap=150):
    splitter = CharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    chunks = splitter.split_documents(docs)
    for i, c in enumerate(chunks):
        c.metadata["chunk_id"] = i # to refer to later
    return chunks

# recursive chunking
    # tries to split on natural boundaries first, then falls back to character count
    # order: by paragraph (\n\n) -> by lines (\n) -> by words (' ') -> 'by characters ('')
def chunk_docs_recursive(docs, chunk_size=1000, chunk_overlap=150):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", " ", ""]
    )
    chunks = splitter.split_documents(docs)
    for i, c in enumerate(chunks):
        c.metadata["chunk_id"] = i 
    return chunks