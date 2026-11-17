import os 
from langchain_chroma import Chroma 
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

CHROMA_DIR = "vector_db"
COLLECTION_NAME = "meeting_transcript"
# Multilingual embedding model supporting English, Hindi, and 50+ languages
EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

_embeddings = None

def get_embeddings():
    global _embeddings
    if _embeddings is None:
        _embeddings = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL,
            model_kwargs={"device": "cpu"}
        )
    return _embeddings

def build_vector_store(transcript: str) -> Chroma:
    print("Building vector store...")

    embeddings = get_embeddings()

    # Clear old collection from previous sessions to prevent cross-meeting pollution
    try:
        old_vs = Chroma(
            collection_name=COLLECTION_NAME,
            embedding_function=embeddings,
            persist_directory=CHROMA_DIR
        )
        old_vs.delete_collection()
    except Exception as e:
        print(f"Vector store reset note: {e}")

    if not transcript or not transcript.strip():
        print("Warning: transcript is empty.")
        docs = [Document(page_content="No transcript content available.", metadata={'chunk_index': 0})]
    else:
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=600,
            chunk_overlap=80
        )
        chunks = splitter.split_text(transcript.strip())
        if not chunks:
            chunks = [transcript.strip()]

        docs = [
            Document(page_content=chunk, metadata={'chunk_index': i})
            for i, chunk in enumerate(chunks)
        ]

    vector_store = Chroma.from_documents(
        documents=docs,
        embedding=embeddings,
        collection_name=COLLECTION_NAME,
        persist_directory=CHROMA_DIR
    )

    print(f"Vector store built with {len(docs)} chunk(s).")
    return vector_store

def load_vector_store() -> Chroma:
    embeddings = get_embeddings()
    vector_store = Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=CHROMA_DIR
    )
    return vector_store

def get_retriever(vector_store: Chroma, k: int = 6):
    return vector_store.as_retriever(
        search_type='similarity',
        search_kwargs={"k": k}
    )
