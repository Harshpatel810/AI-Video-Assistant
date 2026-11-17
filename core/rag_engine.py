import os
import sys

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

from langchain_openrouter import ChatOpenRouter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from core.vector_store import build_vector_store, load_vector_store, get_retriever

def get_llm():
    return ChatOpenRouter(
        model="nvidia/nemotron-3.5-super-vl:free",
        openrouter_api_key=os.getenv("OPENROUTER_API_KEY"),
        temperature=0.3,
    )

def format_docs(docs):
    formatted = "\n\n".join([doc.page_content for doc in docs])
    return formatted

SYSTEM_PROMPT = """You are an expert video and meeting assistant. Answer the user's question accurately using ONLY the context provided below from the transcript.

Important rules:
1. The transcript may be in English, Hindi (Devanagari script), or Hinglish (mixed Hindi-English).
2. Comprehend the content across all these languages. If the transcript is in Hindi or Hinglish and the user asks in English, translate, summarize, and answer clearly in English.
3. If the user asks in Hindi/Hinglish, reply in Hindi/Hinglish.
4. Directly answer the question based on the content of the transcript.
5. Only say "I could not find this information in the meeting transcript." if the provided context genuinely does not contain any relevant details.

Context from transcript:
{context}"""

def build_rag_chain(transcript: str):
    vector_store = build_vector_store(transcript)
    retriever = get_retriever(vector_store, k=6)
    llm = get_llm()

    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        ("human", "{question}"),
    ])

    rag_chain = (
        {
            "context": retriever | RunnableLambda(format_docs),
            "question": RunnablePassthrough(),
        }
        | prompt
        | llm
        | StrOutputParser()
    )

    return rag_chain

def load_rag_chain():
    vector_store = load_vector_store()
    retriever = get_retriever(vector_store, k=6)
    llm = get_llm()

    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        ("human", "{question}"),
    ])

    rag_chain = (
        {
            "context": retriever | RunnableLambda(format_docs),
            "question": RunnablePassthrough(),
        }
        | prompt
        | llm
        | StrOutputParser()
    )

    return rag_chain

def ask_question(rag_chain, question: str) -> str:
    print(f"Question: {question}")
    try:
        answer = rag_chain.invoke(question)
    except Exception as e:
        print(f"Error invoking RAG chain: {e}")
        return f"Error retrieving answer: {e}"

    try:
        print(f"Answer: {answer}")
    except Exception:
        pass
    return answer
