import os
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from dotenv import load_dotenv

load_dotenv()


def get_llm():
    """Returns GPT-4o-mini instance."""
    return ChatOpenAI(
        model="gpt-4o-mini",
        api_key=os.getenv("OPENAI_API_KEY"),
        temperature=0.1  # low temperature for factual RAG answers
    )


# Prompt template for RAG
# Instructs LLM to answer ONLY from provided context
RAG_PROMPT = ChatPromptTemplate.from_template("""
You are a helpful research assistant. Answer the question based ONLY on the 
provided context from research papers. If the answer is not in the context, 
say "I cannot find this information in the provided papers."

Do not use any outside knowledge. Be precise and cite which part of the 
context supports your answer.

Context:
{context}

Question: {question}

Answer:
""")


def format_docs(docs):
    """Joins retrieved document chunks into a single context string."""
    return "\n\n".join(
        f"[Source: {doc.metadata.get('source', 'unknown')}, "
        f"Page: {doc.metadata.get('page', '?')}]\n{doc.page_content}"
        for doc in docs
    )


def build_rag_chain(retriever):
    """
    Builds the RAG chain.

    Flow:
        Question
            |
            v
        Retriever (gets relevant chunks)
            |
            v
        format_docs (joins chunks into context string)
            |
            v
        RAG_PROMPT (fills context + question into prompt)
            |
            v
        LLM (GPT-4o-mini generates answer)
            |
            v
        StrOutputParser (extracts text from response)

    Args:
        retriever: any retriever instance (dense, bm25, or hybrid)

    Returns:
        Runnable RAG chain
    """
    llm = get_llm()

    return {"retriever": retriever, "llm": llm, "prompt": RAG_PROMPT}


def ask(question: str, retriever, return_sources: bool = False):
    """
    Ask a question using the RAG pipeline.

    Args:
        question: user's question
        retriever: retriever instance to use
        return_sources: if True, also return the retrieved source chunks

    Returns:
        answer string, and optionally list of source documents
    """
    llm = get_llm()

    # Get relevant documents from retriever
    docs = retriever.invoke(question)

    # Format context from retrieved docs
    context = format_docs(docs)

    # Build and invoke the chain manually
    prompt_value = RAG_PROMPT.invoke({
        "context": context,
        "question": question
    })

    answer = llm.invoke(prompt_value)
    answer_text = answer.content

    if return_sources:
        return answer_text, docs

    return answer_text
