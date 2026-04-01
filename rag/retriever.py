from langchain_community.retrievers import BM25Retriever
from langchain_chroma import Chroma
from langchain_core.documents import Document
from typing import List


def get_dense_retriever(vectorstore: Chroma, k: int = 4):
    return vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": k}
    )


def get_bm25_retriever(chunks, k: int = 4):
    retriever = BM25Retriever.from_documents(chunks)
    retriever.k = k
    return retriever


class HybridRetriever:
    """
    Manual hybrid retriever combining BM25 + dense search.
    Merges results from both, deduplicates by content,
    and returns top-k unique chunks.
    """
    def __init__(self, vectorstore, chunks, k=4,
                 dense_weight=0.6, bm25_weight=0.4):
        self.dense = get_dense_retriever(vectorstore, k=k)
        self.bm25 = get_bm25_retriever(chunks, k=k)
        self.k = k
        self.dense_weight = dense_weight
        self.bm25_weight = bm25_weight

    def invoke(self, query: str) -> List[Document]:
        dense_docs = self.dense.invoke(query)
        bm25_docs = self.bm25.invoke(query)

        # Merge and deduplicate by page_content
        seen = set()
        merged = []
        # Add dense results first (higher weight)
        for doc in dense_docs:
            key = doc.page_content[:200]
            if key not in seen:
                seen.add(key)
                merged.append(doc)
        # Add BM25 results
        for doc in bm25_docs:
            key = doc.page_content[:200]
            if key not in seen:
                seen.add(key)
                merged.append(doc)

        return merged[:self.k * 2]


def get_retriever_by_mode(mode: str, vectorstore, chunks, k: int = 4):
    if mode == "dense":
        return get_dense_retriever(vectorstore, k=k)
    elif mode == "bm25":
        return get_bm25_retriever(chunks, k=k)
    elif mode == "hybrid":
        return HybridRetriever(vectorstore, chunks, k=k)
    else:
        raise ValueError(f"Unknown mode: {mode}. Choose: dense, bm25, hybrid")