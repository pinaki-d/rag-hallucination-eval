"""
RAGAS Evaluation Pipeline

Benchmarks 3 retrieval configurations:
  A: chunk_size=512  + dense retrieval
  B: chunk_size=1024 + dense retrieval
  C: chunk_size=512  + hybrid retrieval  ← expected winner

Metrics measured:
  - faithfulness:      Does the answer stay grounded in retrieved context?
  - answer_relevancy:  Is the answer relevant to the question?
  - context_recall:    Does retrieved context cover the ground truth answer?
"""

import os
import pandas as pd
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_recall
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from dotenv import load_dotenv

from rag.ingestor import load_and_chunk_pdfs, build_vectorstore
from rag.retriever import get_retriever_by_mode
from rag.pipeline import ask, format_docs
from evaluation.eval_dataset import EVAL_QUESTIONS

load_dotenv()


def run_rag_for_eval(questions, retriever):
    """
    Runs RAG pipeline on all eval questions and collects:
    - question
    - generated answer
    - retrieved contexts
    - ground truth answer

    Returns a HuggingFace Dataset ready for RAGAS.
    """
    data = {
        "question": [],
        "answer": [],
        "contexts": [],
        "ground_truth": []
    }

    for i, item in enumerate(questions):
        print(f"  Running question {i+1}/{len(questions)}: {item['question'][:60]}...")

        answer, docs = ask(
            question=item["question"],
            retriever=retriever,
            return_sources=True
        )

        data["question"].append(item["question"])
        data["answer"].append(answer)
        data["contexts"].append([doc.page_content for doc in docs])
        data["ground_truth"].append(item["ground_truth"])

    return Dataset.from_dict(data)


def evaluate_config(config_name, chunk_size, retrieval_mode, questions):
    """
    Evaluates one configuration end to end.

    Args:
        config_name: label for this config e.g. "A: 512 + dense"
        chunk_size: chunk size for ingestion
        retrieval_mode: "dense", "bm25", or "hybrid"
        questions: list of eval question dicts

    Returns:
        dict with config name and RAGAS scores
    """
    print(f"\n{'='*50}")
    print(f"Evaluating Config {config_name}")
    print(f"  chunk_size={chunk_size}, retrieval={retrieval_mode}")
    print(f"{'='*50}")

    # Build fresh vectorstore for this chunk size
    chunks = load_and_chunk_pdfs(chunk_size=chunk_size)
    vectorstore = build_vectorstore(
        chunks,
        collection_name=f"papers_{chunk_size}"
    )

    # Get retriever
    retriever = get_retriever_by_mode(
        mode=retrieval_mode,
        vectorstore=vectorstore,
        chunks=chunks,
        k=4
    )

    # Run RAG on all questions
    print(f"\nRunning RAG on {len(questions)} questions...")
    dataset = run_rag_for_eval(questions, retriever)

    # RAGAS needs OpenAI for its own internal LLM judge
    llm = ChatOpenAI(
        model="gpt-4o-mini",
        api_key=os.getenv("OPENAI_API_KEY")
    )
    embeddings = OpenAIEmbeddings(
        api_key=os.getenv("OPENAI_API_KEY")
    )

    print("\nRunning RAGAS evaluation...")
    results = evaluate(
        dataset=dataset,
        metrics=[faithfulness, answer_relevancy, context_recall],
        llm=llm,
        embeddings=embeddings
    )
    
    scores = {
        "config": config_name,
        "chunk_size": chunk_size,
        "retrieval_mode": retrieval_mode,
        "faithfulness": round(sum(results["faithfulness"]) / len(results["faithfulness"]), 4),
        "answer_relevancy": round(sum(results["answer_relevancy"]) / len(results["answer_relevancy"]), 4),
        "context_recall": round(sum(results["context_recall"]) / len(results["context_recall"]), 4),
    }

    print(f"\nResults for {config_name}:")
    for k, v in scores.items():
        if k not in ["config", "chunk_size", "retrieval_mode"]:
            print(f"  {k}: {v:.4f}")

    return scores


def run_full_benchmark():
    """
    Runs all 3 configurations and saves results to CSV.

    Configurations:
        A: chunk_size=512  + dense
        B: chunk_size=1024 + dense
        C: chunk_size=512  + hybrid  ← the differentiator
    """
    configs = [
        ("A: 512 + dense",   512,  "dense"),
        ("B: 1024 + dense",  1024, "dense"),
        ("C: 512 + hybrid",  512,  "hybrid"),
    ]

    all_results = []
    for config_name, chunk_size, retrieval_mode in configs:
        result = evaluate_config(
            config_name=config_name,
            chunk_size=chunk_size,
            retrieval_mode=retrieval_mode,
            questions=EVAL_QUESTIONS
        )
        all_results.append(result)

    # Save to CSV
    df = pd.DataFrame(all_results)
    os.makedirs("data", exist_ok=True)
    df.to_csv("data/benchmark_results.csv", index=False)

    print("\n" + "="*60)
    print("BENCHMARK COMPLETE")
    print("="*60)
    print(df.to_string(index=False))
    print("\nResults saved to data/benchmark_results.csv")

    return df


if __name__ == "__main__":
    run_full_benchmark()
