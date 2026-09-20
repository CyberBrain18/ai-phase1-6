import os
from dotenv import load_dotenv
from groq import Groq
from sentence_transformers import SentenceTransformer
import chromadb
import re

load_dotenv()
client = Groq(api_key=os.environ["GROQ_API_KEY"])

model = SentenceTransformer("all-MiniLM-L6-v2")
chroma_client = chromadb.PersistentClient(path="./chroma_db")


def sentence_aware_chunk(text, max_chunk_size=60, overlap_sentences=1):
    text = text.strip()
    sentences = re.split(r'(?<=[.!?])\s+', text)
    chunks = []
    current_chunk = []
    current_length = 0
    for sentence in sentences:
        if current_length + len(sentence) > max_chunk_size and current_chunk:
            chunks.append(" ".join(current_chunk))
            current_chunk = current_chunk[-overlap_sentences:]
            current_length = sum(len(s) for s in current_chunk)
        current_chunk.append(sentence)
        current_length += len(sentence)
    if current_chunk:
        chunks.append(" ".join(current_chunk))
    return chunks


# --- the multi-document dataset ---
documents = {
    "doc_python_basics.txt": """
Python is a high-level, interpreted programming language known for its readability.
Variables in Python don't require explicit type declarations. Lists are ordered,
mutable collections, while tuples are ordered but immutable. Dictionaries store
key-value pairs and are widely used for structured data.
""",
    "doc_python_performance.txt": """
Python's performance can be a bottleneck for CPU-intensive tasks due to the Global
Interpreter Lock (GIL), which prevents true multi-threading for CPU-bound code.
For performance-critical sections, developers often use libraries like NumPy, which
run compiled C code under the hood, or switch to multiprocessing instead of threading.
""",
    "doc_js_basics.txt": """
JavaScript is a dynamically typed language primarily used for web development.
Variables can be declared with var, let, or const. Arrays in JavaScript are
similar to Python lists but have different built-in methods. Objects store
key-value pairs, similar to Python dictionaries but with different syntax.
""",
    "doc_rag_overview.txt": """
Retrieval-Augmented Generation (RAG) combines a retrieval system with a language model.
Instead of relying solely on parametric knowledge learned during training, RAG systems
fetch relevant documents at inference time.

The retrieval step typically uses vector similarity search. Documents are split into
chunks, embedded, and stored in a vector database like Chroma, Pinecone, or FAISS.

Chunk size matters significantly. Too small, and you lose context. Too large, and
irrelevant information dilutes the embedding, hurting retrieval accuracy. A common
starting point is 200-500 tokens per chunk with some overlap between chunks.
""",
    "doc_vector_db_comparison.txt": """
Choosing a vector database depends on your scale and infrastructure needs. Chroma is
lightweight and great for local development or small projects. Pinecone is fully
managed and scales well for production but has ongoing costs. FAISS is a library,
not a full database — extremely fast, but you handle persistence and infra yourself.
""",
}

all_chunks = []
all_metadatas = []
for doc_name, doc_text in documents.items():
    doc_chunks = sentence_aware_chunk(doc_text, max_chunk_size=300, overlap_sentences=1)
    for i, chunk in enumerate(doc_chunks):
        all_chunks.append(chunk)
        all_metadatas.append({"source": doc_name, "chunk_index": i})

chunk_embeddings_multi = model.encode(all_chunks).tolist()

# use a fresh collection name to avoid clashing with any old data from previous runs
collection_multi = chroma_client.get_or_create_collection(name="rag_eval_demo_v2")
collection_multi.add(
    documents=all_chunks,
    embeddings=chunk_embeddings_multi,
    metadatas=all_metadatas,
    ids=[f"chunk_{i}" for i in range(len(all_chunks))]
)


# --- the eval set ---
eval_set = [
    {"query": "What determines good chunk size?", "expected_source": "doc_rag_overview.txt"},
    {"query": "Why is Python slow for CPU-heavy tasks?", "expected_source": "doc_python_performance.txt"},
    {"query": "Which vector database should I use for production?", "expected_source": "doc_vector_db_comparison.txt"},
    {"query": "How do I declare a variable in JavaScript?", "expected_source": "doc_js_basics.txt"},
    {"query": "What's the difference between lists and tuples in Python?", "expected_source": "doc_python_basics.txt"},
]


def eval_retrieval(eval_set, collection, top_k=1):
    correct = 0
    results_log = []

    for item in eval_set:
        query_embedding = model.encode([item["query"]]).tolist()
        results = collection.query(query_embeddings=query_embedding, n_results=top_k)

        retrieved_sources = [meta["source"] for meta in results["metadatas"][0]]
        is_correct = item["expected_source"] in retrieved_sources

        correct += is_correct
        results_log.append({
            "query": item["query"],
            "expected": item["expected_source"],
            "got": retrieved_sources,
            "correct": is_correct
        })

    accuracy = correct / len(eval_set)
    return accuracy, results_log


if __name__ == "__main__":
    accuracy, log = eval_retrieval(eval_set, collection_multi)

    print(f"Retrieval accuracy: {accuracy:.1%}\n")
    for r in log:
        status = "✓" if r["correct"] else "✗"
        print(f"{status} Query: {r['query']}")
        print(f"   Expected: {r['expected']} | Got: {r['got']}\n")