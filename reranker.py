from sentence_transformers import CrossEncoder, SentenceTransformer
import chromadb
import os
import json
from dotenv import load_dotenv
from groq import Groq
import re

load_dotenv()
client = Groq(api_key=os.environ["GROQ_API_KEY"])

# a small, fast, local embedding model — runs on your machine, no API needed
model = SentenceTransformer("all-MiniLM-L6-v2")

# creates a local, persistent database on disk (no server, no API key needed)
chroma_client = chromadb.PersistentClient(path="./chroma_db")

# a cross-encoder: takes (query, chunk) PAIRS and outputs a relevance score directly
reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

tricky_doc = """
Our return policy allows customers to return unused items within 30 days of purchase
for a full refund. Refunds are processed within 5-7 business days after we receive
the item.

Shipping typically takes 3-5 business days for domestic orders. International orders
may take 2-3 weeks depending on customs processing. We are not responsible for delays
caused by customs.

If your order is damaged during shipping, please contact support within 48 hours of
delivery. We do not offer refunds for items damaged due to customer misuse. Refunds
for damaged-in-transit items are processed as store credit within 3 business days.
"""

def sentence_aware_chunk(text, max_chunk_size=300, overlap_sentences=1):
    """Splits text into sentences first, then groups sentences into chunks
    without ever cutting a sentence in half."""
    text = text.strip()
    # naive sentence splitter — splits on '. ', '! ', '? ' followed by a capital letter
    # (good enough for now; real systems use a proper sentence tokenizer — nltk/spacy)
    sentences = re.split(r'(?<=[.!?])\s+', text)

    chunks = []
    current_chunk = []
    current_length = 0

    for sentence in sentences:
        if current_length + len(sentence) > max_chunk_size and current_chunk:
            chunks.append(" ".join(current_chunk))
            # keep last N sentences for overlap, so context carries across chunk boundaries
            current_chunk = current_chunk[-overlap_sentences:]
            current_length = sum(len(s) for s in current_chunk)

        current_chunk.append(sentence)
        current_length += len(sentence)

    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return chunks

# track which document + which chunk index, for traceability
tricky_chunks = sentence_aware_chunk(tricky_doc, max_chunk_size=250, overlap_sentences=1)
for i, c in enumerate(tricky_chunks):
    print(f"--- Chunk {i} ---\n{c}\n")

tricky_embeddings = model.encode(tricky_chunks).tolist()
collection_tricky = chroma_client.get_or_create_collection(name="rag_tricky")
collection_tricky.add(
    documents=tricky_chunks,
    embeddings=tricky_embeddings,
    ids=[f"tchunk_{i}" for i in range(len(tricky_chunks))]
)

# def rag_query_reranked(query, collection, initial_k=5, final_k=2):
#     # Step 1: broad retrieval using embeddings (fast, cast a wide net)
#     query_embedding = model.encode([query]).tolist()
#     results = collection.query(query_embeddings=query_embedding, n_results=initial_k)

#     candidates = list(zip(results["documents"][0], results["metadatas"][0], results["distances"][0]))

#     # Step 2: rerank — score each (query, chunk) pair directly
#     pairs = [[query, doc] for doc, meta, dist in candidates]
#     rerank_scores = reranker.predict(pairs)

#     # Step 3: sort by the NEW reranker scores, not the original embedding distances
#     reranked = sorted(zip(candidates, rerank_scores), key=lambda x: x[1], reverse=True)

#     print(f"Query: {query}\n")
#     for (doc, meta, orig_dist), rerank_score in reranked[:final_k]:
#         print(f"[rerank score: {rerank_score:.3f}] [orig distance: {orig_dist:.3f}] [source: {meta['source']}]")
#         print(f"{doc}\n")

#     return reranked[:final_k]


# rag_query_reranked("How are key-value pairs stored?", collection_multi)

query_tricky = "How long does it take to get a refund for a damaged item?"

# Stage 1 only — raw embedding ranking
results = collection_tricky.query(
    query_embeddings=model.encode([query_tricky]).tolist(),
    n_results=3
)
print("--- EMBEDDING-ONLY RANKING ---")
for doc, dist in zip(results["documents"][0], results["distances"][0]):
    print(f"[distance: {dist:.3f}] {doc}\n")

# Stage 2 — reranked
candidates = results["documents"][0]
pairs = [[query_tricky, doc] for doc in candidates]
scores = reranker.predict(pairs)
reranked = sorted(zip(candidates, scores), key=lambda x: x[1], reverse=True)

print("--- RERANKED ---")
for doc, score in reranked:
    print(f"[rerank score: {score:.3f}] {doc}\n")