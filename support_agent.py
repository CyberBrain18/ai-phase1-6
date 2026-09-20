from sentence_transformers import SentenceTransformer
import chromadb
import re
import os
from dotenv import load_dotenv
from groq import Groq
import json

load_dotenv()
client = Groq(api_key=os.environ["GROQ_API_KEY"])
MODEL = "openai/gpt-oss-120b"
DISTANCE_THRESHOLD = 1.0

model = SentenceTransformer("all-MiniLM-L6-v2")
chroma_client = chromadb.PersistentClient(path='./chroma_db')
print("Model and client loaded")

documents = {
    "shipping_policy.txt": """
We offer standard and express shipping. Standard shipping takes 5-7 business days
and costs $4.99. Express shipping takes 1-2 business days and costs $14.99. Orders
over $50 qualify for free standard shipping. We currently only ship within the
United States.
""",
    "returns_policy.txt": """
Items can be returned within 30 days of delivery for a full refund, provided they
are unused and in original packaging. To start a return, contact support with your
order number. Return shipping costs are the customer's responsibility unless the
item arrived damaged or defective.
""",
    "refund_policy.txt": """
Refunds are processed within 5-7 business days after we receive the returned item.
Refunds are issued to the original payment method. If an item arrived damaged, we
offer a full refund including original shipping costs, processed as store credit
within 3 business days instead of a refund to the card.
""",
}

# def simple_chunk(text, chunk_size=200):
#     text = text.strip() #remove whitespaces leading and trailing    
#     return [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]

# all_chunks = []
# all_metadata = []

# for doc_name, doc_text in documents.items():
#     doc_chunks = simple_chunk(doc_text)
#     for i, chunk in enumerate(doc_chunks):
#         all_chunks.append(chunk)
#         all_metadata.append({"source": doc_name, "chunk_index": i})

# print(f"Total chunks: {len(all_chunks)}")
# for i, c in enumerate(all_chunks):
#     print(f"--- Chunk{i} ({all_metadata[i]['source']}) ---\n{c}\n")

def sentence_aware_chunk(text, max_chunk_size=200, overlap_sentences=1):
    text = text.strip();
    sentences = re.split(r'(?<=[.!?])\s+', text)
    chunks = []
    current_chunk = []
    current_length = 0
    for sentence in sentences:
        if current_length + len(sentence)>max_chunk_size and current_chunk:
            chunks.append(" ".join(current_chunk))
            current_chunk = current_chunk[-overlap_sentences:]
            current_length = sum(len(s) for s in current_chunk)
        current_chunk.append(sentence)
        current_length += len(sentence)
    if current_chunk:
        chunks.append(" ".join(current_chunk))
    return chunks

def rag_answer(query, collection, threshold=DISTANCE_THRESHOLD, n_results=2):
    query_embedding = model.encode([query]).tolist()
    results = collection.query(query_embeddings=query_embedding, n_results=n_results)
    best_distance = results["distances"][0][0]
    if best_distance>threshold:
        return "I don't have information about that in our support docs.", best_distance
    context = "\n\n".join(results['documents'][0])
    prompt = f"""Answer the customer's question using ONLY the context below.

Context:
{context}

Question: {query}"""
    response = client.chat.completions.create(model=MODEL, messages=[{"role": "user", "content": prompt}])
    return response.choices[0].message.content, best_distance

all_chunks = []
all_metadata = []

for doc_name, doc_text in documents.items():
     doc_chunks = sentence_aware_chunk(doc_text)
     for i, chunk in enumerate(doc_chunks):
         all_chunks.append(chunk)
         all_metadata.append({"source": doc_name, "chunk_index": i})
    
chunk_embeddings = model.encode(all_chunks).tolist()

collection = chroma_client.get_or_create_collection(name="support_docs")
collection.add(
    documents=all_chunks,
    embeddings=chunk_embeddings,
    metadatas=all_metadata,
    ids=[f"chunk_{i}" for i in range(len(all_chunks))]
)

fake_orders = {
    "ORD1001": {"status": "shipped", "item": "wireless headphones", "amount": 49.99},
    "ORD1002": {"status": "delivered", "item": "phone case", "amount": 14.99},
    "ORD1003": {"status": "processing", "item": "desk lamp", "amount": 29.99},
}

def get_order_status(order_id: str) -> str:
    order = fake_orders.get(order_id.upper())
    if not order:
        return f"No order found with id {order_id}"
    return f"Order {order_id}: {order['status']}, item: {order['item']}, amount: ${order['amount']}"

def issue_refund(order_id: str, amount: float) -> str:
    order = fake_orders.get(order_id.upper())
    if not order:
        return f"Cannot issue refund — no order found with ID {order_id}."
    return f"[MOCK] Refund of ${amount} issued for order {order_id}."

def rag_lookup(question: str) -> str:
    answer, dist = rag_answer(question, collection)
    return answer

tool_config = {
    "get_order_status": {"function": get_order_status, "requires_approval": False},
    "issue_refund": {"function": issue_refund, "requires_approval": True},
    "rag_lookup": {"function": rag_lookup, "requires_approval": False}
}

tools = [
    {
        "type": "function",
        "function": {
            "name": "get_order_status",
            "description": "Looks up the status of a customer's order by order ID.",
            "parameters": {
                "type": "object",
                "properties": {"order_id": {"type": "string"}},
                "required": ["order_id"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "issue_refund",
            "description": "Issues a refund for a given order and amount.",
            "parameters": {
                "type": "object",
                "properties": {
                    "order_id": {"type": "string"},
                    "amount": {"type": "number"}
                },
                "required": ["order_id", "amount"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "rag_lookup",
            "description": "Looks up company policy information (shipping,  returns, refunds) to answer a customer's question.",
             "parameters": {
                "type": "object",
                "properties": {"question": {"type": "string"}},
                "required": ["question"]
            }
        }
    }
]

def run_support_agent(user_message):
    messages = [
        {
            "role": "system",
            "content": "You are a customer support agent. For policy questions (shipping, returns, refunds), you have context already retrieved for you when relevant. For order-specific questions, use the get_order_status tool. Only use issue_refund if the customer explicitly asks for a refund and you have confirmed the order details."
        }
    ]
    
    # rag_response, dist = rag_answer(user_message, collection)
    # if dist <= DISTANCE_THRESHOLD:
    #     messages.append({"role": "system", "content": f"Relevant policy context:\n{rag_response}"})
        
    messages.append({"role": "user", "content": user_message})
    while True:
        response = client.chat.completions.create(model=MODEL, messages=messages, tools=tools)
        msg = response.choices[0].message
        messages.append(msg)

        if msg.tool_calls:
            for call in msg.tool_calls:
                func_name = call.function.name
                func_args = json.loads(call.function.arguments)
                config = tool_config[func_name]

                if config["requires_approval"]:
                    print(f"\n[APPROVAL NEEDED] Agent wants to call: {func_name}({func_args})")
                    approval = input("Approve? (yes/no): ").strip().lower()
                    if approval != "yes":
                        result = "Action rejected by the user. Do not attempt this again; ask what they'd like instead."
                    else:
                        result = config["function"](**func_args)
                else:
                    result = config["function"](**func_args)

                print(f"[Tool result: {result}]")
                messages.append({
                    "role": "tool",
                    "tool_call_id": call.id,
                    "content": str(result)
                })
            continue
        else:
            print("\nFINAL ANSWER:", msg.content)
            break

if __name__ == "__main__":
    run_support_agent("I'd like a refund of $49.99 for order ORD1001")