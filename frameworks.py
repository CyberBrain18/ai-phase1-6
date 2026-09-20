from llama_index.core import VectorStoreIndex, Document, Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

Settings.embed_model = HuggingFaceEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2")

documents_text = {
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

docs = [Document(text=text, metadata={"source": name}) for name, text in documents_text.items()]

index = VectorStoreIndex.from_documents(docs)

query_engine = index.as_query_engine(similarity_top_k=2)
response = query_engine.query("How long does a refund take?")
print(response)