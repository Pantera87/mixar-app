import chromadb
from chromadb.utils import embedding_functions
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"
fn = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="BAAI/bge-large-en-v1.5", device=device
)
client = chromadb.PersistentClient(path=r"C:\Users\Thodoris\rag\blender_rag_db")
col = client.get_or_create_collection(name="blender_api", embedding_function=fn)

print(f"Collection 'blender_api' document count: {col.count()}")
if col.count() == 0:
    print("EMPTY — your ingestion script wrote elsewhere, or to a different collection name.")

for q in ["create a material", "add a light", "create a cube"]:
    print(f"\n=== Query: {q} ===")
    res = col.query(query_texts=[q], n_results=3)
    for doc, dist in zip(res["documents"][0], res["distances"][0]):
        print(f"--- (distance {dist:.3f}) ---")
        print(str(doc)[:300])
