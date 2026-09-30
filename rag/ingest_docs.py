import os
import re
import chromadb
from chromadb.utils import embedding_functions
from langchain_text_splitters import RecursiveCharacterTextSplitter
from bs4 import BeautifulSoup
import torch

# ==========================================
# 🚀 PERFORMANCE FIX: AMD 7900XTX ALLOCATION
# ==========================================
# Modern PyTorch builds with AMD ROCm map directly to the "cuda" string identifier.
# This assigns your 24GB VRAM pool to handle the matrix processing workload.
amd_device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"🌟 Initializing Ingestion Pipeline on Device: {amd_device.upper()} (Target: RX 7900 XTX)")

if amd_device == "cpu":
    print("⚠️ WARNING: GPU acceleration wasn't found. Processing will fall back to CPU.")
    print("Run this to fix: python -m pip install torch --index-url https://pytorch.org --force-reinstall")

# 1. Initialize local persistent DB — anchored to THIS script's folder so the
#    ingest always writes into the same database the server reads from
#    (mixar-rag-server.py uses SCRIPT_DIR/blender_rag_db), regardless of the
#    current working directory the script is launched from.
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DB_DIR = os.path.join(SCRIPT_DIR, "blender_rag_db")
chroma_client = chromadb.PersistentClient(path=DB_DIR)

# Force the embedding model straight into your GPU VRAM
embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="BAAI/bge-large-en-v1.5",
    device=amd_device
)
collection = chroma_client.get_or_create_collection(
    name="blender_api", embedding_function=embedding_fn
)

# 2. Text splitter tailored to handle bpy structural code snippets
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=700,
    chunk_overlap=70,
    separators=["\n\n", "\n", " ", ""]
)

# Anchor the docs folder to this script too (the cached manual pages live in
# docs_cache next to the ingest script).
docs_dir = os.path.join(SCRIPT_DIR, "docs_cache")
doc_id = 0
chunk_n = {}  # per-file chunk counter for source-scoped ids

print(f"\n📂 Scanning directory: {docs_dir}")
print("Processing Blender documentation files...")

# Batch variables to group operations for lightning-fast database writes
documents_batch = []
metadatas_batch = []
ids_batch = []
BATCH_SIZE = 100  # Pushing chunks in bulk keeps GPU memory utilized efficiently

for root, dirs, files in os.walk(docs_dir):
    for filename in files:
        if (
            filename.endswith(".rst") or filename.endswith(".md")
            or (
                filename.endswith(".html")
                and filename.lower() not in ("index.html", "genindex.html")
            )  # Sphinx index pages are pure link walls — useless for RAG
        ):
            file_path = os.path.join(root, filename)
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    # Strip template junk out of HTML files
                    if filename.endswith(".html"):
                        soup = BeautifulSoup(f.read(), "html.parser")
                        # Strip navigation chrome (sidebars, TOC, breadcrumbs) BEFORE
                        # text extraction — otherwise the page's nav link list gets
                        # embedded and poisons semantic search with title walls.
                        for tag in soup.find_all(["nav", "aside", "header", "footer"]):
                            tag.decompose()
                        for tag in soup.find_all(
                            class_=re.compile(r"toc|breadcrumb|sidebar|menu|pagination", re.I)
                        ):
                            tag.decompose()
                        main_content = soup.find(role="main") or soup.find("body")
                        if not main_content:
                            continue
                        raw_text = main_content.get_text(separator="\n")
                    else:
                        raw_text = f.read()
                        
                    chunks = text_splitter.split_text(raw_text)
                    
                    for chunk in chunks:
                        if len(chunk.strip()) > 50:  # Skip empty artifacts
                            documents_batch.append(chunk)
                            metadatas_batch.append({"source": filename})
                            # Source-scoped id so re-ingesting a page replaces
                            # exactly that page's chunks (via upsert below)
                            # instead of colliding with the global id_N sequence.
                            n = chunk_n.get(filename, 0)
                            ids_batch.append(f"{filename}::{n}")
                            chunk_n[filename] = n + 1
                            doc_id += 1
                            
                            # Push in large batches to maximize GPU throughput
                            if len(documents_batch) >= BATCH_SIZE:
                                collection.upsert(
                                    documents=documents_batch,
                                    metadatas=metadatas_batch,
                                    ids=ids_batch
                                )
                                documents_batch, metadatas_batch, ids_batch = [], [], []
                                print(f"⚡ Processed and indexed {doc_id} snippets...")
                                
            except Exception as e:
                print(f"Skipping corrupt file {filename}: {e}")

# Flush any remaining items in the queue
if documents_batch:
    collection.upsert(
        documents=documents_batch,
        metadatas=metadatas_batch,
        ids=ids_batch
    )

print(f"\n🎉 SUCCESS: Indexed {doc_id} Blender API documentation snippets into 'blender_rag_db' using your GPU!")
