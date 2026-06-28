import os
from neo4j import GraphDatabase
from transformers import AutoTokenizer, AutoModel
import torch

# Connect to the local Docker Neo4j instance
URI = "bolt://localhost:7687"
AUTH = ("neo4j", "researchos_password")

print("Loading local HuggingFace embedding model (all-MiniLM-L6-v2)...")
print("This runs 100% locally on your machine for free!")
# We use transformers directly since it was already installed by unstructured
tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')
model = AutoModel.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')

def chunk_text(text: str, chunk_size: int = 400, overlap: int = 100):
    """
    Splits a massive text into smaller overlapping chunks of words.
    Overlap ensures we don't accidentally cut a sentence in half and lose context.
    """
    words = text.split()
    chunks = []
    i = 0
    while i < len(words):
        chunk = " ".join(words[i:i + chunk_size])
        chunks.append(chunk)
        i += chunk_size - overlap
    return chunks

def get_embedding(text: str):
    """
    Converts a chunk of text into a mathematical vector (384 dimensions).
    """
    inputs = tokenizer(text, padding=True, truncation=True, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
    # Mean pooling to get a single vector representing the whole sentence
    embeddings = outputs.last_hidden_state.mean(dim=1)
    return embeddings[0].tolist()

def create_vector_index(driver):
    """
    Tells Neo4j to build a Vector Index optimized for Cosine Similarity search.
    """
    with driver.session() as session:
        # Create vector index if it doesn't exist on Chunk nodes
        # Our MiniLM model outputs 384 dimensions
        session.run("""
        CREATE VECTOR INDEX paper_chunks IF NOT EXISTS
        FOR (c:Chunk)
        ON (c.embedding)
        OPTIONS {indexConfig: {
            `vector.dimensions`: 384,
            `vector.similarity_function`: 'cosine'
        }}
        """)
        print("Neo4j Vector Index created/verified.")

def insert_chunks_into_graph(paper_title: str, chunks: list[str]):
    """
    Inserts vector chunks into Neo4j and connects them to the Parent Paper node.
    """
    with GraphDatabase.driver(URI, auth=AUTH) as driver:
        create_vector_index(driver)
        
        with driver.session() as session:
            for i, chunk_text_content in enumerate(chunks):
                print(f"Vectorizing chunk {i+1}/{len(chunks)}...")
                embedding = get_embedding(chunk_text_content)
                
                # MERGE the chunk, set embedding, and link to the Paper
                session.run("""
                MATCH (p:Paper {title: $title})
                MERGE (c:Chunk {id: $chunk_id})
                SET c.text = $text, c.embedding = $embedding
                MERGE (p)-[:HAS_CHUNK]->(c)
                """,
                title=paper_title,
                chunk_id=f"{paper_title}_chunk_{i}",
                text=chunk_text_content,
                embedding=embedding
                )
    print(f"✅ Successfully vectorized and inserted {len(chunks)} chunks for '{paper_title}'.")

if __name__ == "__main__":
    from ingest_arxiv import fetch_and_parse_arxiv
    
    print("\n--- Phase 2: Vector Database Ingestion ---")
    print("Fetching 1 paper for Vectorization testing...")
    
    # We grab 1 paper to test our new Vectorization logic
    papers = fetch_and_parse_arxiv(query="GraphRAG", max_results=1)
    
    if papers:
        paper = papers[0]
        print(f"\n--- Chunking text for: {paper['title']} ---")
        
        # 1. Chunk the massive text
        chunks = chunk_text(paper["text"], chunk_size=400, overlap=100)
        print(f"Split the paper into {len(chunks)} overlapping chunks.")
        
        # 2. Embed and insert into Neo4j
        insert_chunks_into_graph(paper_title=paper["title"], chunks=chunks)
    else:
        print("No papers found.")
