import os
from neo4j import GraphDatabase
from transformers import AutoTokenizer, AutoModel
import torch
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

# 1. Connect to Neo4j
URI = "bolt://localhost:7687"
AUTH = ("neo4j", "researchos_password")

# 2. Connect to the LLM (OpenRouter)
api_key = os.environ.get("Open_Router_API")
client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=api_key
)

# 3. Load the EXACT same embedding model we used to vectorize the database
print("Loading local embedding model...")
tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')
model = AutoModel.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')

def embed_question(question: str):
    """
    Converts the user's plain english question into a mathematical vector.
    """
    inputs = tokenizer(question, padding=True, truncation=True, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
    embeddings = outputs.last_hidden_state.mean(dim=1)
    return embeddings[0].tolist()

def perform_graph_rag(question: str):
    """
    The magic of GraphRAG!
    Searches the Vector DB for text chunks, then traverses the Graph DB 
    for connected Authors, Methods, and Datasets to build the ultimate context.
    """
    print(f"\n🧠 Question: '{question}'")
    
    # Step 1: Embed the question
    question_vector = embed_question(question)
    
    # Step 2: Query Neo4j
    # This Cypher query does BOTH Vector Search AND Graph Traversal simultaneously!
    cypher_query = """
    // 1. Semantic Vector Search: Find the top 3 most relevant text chunks
    CALL db.index.vector.queryNodes('paper_chunks', 3, $question_vector)
    YIELD node AS chunk, score
    
    // 2. Graph Traversal: For those chunks, find the parent Paper and all its relationships
    MATCH (paper:Paper)-[:HAS_CHUNK]->(chunk)
    OPTIONAL MATCH (author:Author)-[:WROTE]->(paper)
    OPTIONAL MATCH (paper)-[:USES_METHOD]->(method:Method)
    OPTIONAL MATCH (paper)-[:EVALUATED_ON]->(dataset:Dataset)
    
    // 3. Aggregate all this rich context into a single JSON-like payload
    RETURN 
        paper.title AS title, 
        collect(DISTINCT chunk.text) AS relevant_text,
        collect(DISTINCT author.name) AS authors,
        collect(DISTINCT method.name) AS methods,
        collect(DISTINCT dataset.name) AS datasets
    """
    
    context = ""
    with GraphDatabase.driver(URI, auth=AUTH) as driver:
        with driver.session() as session:
            print("🔍 Searching Vector Index and Traversing Graph...")
            results = session.run(cypher_query, question_vector=question_vector)
            
            for record in results:
                title = record["title"]
                authors = ", ".join(record["authors"])
                methods = ", ".join(record["methods"])
                datasets = ", ".join(record["datasets"])
                text = " ".join(record["relevant_text"])
                
                # Format the context payload for the LLM
                context += f"\n\n--- PAPER: {title} ---\n"
                context += f"Authors: {authors}\n"
                context += f"Methods Used: {methods}\n"
                context += f"Datasets Evaluated On: {datasets}\n"
                context += f"Relevant Text Snippet: {text}\n"

    if not context:
        print("No relevant context found in the database.")
        return "I'm sorry, but I couldn't find any relevant information in the Knowledge Graph to answer your question."
        
    # Step 3: Feed the context to the LLM to generate the final answer
    print("🤖 Synthesizing answer using LLM...")
    prompt = f"""
    You are an expert AI Research Assistant. Answer the user's question using ONLY the provided context. 
    Do not hallucinate external information. If the answer is not in the context, say so.
    
    CONTEXT:
    {context}
    
    USER QUESTION: 
    {question}
    """
    
    response = client.chat.completions.create(
        model="openai/gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}]
    )
    
    answer = response.choices[0].message.content
    print("\n================ FINAL ANSWER ================\n")
    print(answer)
    print("\n==============================================")
    
    return answer


if __name__ == "__main__":
    # Feel free to change this question to test different things!
    test_question = "What datasets were evaluated in this research?"
    perform_graph_rag(test_question)
