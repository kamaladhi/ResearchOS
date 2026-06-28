#information extraction module from pdfs using LLM. using openrouter

import os
import json
from pydantic import BaseModel, Field
from openai import OpenAI
from dotenv import load_dotenv

# Load environment variables (like OPENAI_API_KEY) from a .env file
load_dotenv()

# ==========================================
# 1. Schema Definition (The Knowledge Graph)
# ==========================================
# Pydantic forces the LLM to output exactly these fields in JSON format.

class Author(BaseModel):
    name: str

class Method(BaseModel):
    name: str
    description: str

class Dataset(BaseModel):
    name: str

class PaperExtraction(BaseModel):
    title: str
    authors: list[Author]
    methods_used: list[Method] = Field(description="Methods, algorithms, or models used in this paper")
    datasets_used: list[Dataset] = Field(description="Datasets the authors evaluated their work on")
    research_gap: str = Field(description="What problem or gap in previous research are they solving?")
    key_findings: str = Field(description="The main conclusion or result of the paper")

# ==========================================
# 2. LLM Extraction Engine
# ==========================================

def extract_entities(paper_text: str) -> PaperExtraction:
    """
    Sends the raw text to an LLM and forces it to return structured JSON.
    """
    api_key = os.environ.get("OPEN_ROUTER_API") or os.environ.get("OPENAI_API_KEY")
    base_url = "https://openrouter.ai/api/v1" if os.environ.get("OPEN_ROUTER_API") else "https://api.openai.com/v1"
    
    client = OpenAI(
        base_url=base_url,
        api_key=api_key
    )
    
    # A full paper can be 100k+ chars. To save money during our MVP, 
    # we'll only feed it the first 40,000 characters (usually covers Abstract, Intro, and Methods).
    truncated_text = paper_text[:40000] 

    print("Sending text to LLM for extraction... (This might take 10-20 seconds)")
    
    # We use OpenAI's new 'Structured Outputs' feature (.parse)
    response = client.beta.chat.completions.parse(
        model="openai/gpt-4o-mini", # Works for both OpenRouter and OpenAI
        messages=[
            {"role": "system", "content": "You are an expert AI designed to extract structured scientific knowledge from academic literature."},
            {"role": "user", "content": f"Extract the requested entities from this paper text:\n\n{truncated_text}"}
        ],
        response_format=PaperExtraction, # <--- The magic happens here
    )

    return response.choices[0].message.parsed

if __name__ == "__main__":
    # For testing, we'll import our previous script to get 1 paper
    from ingest_arxiv import fetch_and_parse_arxiv
    
    print("Fetching 1 paper for LLM Extraction testing...")
    papers = fetch_and_parse_arxiv(query="GraphRAG", max_results=1)
    
    if papers:
        paper = papers[0]
        print(f"\n--- Extracting Entities for: {paper['title']} ---")
        
        # Call the LLM
        extracted_data = extract_entities(paper["text"])
        
        # Print the structured JSON result
        print("\n=== KNOWLEDGE GRAPH ENTITIES EXTRACTED ===")
        print(extracted_data.model_dump_json(indent=2))
    else:
        print("No papers found to process.")
