import sys
import os
from fastapi import APIRouter
from api.schemas.requests import IngestRequest
from api.core.exceptions import GraphRAGException

# Add parent dir to path to import logic
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from ingest_arxiv import fetch_and_parse_arxiv
from extraction import extract_entities
from database import push_paper_to_graph
from vector_db import chunk_text, insert_chunks_into_graph

router = APIRouter(prefix="/ingest", tags=["Ingestion"])

@router.post("/")
async def ingest_papers(request: IngestRequest):
    try:
        papers = fetch_and_parse_arxiv(query=request.query, max_results=request.max_results)
        results = []
        for paper in papers:
            extracted_data = extract_entities(paper["text"])
            push_paper_to_graph(extracted_data)
            chunks = chunk_text(paper["text"], chunk_size=400, overlap=100)
            insert_chunks_into_graph(paper_title=paper["title"], chunks=chunks)
            results.append({"title": paper["title"], "status": "Successfully ingested!"})
            
        return {"message": f"Processed {len(papers)} papers.", "details": results}
    except Exception as e:
        raise GraphRAGException(str(e))
