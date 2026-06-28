#Code base for ingestion and parsing of the pdf

import arxiv
import os
from unstructured.partition.pdf import partition_pdf
import requests

"""
=============================================================================
ARCHITECTURAL DECISION RECORD (ADR)
Why arXiv for Phase 1 (MVP)?
----------------------------
For this MVP, we use the arXiv API because it is free, open, and optimized 
for programmatic bulk PDF downloads without CAPTCHAs or IP blocking.

Future Scalability (V2):
arXiv is a preprint server and lacks official metadata for Journal Quartiles 
(Q1, Q2, Q3) or strict Journal vs. Conference classifications. 
In a production enterprise environment, this ingestion module would be 
swapped/augmented with the Semantic Scholar API or OpenAlex API. We would 
query those APIs for `venueType` and cross-reference an SJR (SCImago Journal 
Rank) database during the Ingestion Stage to ensure only high-impact 
peer-reviewed literature enters the Knowledge Graph.
=============================================================================
"""



def fetch_and_parse_arxiv(query: str, max_results: int = 5, download_dir: str = "data/pdfs"):
    """
    Fetches PDFs from arXiv based on a search query and extracts their raw text.
    Query : hints what we are looking for in the pdf.
    Max_result controls how many papers we want to download and 
    download_dir stores the downloaded paper.
    """
    os.makedirs(download_dir, exist_ok=True)
    
    #1. Search arXiv API
    print(f"Searching arXiv for: '{query}'")
    search = arxiv.Search(
        query=query,
        max_results=max_results,
        sort_by=arxiv.SortCriterion.Relevance # using arxiv api to fetch pdfs using sort_by Relevence
    )
    
    parsed_papers = [] # this list 
    
    #Configure the client with retries and a delay to prevent 429 Rate Limits
    client = arxiv.Client(
        page_size=10,
        delay_seconds=5.0,
        num_retries=5
    )
    
    for result in client.results(search):
        print(f"\nProcessing: {result.title}")
        pdf_path = os.path.join(download_dir, f"{result.get_short_id()}.pdf")
        
        #2. Download the PDF locally
        if not os.path.exists(pdf_path):
            print(f"Downloading {result.pdf_url} to {pdf_path}...")
            response = requests.get(result.pdf_url)
            with open(pdf_path, 'wb') as f:
                f.write(response.content)
        else:
            print(f"File already exists: {pdf_path}")
            
        #3. Parse the PDF using Unstructured
        print("Parsing PDF text (this might take a moment)...")
        try:
            # partition_pdf extracts text, lists, and tables intelligently
            elements = partition_pdf(filename=pdf_path)
            
            #Combine all the text elements into one giant string for the LLM
            full_text = "\n\n".join([str(el) for el in elements])
            print(f"Successfully extracted {len(full_text)} characters.")
            
            parsed_papers.append({
                "id": result.get_short_id(),
                "title": result.title,
                "authors": [author.name for author in result.authors],
                "summary": result.summary,
                "text": full_text
            })
        except Exception as e:
            print(f"Error parsing PDF: {e}")
            
    return parsed_papers

'''
=============================================================================
V2 ARCHITECTURE PREVIEW (PORTFOLIO SHOWCASE)
The following code demonstrates how we would upgrade this system to use 
Semantic Scholar for strict Journal/Conference filtering. 
It is commented out for the MVP Phase 1.
=============================================================================
def fetch_from_semantic_scholar(query: str, require_q1_journal: bool = True):
    """
    Production-grade ingestion using Semantic Scholar Graph API.
    """
    import requests
    
    print(f"Searching Semantic Scholar for: {query}...")
    url = "https://api.semanticscholar.org/graph/v1/paper/search"
    
    Notice how we can explicitly filter for Journal Articles!
    params = {
        "query": query,
        "publicationTypes": "JournalArticle", 
        "fields": "title,authors,venue,year,openAccessPdf",
        "limit": 5
    }
    
    In a real system, we would inject our API key here to bypass rate limits
    headers = {"x-api-key": "YOUR_S2_API_KEY"}
    
    response = requests.get(url, params=params, headers=headers)
    papers = response.json().get("data", [])
    
    valid_papers = []
    for paper in papers:
        venue_name = paper.get("venue")
        
        if require_q1_journal:
            We would cross-reference venue_name with a local SCImago (SJR) database here
            if is_q1_journal(venue_name):
                valid_papers.append(paper)
            pass
            
        We would then download the PDF using paper['openAccessPdf']['url']
        and pass it to unstructured.partition_pdf() just like we did with arXiv!
        
    return valid_papers
=============================================================================
'''
if __name__ == "__main__":
    #We'll just grab 2 papers for testing so it doesn't take forever
    papers = fetch_and_parse_arxiv(query="Large Language Models Agent Reasoning", max_results=2)
    
    print("\n--- Summary of Extraction ---")
    for p in papers:
        print(f"- {p['title']} ({len(p['text'])} chars extracted)")
