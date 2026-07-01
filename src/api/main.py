import sys
import os
import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# Add parent directory to python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from api.routers import chat, ingestion

app = FastAPI(
    title="ResearchOS API",
    description="Enterprise Multi-Agent GraphRAG Backend",
    version="2.0.0"
)

# Enable CORS for the frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include Routers
app.include_router(chat.router)
app.include_router(ingestion.router)

if __name__ == "__main__":
    print("Starting Enterprise ResearchOS API...")
    uvicorn.run("api.main:app", host="0.0.0.0", port=8000, reload=True)
