import os
from typing import TypedDict
from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from pydantic import BaseModel, Field

from neo4j import GraphDatabase
from transformers import AutoTokenizer, AutoModel
import torch
from dotenv import load_dotenv

load_dotenv()

# 1. Connect to Neo4j
URI = "bolt://localhost:7687"
AUTH = ("neo4j", "researchos_password")

print("Loading local embedding model for LangGraph...")
tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')
model = AutoModel.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')

def embed_question(question: str):
    inputs = tokenizer(question, padding=True, truncation=True, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state.mean(dim=1)[0].tolist()

def retrieve_context(question: str):
    """Re-used logic from Phase 2 to get context from Neo4j"""
    question_vector = embed_question(question)
    cypher_query = """
    CALL db.index.vector.queryNodes('paper_chunks', 3, $question_vector)
    YIELD node AS chunk, score
    MATCH (paper:Paper)-[:HAS_CHUNK]->(chunk)
    OPTIONAL MATCH (author:Author)-[:WROTE]->(paper)
    OPTIONAL MATCH (paper)-[:USES_METHOD]->(method:Method)
    OPTIONAL MATCH (paper)-[:EVALUATED_ON]->(dataset:Dataset)
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
            results = session.run(cypher_query, question_vector=question_vector)
            for record in results:
                context += f"\n\n--- PAPER: {record['title']} ---\n"
                context += f"Authors: {', '.join(record['authors'])}\n"
                context += f"Methods: {', '.join(record['methods'])}\n"
                context += f"Datasets: {', '.join(record['datasets'])}\n"
                context += f"Text: {' '.join(record['relevant_text'])}\n"
    return context

# ==========================================
# Multi-Agent State & Schemas
# ==========================================
class ReviewerOutput(BaseModel):
    answer: str = Field(description="The final verified answer.")
    citations: list[str] = Field(description="A list of specific papers and authors used. Format: 'Title' by Author.")
    follow_up_questions: list[str] = Field(description="Exactly 3 smart follow-up questions the user could ask next.")

class AgentState(TypedDict):
    question: str
    context: str
    draft_answer: str
    final_answer: ReviewerOutput

# Initialize the LLM via OpenRouter for LangChain
api_key = os.environ.get("Open_Router_API")
llm = ChatOpenAI(
    model="openai/gpt-4o-mini",
    api_key=api_key,
    base_url="https://openrouter.ai/api/v1",
    temperature=0.2
)

# ==========================================
# Agent Nodes
# ==========================================
def retriever_node(state: AgentState):
    print("🕵️ Agent 1 (Retriever): Fetching Graph context from Neo4j...")
    context = retrieve_context(state["question"])
    return {"context": context}

def researcher_node(state: AgentState):
    print("📝 Agent 2 (Researcher): Drafting initial answer based strictly on context...")
    if not state.get("context"):
        return {"draft_answer": "No relevant information found."}
        
    messages = [
        SystemMessage(content="You are a meticulous researcher. Answer the user's question using ONLY the provided context."),
        HumanMessage(content=f"CONTEXT: {state['context']}\n\nQUESTION: {state['question']}")
    ]
    response = llm.invoke(messages)
    return {"draft_answer": response.content}

def reviewer_node(state: AgentState):
    print("🧐 Agent 3 (Reviewer): Checking Researcher's draft for hallucinations and formatting JSON...")
    if not state.get("context"):
        empty_output = ReviewerOutput(
            answer=state["draft_answer"], 
            citations=[], 
            follow_up_questions=["How does GraphRAG work?", "What can you do?"]
        )
        return {"final_answer": empty_output}
         
    messages = [
        SystemMessage(content="You are a strict reviewer. Check if the draft answer hallucinates. If it does, correct it. Output a strict JSON containing the final answer, citations, and 3 follow-up questions."),
        HumanMessage(content=f"CONTEXT: {state['context']}\n\nDRAFT ANSWER: {state['draft_answer']}\n\nQUESTION: {state['question']}")
    ]
    # Force the LLM to output our exact Pydantic schema
    structured_llm = llm.with_structured_output(ReviewerOutput)
    response = structured_llm.invoke(messages)
    return {"final_answer": response}

# ==========================================
# Build the LangGraph Workflow
# ==========================================
workflow = StateGraph(AgentState)
workflow.add_node("retriever", retriever_node)
workflow.add_node("researcher", researcher_node)
workflow.add_node("reviewer", reviewer_node)

# Define the sequence of agents
workflow.set_entry_point("retriever")
workflow.add_edge("retriever", "researcher")
workflow.add_edge("researcher", "reviewer")
workflow.add_edge("reviewer", END)

# Compile the graph into an executable agent!
research_agent = workflow.compile()

def run_agentic_workflow(question: str):
    print(f"\n🚀 Starting LangGraph Workflow for: '{question}'")
    # Invoke the graph with the initial state
    final_state = research_agent.invoke({"question": question})
    print("✅ Workflow Complete!\n")
    return final_state["final_answer"]

if __name__ == "__main__":
    answer = run_agentic_workflow("What datasets were evaluated?")
    print(answer)
