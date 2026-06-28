from neo4j import GraphDatabase
import json

# Connect to the local Docker Neo4j instance we defined in docker-compose.yml
URI = "bolt://localhost:7687"
AUTH = ("neo4j", "researchos_password")

def push_paper_to_graph(extracted_data):
    """
    Takes the structured data from our LLM and runs Cypher queries
    to insert it into the Neo4j database as interconnected nodes.
    """
    # If the data is passed as a Pydantic model, convert to dict
    if not isinstance(extracted_data, dict):
        data = extracted_data.model_dump()
    else:
        data = extracted_data
        
    print(f"Connecting to Neo4j to inject: '{data['title']}'...")
    
    # We use Neo4j's Python Driver
    with GraphDatabase.driver(URI, auth=AUTH) as driver:
        with driver.session() as session:
            
            # 1. Create the Paper Node (MERGE prevents duplicates)
            session.run(
                "MERGE (p:Paper {title: $title}) "
                "SET p.research_gap = $gap, p.key_findings = $findings",
                title=data["title"], 
                gap=data["research_gap"], 
                findings=data["key_findings"]
            )
            
            # 2. Create Author Nodes and Link them to the Paper
            for author in data["authors"]:
                session.run(
                    "MERGE (a:Author {name: $author_name}) "
                    "WITH a "
                    "MATCH (p:Paper {title: $title}) "
                    "MERGE (a)-[:WROTE]->(p)",
                    author_name=author["name"],
                    title=data["title"]
                )
                
            # 3. Create Method Nodes and Link them
            for method in data["methods_used"]:
                session.run(
                    "MERGE (m:Method {name: $method_name}) "
                    "SET m.description = $desc "
                    "WITH m "
                    "MATCH (p:Paper {title: $title}) "
                    "MERGE (p)-[:USES_METHOD]->(m)",
                    method_name=method["name"],
                    desc=method["description"],
                    title=data["title"]
                )
                
            # 4. Create Dataset Nodes and Link them
            for dataset in data["datasets_used"]:
                session.run(
                    "MERGE (d:Dataset {name: $dataset_name}) "
                    "WITH d "
                    "MATCH (p:Paper {title: $title}) "
                    "MERGE (p)-[:EVALUATED_ON]->(d)",
                    dataset_name=dataset["name"],
                    title=data["title"]
                )
                
    print("✅ Successfully constructed Subgraph in Neo4j!")

if __name__ == "__main__":
    print("This is a module meant to be imported by the main script.")
