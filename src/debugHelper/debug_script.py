import os
import chromadb
from pinecone import Pinecone
from dotenv import load_dotenv

load_dotenv()

# --- Configuration ---
CHROMA_LOCAL_PATH = "vector_db"
PINECONE_INDEX_NAME = "daicus-rag"
# The embedding model 'all-MiniLM-L6-v2' has 384 dimensions. This is needed for the Pinecone query.
EMBEDDING_DIMENSION = 384 

def verify_all_sources():
    """
    Accurately verifies all unique 'sourceHelper' values stored in the vector databases
    by fetching from the entire collection/index, not just a search sample.
    """
    print("=" * 60)
    print("ACCURATE VERIFICATION OF ALL STORED SOURCES")
    print("=" * 60)

    # --- 1. Verify Local Chroma ---
    # This method directly accesses the collection and gets all metadata.
    print("\n--- LOCAL CHROMA ---")
    try:
        # Connect directly to the Chroma client
        client = chromadb.PersistentClient(path=CHROMA_LOCAL_PATH)
        # Get the collection (LangChain creates it with this name by default)
        collection = client.get_collection(name="langchain")
        
        # Get ALL items in the collection. We only need the metadata.
        # This is far more reliable than a semantic search.
        results = collection.get(include=["metadatas"])
        
        metadatas = results.get('metadatas', [])
        if not metadatas:
             print("No documents found in the Chroma collection.")
        else:
            source_helpers = {meta.get('sourceHelper', 'NOT_FOUND') for meta in metadatas}
            print(f"Found {len(source_helpers)} unique source(s) in the entire database:")
            for source in sorted(source_helpers):
                print(f"  - '{source}'")

    except Exception as e:
        print(f"ERROR connecting to or reading from Local Chroma: {e}")
        print("Hint: Did you run the ingestion script for local-chroma?")

    # --- 2. Verify Pinecone ---
    # Pinecone is query-based, so we query with a dummy vector to get a large,
    # broad sample of up to 10,000 documents from the index.
    print("\n--- PINECONE ---")
    try:
        pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
        index = pc.Index(PINECONE_INDEX_NAME)
        
        # Query for the max number of vectors (10k) to get a full picture.
        # We use a vector of zeros as a neutral "dummy" query.
        response = index.query(
            vector=[0] * EMBEDDING_DIMENSION, 
            top_k=10000, # Pinecone's max top_k
            include_metadata=True
        )
        
        matches = response.get('matches', [])
        if not matches:
            print("No documents found in the Pinecone index.")
        else:
            source_helpers = {match['metadata'].get('sourceHelper', 'NOT_FOUND') for match in matches}
            print(f"Found {len(source_helpers)} unique source(s) in the entire database:")
            for source in sorted(source_helpers):
                print(f"  - '{source}'")

    except Exception as e:
        print(f"ERROR connecting to or reading from Pinecone: {e}")
        print("Hint: Did you run the ingestion script for pinecone?")


if __name__ == "__main__":
    verify_all_sources()