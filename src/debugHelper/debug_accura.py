# Quick debug version - test just one query to see what's happening
import os
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

load_dotenv()

CHROMA_LOCAL_PATH = "vector_db"
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

def debug_single_query():
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
    vector_store = Chroma(persist_directory=CHROMA_LOCAL_PATH, embedding_function=embeddings)
    
    # Test query
    query = "What is the maximum payment for a bail bond under the Allstate policy?"
    target_source = "allstatepolicy.pdf"
    
    print(f"Testing query: {query}")
    print(f"Looking for source: {target_source}")
    
    # First, do a general search to see what we get
    print("\n--- GENERAL SEARCH ---")
    general_retriever = vector_store.as_retriever()
    general_docs = general_retriever.invoke(query)
    
    print(f"General search returned {len(general_docs)} documents")
    for i, doc in enumerate(general_docs):
        print(f"Doc {i+1}:")
        print(f"  Content: {doc.page_content[:100]}...")
        print(f"  All metadata: {doc.metadata}")
        print(f"  sourceHelper: {doc.metadata.get('sourceHelper', 'MISSING')}")
        print()
    
    # Now try filtered search with different approaches
    print("\n--- FILTERED SEARCH ATTEMPTS ---")
    
    # Attempt 1: Direct filter
    try:
        print("Attempt 1: Direct filter")
        filtered_retriever = vector_store.as_retriever(
            search_kwargs={'filter': {'sourceHelper': target_source}}
        )
        filtered_docs = filtered_retriever.invoke(query)
        print(f"✓ Success: {len(filtered_docs)} documents")
    except Exception as e:
        print(f"✗ Failed: {e}")
    
    # Attempt 2: Check if the field exists with a different case
    print("\nAttempt 2: Trying different field names")
    possible_field_names = ['sourceHelper', 'sourcehelper', 'SourceHelper', 'source', 'source_helper']
    
    for field_name in possible_field_names:
        try:
            filtered_retriever = vector_store.as_retriever(
                search_kwargs={'filter': {field_name: target_source}}
            )
            filtered_docs = filtered_retriever.invoke(query)
            print(f"✓ Field '{field_name}': {len(filtered_docs)} documents")
            if filtered_docs:
                break
        except Exception as e:
            print(f"✗ Field '{field_name}': {e}")
    
    # Attempt 3: Try with original filename format
    print("\nAttempt 3: Trying different source values")
    possible_source_values = [
        "allstatepolicy.pdf",
        "AllstatePolicy.pdf", 
        "data/raw/AllstatePolicy.pdf",
        "data/raw/allstatepolicy.pdf"
    ]
    
    for source_value in possible_source_values:
        try:
            filtered_retriever = vector_store.as_retriever(
                search_kwargs={'filter': {'sourceHelper': source_value}}
            )
            filtered_docs = filtered_retriever.invoke(query)
            print(f"✓ Source '{source_value}': {len(filtered_docs)} documents")
            if filtered_docs:
                break
        except Exception as e:
            print(f"✗ Source '{source_value}': {e}")

if __name__ == "__main__":
    debug_single_query()