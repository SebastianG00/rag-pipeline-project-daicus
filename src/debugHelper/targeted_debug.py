import os
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

load_dotenv()

CHROMA_LOCAL_PATH = "vector_db"
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

def test_filter_mechanism():
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
    vector_store = Chroma(persist_directory=CHROMA_LOCAL_PATH, embedding_function=embeddings)
    
    # Get a few docs without filtering to see what we actually have
    retriever = vector_store.as_retriever(search_kwargs={'k': 5})
    docs = retriever.invoke("insurance")
    
    print("Available documents and their sourceHelper values:")
    print("=" * 60)
    unique_sources = set()
    for doc in docs:
        source_helper = doc.metadata.get('sourceHelper')
        unique_sources.add(source_helper)
        print(f"sourceHelper: '{source_helper}'")
        print(f"Full metadata keys: {list(doc.metadata.keys())}")
        print("-" * 40)
    
    print(f"\nUnique sourceHelper values: {unique_sources}")
    
    # Now test filtering with EXACT values we found
    print("\n" + "=" * 60)
    print("TESTING FILTERS WITH EXACT VALUES")
    print("=" * 60)
    
    for source in unique_sources:
        if source:  # Skip None values
            print(f"\nTesting filter with EXACT value: '{source}'")
            
            # Test the filter syntax that should work for Chroma
            try:
                # Method 1: Using search_kwargs
                filtered_retriever = vector_store.as_retriever(
                    search_kwargs={
                        'filter': {'sourceHelper': source},
                        'k': 10
                    }
                )
                filtered_docs = filtered_retriever.invoke("insurance")
                print(f"  ✓ Method 1 (search_kwargs): Found {len(filtered_docs)} documents")
                
            except Exception as e:
                print(f"  ✗ Method 1 failed: {e}")
                
            try:
                # Method 2: Using similarity_search with filter
                filtered_docs = vector_store.similarity_search(
                    "insurance",
                    filter={'sourceHelper': source},
                    k=10
                )
                print(f"  ✓ Method 2 (similarity_search): Found {len(filtered_docs)} documents")
                
            except Exception as e:
                print(f"  ✗ Method 2 failed: {e}")
            
            try:
                # Method 3: Different filter syntax
                filtered_docs = vector_store.similarity_search(
                    "insurance",
                    filter={'sourceHelper': {'$eq': source}},
                    k=10
                )
                print(f"  ✓ Method 3 (nested syntax): Found {len(filtered_docs)} documents")
                
            except Exception as e:
                print(f"  ✗ Method 3 failed: {e}")

if __name__ == "__main__":
    test_filter_mechanism()