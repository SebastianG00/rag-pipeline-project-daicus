import os
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone import Pinecone as LangChainPinecone
import chromadb

load_dotenv()

# Configuration
CHROMA_LOCAL_PATH = "vector_db"
PINECONE_INDEX_NAME = "daicus-rag"
CHROMA_CLOUD_COLLECTION_NAME = "daicus-rag-cloud"
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

def inspect_metadata(db_type: str, embeddings: HuggingFaceEmbeddings):
    """
    Inspect the metadata structure of documents in each vector database
    """
    print(f"\n{'='*60}")
    print(f"INSPECTING METADATA FOR {db_type.upper()}")
    print(f"{'='*60}")
    
    vector_store = None
    
    try:
        # Initialize the appropriate vector store
        if db_type == "local-chroma":
            vector_store = Chroma(persist_directory=CHROMA_LOCAL_PATH, embedding_function=embeddings)
        elif db_type == "pinecone":
            vector_store = LangChainPinecone.from_existing_index(PINECONE_INDEX_NAME, embeddings)
        elif db_type == "chroma-cloud":
            client = chromadb.CloudClient(
                tenant=os.getenv("CHROMA_TENANT"),
                database=os.getenv("CHROMA_DATABASE"),
                api_key=os.getenv("CHROMA_API_KEY")
            )
            vector_store = Chroma(
                client=client, 
                collection_name=CHROMA_CLOUD_COLLECTION_NAME,
                embedding_function=embeddings
            )
        
        # Get a sample of documents to inspect metadata
        retriever = vector_store.as_retriever(search_kwargs={'k': 10})
        sample_docs = retriever.invoke("insurance policy")
        
        print(f"Retrieved {len(sample_docs)} sample documents")
        print(f"\n--- METADATA INSPECTION ---")
        
        # Group documents by their metadata to see patterns
        metadata_patterns = {}
        unique_sources = set()
        
        for i, doc in enumerate(sample_docs):
            print(f"\nDocument {i+1}:")
            print(f"  Content preview: {doc.page_content[:100]}...")
            print(f"  Metadata: {doc.metadata}")
            
            # Check for sourceHelper specifically
            source_helper = doc.metadata.get('sourceHelper')
            if source_helper:
                unique_sources.add(source_helper)
            
            # Create a pattern key for grouping similar metadata structures
            metadata_keys = tuple(sorted(doc.metadata.keys()))
            if metadata_keys not in metadata_patterns:
                metadata_patterns[metadata_keys] = []
            metadata_patterns[metadata_keys].append(doc.metadata)
        
        print(f"\n--- SUMMARY ---")
        print(f"Unique sourceHelper values found: {unique_sources}")
        print(f"Number of different metadata patterns: {len(metadata_patterns)}")
        
        for pattern_keys, examples in metadata_patterns.items():
            print(f"\nMetadata pattern with keys: {pattern_keys}")
            print(f"  Number of documents with this pattern: {len(examples)}")
            print(f"  Example metadata: {examples[0]}")
        
        # Test specific filtering
        print(f"\n--- TESTING SPECIFIC FILTERS ---")
        
        # Test each unique source
        for source in list(unique_sources)[:3]:  # Test first 3 sources
            print(f"\nTesting filter for sourceHelper='{source}':")
            
            try:
                if db_type == "pinecone":
                    # Pinecone syntax
                    filtered_retriever = vector_store.as_retriever(
                        search_kwargs={'filter': {'sourceHelper': {'$eq': source}}}
                    )
                else:
                    # Chroma syntax
                    filtered_retriever = vector_store.as_retriever(
                        search_kwargs={'filter': {'sourceHelper': source}}
                    )
                
                filtered_docs = filtered_retriever.invoke("insurance")
                print(f"  ✓ Successfully retrieved {len(filtered_docs)} documents")
                
                if filtered_docs:
                    print(f"  First result sourceHelper: {filtered_docs[0].metadata.get('sourceHelper')}")
                
            except Exception as e:
                print(f"  ✗ Filter failed: {e}")
        
        # Test if documents exist without filtering
        print(f"\n--- TESTING GENERAL SEARCH ---")
        try:
            general_retriever = vector_store.as_retriever()
            general_docs = general_retriever.invoke("Allstate policy")
            print(f"General search for 'Allstate policy' returned {len(general_docs)} documents")
            
            if general_docs:
                print("Sources found in general search:")
                for doc in general_docs:
                    source_helper = doc.metadata.get('sourceHelper', 'NO_SOURCE_HELPER')
                    print(f"  - {source_helper}")
        
        except Exception as e:
            print(f"General search failed: {e}")
            
    except Exception as e:
        print(f"ERROR initializing {db_type}: {e}")

def main():
    """Main function to inspect all vector databases"""
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
    
    # Inspect each database
    inspect_metadata("local-chroma", embeddings)
    inspect_metadata("pinecone", embeddings)
    inspect_metadata("chroma-cloud", embeddings)
    
    print(f"\n{'='*60}")
    print("INSPECTION COMPLETE")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()