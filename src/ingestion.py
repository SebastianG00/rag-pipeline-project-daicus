import os
import shutil
import argparse
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.document_loaders import BSHTMLLoader
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_pinecone import Pinecone as LangChainPinecone
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec
# Importing the base chromadb client
import chromadb 
import time

#load environment variables from .env file
load_dotenv()

# --- CONFIGURATION ---
# Centralizing constants makes the script easier to manage and modify.
# An intern can quickly see and change key parameters without hunting through the code.
DATA_DIR = "data/raw/"
CHROMA_DIR = "vector_db"
PINECONE_INDEX_NAME = "daicus-rag"
CHROMA_CLOUD_COLLECTION_NAME = "daicus-rag-cloud"
# Using a specific model name ensures that the same embeddings are used everywhere.
# Changing this model would require re-ingesting all documents.
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

def load_documents():
    """
    Load documents from the data/raw directory.
    Supports PDF and HTML formats.
    """
    documents = []
    # This dictionary maps file extensions to their corresponding LangChain loader class.
    loader_mapper = {
        ".pdf": PyPDFLoader,
        ".html": BSHTMLLoader,
    }
    
    print("Loading documents from:", DATA_DIR)

    for filename in os.listdir(DATA_DIR):
        file_path = os.path.join(DATA_DIR, filename)
        file_extension = os.path.splitext(filename)[1].lower()

        if file_extension in loader_mapper:
            try:
                loaderType = loader_mapper[file_extension]
                loader = loaderType(file_path)
                loaded_docs = loader.load()

                # --- METADATA STANDARDIZATION ---
                # This is a critical best practice for RAG. We create a 'sourceHelper' field 
                # with a clean, all-lowercase version of the filename. This prevents 
                # case-sensitivity errors when we later try to filter by source document.

                # It ensures the "writer" (ingestion) and "reader" (retrieval) speak the same language!!!!
                for doc in loaded_docs:
                    # FIXED: Ensure consistent lowercase naming
                    original_filename = os.path.basename(doc.metadata['source'])
                    doc.metadata['sourceHelper'] = original_filename.lower()
                    

                documents.extend(loaded_docs)
                print(f"Loaded {len(loaded_docs)} pages from {filename}")
            except Exception as e:
                print(f"Error loading {filename}: {e}")
        else:
            print(f"Unsupported file type: {file_extension} for file {filename}")
    return documents

def chunk_documents(documents: list[Document], chunk_size: int = 1000, chunk_overlap: int = 100):
    """
    Chunk documents into smaller pieces for processing.
    """
    print("\nChunking documents...")
    text_splitter = RecursiveCharacterTextSplitter(
        # chunk_size is a key parameter to tune for performance. Too large, and you get noise;
        # too small, and you lose context.
        chunk_size=chunk_size,
        # chunk_overlap ensures that context is not lost at the boundaries of chunks.
        # This is also a key parameter to tune. It should be large enough to maintain context
        chunk_overlap=chunk_overlap,
        length_function=len,
        add_start_index=True,
    )

    chunkies = text_splitter.split_documents(documents)
    print(f"Split {len(documents)} document pages into {len(chunkies)} chunks.")

    return chunkies 


def embedStoreChromaLocal(chunkies: list[Document], embeddings: HuggingFaceEmbeddings):
    """
    Stores the embedded chunks in a local Chroma vector store.
    """
    print("\n--- Storing in Local Chroma ---")
    if os.path.exists(CHROMA_DIR):
        print(f"Clearing existing Chroma vector store at: {CHROMA_DIR}")
        shutil.rmtree(CHROMA_DIR)
        
    # This single command handles embedding all chunks and storing them in the local directory.
    vectorStoredChroma = Chroma.from_documents(chunkies, embeddings, persist_directory=CHROMA_DIR)
    print(f"Successfully created vector store with {vectorStoredChroma._collection.count()} documents.")

def embedStorePinecone(chunkies: list[Document], embeddings: HuggingFaceEmbeddings):
    """
    Stores the embedded chunks in a Pinecone vector store, handling large uploads
    by batching them to stay under the API request size limit.
    """
    print("\n--- Storing in Pinecone with batching ---")
    
    pinecone = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
    # IMPORTANT!!!
    # The embedding model's dimension must match the Pinecone index's dimension.
    # We get this programmatically to avoid hardcoding a number that might change if the model is updated.
    embedding_dimension = len(embeddings.embed_query("test"))
    
    # Ensure the index exists, create if not
    if PINECONE_INDEX_NAME not in pinecone.list_indexes().names():
        print(f"Creating new Pinecone index: {PINECONE_INDEX_NAME}")
        pinecone.create_index(
            name=PINECONE_INDEX_NAME,
            dimension=embedding_dimension,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1")
        )
        print("Waiting for index to be ready...")
        time.sleep(10)
    else:
        print(f"Using existing Pinecone index: {PINECONE_INDEX_NAME}. Clearing contents...")
        index = pinecone.Index(PINECONE_INDEX_NAME)
        index.delete(delete_all=True)

    # Instead of from_documents, we connect to the index and add documents in batches.
    # This gives us control over the request size.
    vector_store = LangChainPinecone.from_existing_index(
        index_name=PINECONE_INDEX_NAME,
        embedding=embeddings
    )
    # IMPORTANT!!!
    # --- BATCHING ---
    # Pinecone has a request size limit (e.g., 2MB). Trying to upload thousands of documents
    # in a single request will fail. We process them in smaller batches to stay under this limit.

    batch_size = 100  # A safe batch size to stay under the 2MB limit. You can tune this.
    total_chunks = len(chunkies)
    print(f"Beginning upload of {total_chunks} chunks to Pinecone in batches of {batch_size}...")

    for i in range(0, total_chunks, batch_size):
        # Get the batch of chunks to upload
        batch = chunkies[i:i + batch_size]
        
        # The add_documents method handles embedding and upserting for this small batch
        vector_store.add_documents(batch)
        
        print(f"  - Upserted batch {i//batch_size + 1}/{(total_chunks + batch_size - 1)//batch_size}...")

    print(f"Successfully uploaded {total_chunks} chunks to the Pinecone index.")

def embedStoreChromaCloud(chunkies: list[Document], embeddings: HuggingFaceEmbeddings):
    """
    Stores the embedded chunks in Chroma Cloud, handling large uploads by
    batching them to stay under the API record count limit.
    """
    print("\n--- Storing in Chroma Cloud with batching ---")
    
    # Initialize the cloud client
    client = chromadb.CloudClient(
        tenant=os.getenv("CHROMA_TENANT"),
        database=os.getenv("CHROMA_DATABASE"),
        api_key=os.getenv("CHROMA_API_KEY")
    )
    
    # Get or create the collection
    collection = client.get_or_create_collection(CHROMA_CLOUD_COLLECTION_NAME)
    
    # Clearing a cloud collection is more complex than a local one.
    # We must fetch the IDs of existing documents in batches and then delete them.
    # This is necessary because there is no simple "delete_all" command for a cloud collection.
    count = collection.count()
    if count > 0:
        print(f"Using existing Chroma Cloud collection: {CHROMA_CLOUD_COLLECTION_NAME}. Clearing {count} contents in batches...")

        # Loop to get and delete documents in batches of 100
        while True:
            #Fetch a batch of documents (max 100)
            results = collection.get(limit=100)
            ids_to_delete = results['ids']
            
            #If there are no more IDs to fetch, break the loop
            if not ids_to_delete:
                break
            
            # Delete the fetched batch of documents
            collection.delete(ids=ids_to_delete)
            print(f"  - Deleted batch of {len(ids_to_delete)} documents...")
                        
    vector_store = Chroma(
        client=client,
        collection_name=CHROMA_CLOUD_COLLECTION_NAME,
        embedding_function=embeddings,
    )

    batch_size = 200 
    total_chunks = len(chunkies)
    print(f"Beginning upload of {total_chunks} chunks to Chroma Cloud in batches of {batch_size}...")

    for i in range(0, total_chunks, batch_size):
        batch = chunkies[i:i + batch_size]
        vector_store.add_documents(batch)
        print(f"  - Upserted batch {i//batch_size + 1}/{(total_chunks + batch_size - 1)//batch_size}...")
    
    print(f"Successfully uploaded {total_chunks} chunks to Chroma Cloud.")

if __name__ == "__main__":
    
    # Setup command-line argument parsing
    parser = argparse.ArgumentParser(description="Ingest documents into a vector store.")
    parser.add_argument("--db", choices=["local-chroma", "pinecone", "chroma-cloud"], required=True, help="The vector database to use.")
    args = parser.parse_args()

    # Load the documents
    loaded_docs = load_documents()

    if loaded_docs:
        print("\n--- Document Loading Complete ---")
        #PART1: LOAD DOCUMENTS INTO SUTRUCTURED OBJECTS
        print(f"Total number of pages loaded: {len(loaded_docs)}")

        #TESTING: 
        #Let's inspect the first loaded document aka page to see its structure.
        #LangChain document loaders return structured Python objects, not actual JSON, but they behave like JSON, meaning they're easy to serialize, inspect, and work with.

        print("\n--- Example of a Structured Document Object ---")
        first_doc = loaded_docs[0]
        
        print("\nPage Content (the text to be chunked and embedded):")
        #Printing only the first 300 characters 
        print(first_doc.page_content[:300] + "...")
        
        print("\n--- Metadata (the preserved structure): ---")
        print(first_doc.metadata)

        #PART2: SPLIT TEXT INTO STRUCTURED CHUNKS
        chunked_docs = chunk_documents(loaded_docs)
        #TESTING:
        #Pick a chunk that is around the middle to inspect its structure.
        print("\n--- Example of a Chunked Document Object ---")
        middle_chunk = chunked_docs[len(chunked_docs) // 2]
        
        print(f"\nExample chunk index: {(len(chunked_docs) // 2)}")
        
        print("\n--- Chunk Content (a smaller piece of the original page): ---")
        print(middle_chunk.page_content)

        print("\n--- Metadata (preserved and extended by the splitter): ---")
        print(middle_chunk.metadata)

        print("\n" + "-" * 50)
        #PART 3: EMBED CHUNKS AND STORE IN VECTOR DATABASE

        #Initializing the embedding model
        embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
        
        # Store in the chosen database based on command-line argument
        if args.db == "local-chroma":
            embedStoreChromaLocal(chunked_docs, embeddings)
        elif args.db == "pinecone":
            embedStorePinecone(chunked_docs, embeddings)
        elif args.db == "chroma-cloud":
            embedStoreChromaCloud(chunked_docs, embeddings)

        print("\n--- Ingestion Complete ---")
        
    else:
        print("\n--- No documents were loaded. Please check the 'data/raw/' directory and file types. ---")