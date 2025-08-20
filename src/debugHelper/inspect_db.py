# inspect_db.py
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

CHROMA_LOCAL_PATH = "vector_db"
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

print("--- Inspecting Local Chroma DB ---")

# Initialize components
embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
vector_store = Chroma(persist_directory=CHROMA_LOCAL_PATH, embedding_function=embeddings)

# Perform a simple search to retrieve a document
# We'll search for a common term we know is in the documents
results = vector_store.similarity_search("insurance policy", k=1)

if results:
    # Get the first document that was found
    first_doc = results[0]
    
    print("\nFound a document. Here is its metadata:")
    print("-" * 30)
    print(first_doc.metadata)
    print("-" * 30)
    
    # Check for the specific field we need
    if "sourceHelper" in first_doc.metadata:
        print("✅ The 'sourceHelper' field exists.")
        print(f"   Value: '{first_doc.metadata['sourceHelper']}'")
    else:
        print("❌ CRITICAL ERROR: The 'sourceHelper' field does NOT exist in the metadata!")

else:
    print("Could not retrieve any documents from the database.")