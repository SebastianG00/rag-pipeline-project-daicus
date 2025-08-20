import os
from dotenv import load_dotenv
import chromadb

# --- Configuration ---
# Make sure these match your .env file and Chroma Cloud setup
CHROMA_CLOUD_COLLECTION_NAME = "daicus-rag-cloud"

def check_chroma_cloud_connection():
    """
    A simple script to test the connection to Chroma Cloud and add a single document.
    """
    print("--- Testing Chroma Cloud Connection ---")
    
    # 1. Load environment variables
    load_dotenv()
    api_key = os.getenv("CHROMA_API_KEY")
    tenant = os.getenv("CHROMA_TENANT")
    database = os.getenv("CHROMA_DATABASE")

    if not all([api_key, tenant, database]):
        print("Error: Make sure CHROMA_API_KEY, CHROMA_TENANT, and CHROMA_DATABASE are set in your .env file.")
        return

    try:
        # 2. Initialize the Cloud Client
        print("Initializing Chroma Cloud client...")
        client = chromadb.CloudClient(
            tenant=tenant,
            database=database,
            api_key=api_key
        )
        print("Client initialized successfully.")

        # 3. Get or create the collection
        print(f"Getting or creating collection: '{CHROMA_CLOUD_COLLECTION_NAME}'...")
        collection = client.get_or_create_collection(CHROMA_CLOUD_COLLECTION_NAME)
        print("Collection is ready.")

        # 4. Add a single test document
        print("Attempting to add a single document...")
        collection.add(
            documents=["This is a test document."],
            metadatas=[{"source": "test"}],
            ids=["test_id_123"]
        )
        print("Document added successfully!")

        # 5. Verify the document was added
        count = collection.count()
        print(f"Verification: Collection now contains {count} document(s).")
        
        print("\n--- ✅ CONNECTION SUCCESSFUL ---")

    except Exception as e:
        print("\n--- ❌ CONNECTION FAILED ---")
        print(f"An error occurred: {e}")
        print("\nPlease double-check your API Key, Tenant, and Database values in the .env file.")

if __name__ == "__main__":
    check_chroma_cloud_connection()
