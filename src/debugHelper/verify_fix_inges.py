# Quick test to see what your ingestion code is ACTUALLY producing
import os
from langchain_community.document_loaders import PyPDFLoader

DATA_DIR = "data/raw/"

# Test with one file to see what happens
test_file = None
for filename in os.listdir(DATA_DIR):
    if filename.endswith('.pdf'):
        test_file = filename
        break

if test_file:
    print(f"Testing with file: {test_file}")
    file_path = os.path.join(DATA_DIR, test_file)
    
    # Load the document
    loader = PyPDFLoader(file_path)
    loaded_docs = loader.load()
    
    if loaded_docs:
        first_doc = loaded_docs[0]
        print(f"\nOriginal source in metadata: {first_doc.metadata['source']}")
        print(f"os.path.basename result: {os.path.basename(first_doc.metadata['source'])}")
        print(f"After .lower(): {os.path.basename(first_doc.metadata['source']).lower()}")
        
        # Now apply your exact ingestion logic
        first_doc.metadata['sourceHelper'] = os.path.basename(first_doc.metadata['source']).lower()
        print(f"Final sourceHelper: {first_doc.metadata['sourceHelper']}")
        
        # Compare with what's actually stored in your vector DB
        print(f"\nBUT your vector DB actually contains: naicPolicy.pdf, stateFarmPolicy.pdf (with capitals)")
        print("This means your vector DB was built BEFORE you added the .lower() call!")

print("\n" + "="*60)
print("SOLUTION: You need to re-ingest your data!")
print("="*60)
print("Run these commands to rebuild with proper lowercase:")
print("python3 src/ingestion.py --db local-chroma")
print("python3 src/ingestion.py --db pinecone") 
print("python3 src/ingestion.py --db chroma-cloud")