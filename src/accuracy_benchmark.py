import os
import csv
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone import Pinecone as LangChainPinecone
import chromadb
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# Load environment variables
load_dotenv()

# --- Configuration ---
# Constants are shared with ingestion.py to ensure consistency.
# If the embedding model here is different from the one used during ingestion,
# the retrieval will produce nonsensical results.
CHROMA_LOCAL_PATH = "vector_db"
PINECONE_INDEX_NAME = "daicus-rag"
CHROMA_CLOUD_COLLECTION_NAME = "daicus-rag-cloud"
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# --- Ground Truth Dataset ---
# This is the most critical component for accuracy evaluation. It acts as the "answer key".
# Each entry contains a test query and a list of the exact source documents
# that are considered correct answers. This allows us to quantitatively measure performance.

GROUND_TRUTH = [
    # --- Specific Queries Targeting Real Policies ---
    {"query": "What is the maximum payment for a bail bond under the Allstate policy?", "expected_sources": ["data/raw/allstatepolicy.pdf"]},
    {"query": "Under the Allstate policy, what happens if I acquire an additional auto?", "expected_sources": ["data/raw/allstatepolicy.pdf"]},
    {"query": "What is the liability coverage limit for 'Media' loss under the American Family policy?", "expected_sources": ["data/raw/americanfamilypolicy.pdf"]},
    {"query": "Does the American Family policy offer coverage for 'Certified Acts of Terrorism'?", "expected_sources": ["data/raw/americanfamilypolicy.pdf"]},
    {"query": "According to the NAIC guide, what is a C.L.U.E. report?", "expected_sources": ["data/raw/naicpolicy.pdf"]},
    {"query": "What does the NAIC guide say about Guaranteed Asset Protection (GAP) insurance?", "expected_sources": ["data/raw/naicpolicy.pdf"]},
    {"query": "What is the compulsory excess for a driver under 21 in the Progressive policy?", "expected_sources": ["data/raw/progressivepolicy.pdf"]},
    {"query": "Does the Progressive policy cover ferry transit between Sabah and Labuan?", "expected_sources": ["data/raw/progressivepolicy.pdf"]},
    {"query": "Under the State Farm policy, what is the supplementary payment for pet injury coverage?", "expected_sources": ["data/raw/statefarmpolicy.pdf"]},
    {"query": "Does the State Farm policy provide coverage in Mexico?", "expected_sources": ["data/raw/statefarmpolicy.pdf"]},
    {"query": "What is the Five Year Good Driver Discount in the Geico plan?", "expected_sources": ["data/raw/geicodummypolicy.pdf"]},
    {"query": "How many points are assigned for reckless driving under the Geico plan?", "expected_sources": ["data/raw/geicodummypolicy.pdf"]},
    # --- Ambiguous Queries to Test Precision ---
    {"query": "What are the general recommendations for choosing an auto insurance policy?", "expected_sources": ["data/raw/naicpolicy.pdf", "data/raw/distractor3.pdf"]},
    {"query": "Explain the concept of reinsurance in the global market.", "expected_sources": ["data/raw/distractor4.pdf"]},
    # --- Queries Targeting Distractors to Test for False Positives ---
    {"query": "What does the Manual on Uniform Traffic Control Devices say about parking signs?", "expected_sources": ["data/raw/distractor1.pdf"]},
    {"query": "What are the laws regarding driving under the influence?", "expected_sources": ["data/raw/distractor2.pdf"]},
    # --- "No Answer" Queries ---
    {"query": "Does my auto insurance policy cover damages from a meteor strike?", "expected_sources": []},
    {"query": "Is there a discount for owning a red car?", "expected_sources": []}
]


def evaluate_retrieval(db_type: str, embeddings: HuggingFaceEmbeddings, llm: ChatOpenAI, debug: bool = False) -> dict:
    """
    Evaluates the retrieval accuracy of the specified vector store type (local-chroma, pinecone, chroma-cloud).

    This function implements a filter then retrieve strategy:
        1. It uses a router model to determine which specific document the user is asking about.
        2. It retrieves documents from the vector store based on the identified filter.
        3. It calculates precision and recall based on the retrieved documents against the ground truth.

    """
    print(f"\n--- Evaluating Retrieval Accuracy for {db_type.upper()} ---")

    # Define valid filenames to filter against
    # This set acts as a safeguard. We check the LLM router's output against this list
    # to ensure it hasn't "hallucinated" a filename that doesn't exist.
    VALID_FILENAMES = {
        "allstatepolicy.pdf", "americanfamilypolicy.pdf", "naicpolicy.pdf",
        "progressivepolicy.pdf", "statefarmpolicy.pdf", "geicodummypolicy.pdf",
        "distractor1.pdf", "distractor2.pdf", "distractor3.pdf", "distractor4.pdf",
    }

    vector_store = None
    try:
        if db_type == "local-chroma":
            vector_store = Chroma(persist_directory=CHROMA_LOCAL_PATH, embedding_function=embeddings)
        elif db_type == "pinecone":
            vector_store = LangChainPinecone.from_existing_index(PINECONE_INDEX_NAME, embeddings)
        elif db_type == "chroma-cloud":
            client = chromadb.CloudClient(
                tenant=os.getenv("CHROMA_TENANT"), database=os.getenv("CHROMA_DATABASE"),
                api_key=os.getenv("CHROMA_API_KEY")
            )
            vector_store = Chroma(client=client, collection_name=CHROMA_CLOUD_COLLECTION_NAME, embedding_function=embeddings)
    except Exception as e:
        print(f"ERROR: Failed to initialize vector store for {db_type.upper()}: {e}")
        return {"precision": 0, "recall": 0}

    # --- ROUTER CHAIN ---
    # This is a small, specialized LLM helper query whose only job is to identify the target filename from the query.
    # By giving the LLM context on how to parse the list of valid filenames in the prompt, we constrain its output, making it more reliable.
    # If it can't identify a specific file, it returns 'none'.
    router_prompt_template = """Based on the user's query, identify the exact document filename they are asking about.
The available filenames are: allstatepolicy.pdf, americanfamilypolicy.pdf, naicpolicy.pdf, progressivepolicy.pdf, statefarmpolicy.pdf, geicodummypolicy.pdf, distractor1.pdf, distractor2.pdf, distractor3.pdf, distractor4.pdf
If the query mentions one of these policies or documents, return only the corresponding exact filename in all lowercase.
If the query does not mention a specific document, return the string 'none'.

Query: {query}
Filename:"""
    router_prompt = ChatPromptTemplate.from_template(router_prompt_template)
    router_chain = router_prompt | llm | StrOutputParser()

    total_precision, total_recall, results_data = 0, 0, []

    for item in GROUND_TRUTH:
        query = item["query"]
        expected_sources = set(os.path.basename(s).lower() for s in item["expected_sources"])

        try:
            # Step 1: Get the target filename from the router.
            target_filename = router_chain.invoke({"query": query}).strip().lower()

            retriever = None
            # Step 2: If the router found a valid filename, create a filtered retriever.
            if target_filename in VALID_FILENAMES:
                if debug: print(f"  - Router identified filter: '{target_filename}'")
                
                if db_type == "pinecone":
                    retriever = vector_store.as_retriever(search_kwargs={'filter': {'sourceHelper': {'$eq': target_filename}}})
                else:
                    retriever = vector_store.as_retriever(search_kwargs={'filter': {'sourceHelper': target_filename}})
            else:
                if debug: print("  - Router found no specific source, using general search.")
                retriever = vector_store.as_retriever()
            # Step 3: Perform the retrieval.
            retrieved_docs = retriever.invoke(query)

        except Exception as e:
            print(f"  - ERROR retrieving for query '{query[:50]}...': {e}. Defaulting to empty result.")
            retrieved_docs = []

        # Extract the 'sourceHelper' from the metadata of the retrieved documents.
        retrieved_sources = {doc.metadata.get("sourceHelper") for doc in retrieved_docs if doc.metadata.get("sourceHelper")}
        
        # --- SCORING LOGIC ---
        # Precision: Among all the documents we retrieved, how many were correct? (Measures noise)
        # Recall: Among the documents we SHOULD HAVE retrieved, how many did we find? (Measures misses)
        true_positives = len(retrieved_sources.intersection(expected_sources))
        
        # Logic for "no answer" questions.
        if not expected_sources:
            precision = 1.0 if not retrieved_sources else 0.0
            recall = 1.0 if not retrieved_sources else 0.0
        else:
            precision = true_positives / len(retrieved_sources) if retrieved_sources else 1.0 if not expected_sources else 0.0
            recall = true_positives / len(expected_sources) if expected_sources else 1.0 if not retrieved_sources else 0.0
        
        total_precision += precision
        total_recall += recall
        
        results_data.append({
            "query": query, "expected_sources": ", ".join(sorted(list(expected_sources))),
            "retrieved_sources": ", ".join(sorted(list(retrieved_sources))),
            "precision": precision, "recall": recall
        })

    avg_precision = total_precision / len(GROUND_TRUTH) if GROUND_TRUTH else 0
    avg_recall = total_recall / len(GROUND_TRUTH) if GROUND_TRUTH else 0
    
    # Saving detailed results to a CSV is a best practice. It allows for in-depth analysis
    # of which specific queries are failing, which is crucial for targeted improvements.
    csv_filename = f"accuracy_results_{db_type}.csv"
    with open(csv_filename, "w", newline="", encoding="utf-8") as csvfile:
        fieldnames = ["query", "expected_sources", "retrieved_sources", "precision", "recall"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results_data)
    print(f"Saved detailed results to {csv_filename}")

    return {"precision": avg_precision, "recall": avg_recall}

def main():
    """Main function to run the full accuracy benchmark."""
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
    
    llm = ChatOpenAI(
        model_name="openai/gpt-3.5-turbo", 
        temperature=0,
        base_url="https://openrouter.ai/api/v1", 
        api_key=os.getenv("OPENROUTER_API_KEY")
    )

    local_chroma_scores = evaluate_retrieval("local-chroma", embeddings, llm, debug=False)
    pinecone_scores = evaluate_retrieval("pinecone", embeddings, llm, debug=False)
    cloud_chroma_scores = evaluate_retrieval("chroma-cloud", embeddings, llm, debug=False)
    
    print("\n" + "="*50)
    print("        RETRIEVAL ACCURACY REPORT")
    print("="*50)
    print("\n--- Average Precision@K ---")
    print(f"Local Chroma:   {local_chroma_scores['precision']:.2f}")
    print(f"Pinecone:       {pinecone_scores['precision']:.2f}")
    print(f"Chroma Cloud:   {cloud_chroma_scores['precision']:.2f}")
    
    print("\n--- Average Recall@K ---")
    print(f"Local Chroma:   {local_chroma_scores['recall']:.2f}")
    print(f"Pinecone:       {pinecone_scores['recall']:.2f}")
    print(f"Chroma Cloud:   {cloud_chroma_scores['recall']:.2f}")
    print("\n" + "="*50)

if __name__ == "__main__":
    main()