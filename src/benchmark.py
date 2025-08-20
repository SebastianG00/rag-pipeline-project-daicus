import os
import time
import subprocess
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone import Pinecone as LangChainPinecone
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser
import chromadb

load_dotenv()

CHROMA_LOCAL_PATH = "vector_db"
PINECONE_INDEX_NAME = "daicus-rag"
CHROMA_CLOUD_COLLECTION_NAME = "daicus-rag-cloud"
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

TEST_QUERIES = [
    "What are the requirements for a good driver discount?",
    "How is the policy affected by a new vehicle purchase?",
    "What is the process for filing a claim after an accident?",
]

def run_ingestion(db_type: str) -> float:
    print(f"\n--- Running Ingestion for {db_type.upper()} ---")
    start_time = time.time()
    try:
        subprocess.run(
            ["python3", "src/ingestion.py", "--db", db_type],
            check=True, capture_output=True, text=True, timeout=1200
        )
    except subprocess.TimeoutExpired:
        print(f"ERROR: Ingestion for {db_type.upper()} timed out.")
        return float('inf')
    except subprocess.CalledProcessError as e:
        print(f"ERROR: Ingestion script for {db_type.upper()} failed.")
        print("STDERR:", e.stderr)
        return float('inf')
    duration = time.time() - start_time
    print(f"Ingestion for {db_type.upper()} completed in {duration:.2f} seconds.")
    return duration

def run_rag_pipeline_benchmark(db_type: str, embeddings: HuggingFaceEmbeddings, llm: ChatOpenAI) -> float:
    print(f"\n--- Running Full RAG Benchmark for {db_type.upper()} ---")
    vector_store = None
    try:
        if db_type == "local-chroma":
            vector_store = Chroma(persist_directory=CHROMA_LOCAL_PATH, embedding_function=embeddings)
        elif db_type == "pinecone":
            vector_store = LangChainPinecone.from_existing_index(PINECONE_INDEX_NAME, embeddings)
        elif db_type == "chroma-cloud":
            # Use the CloudClient instance to connect
            client = chromadb.CloudClient(
                tenant=os.getenv("CHROMA_TENANT"),
                database=os.getenv("CHROMA_DATABASE"),
                api_key=os.getenv("CHROMA_API_KEY")
            )
            vector_store = Chroma(
                client=client, collection_name=CHROMA_CLOUD_COLLECTION_NAME,
                embedding_function=embeddings
            )
        else:
            raise ValueError("Invalid database type specified.")
    except Exception as e:
        print(f"ERROR: Failed to initialize retriever for {db_type.upper()}: {e}")
        return float('inf')
    
    retriever = vector_store.as_retriever(search_kwargs={"k": 3})
    prompt = ChatPromptTemplate.from_template(
        "Answer the question based only on the following context:\n\n{context}\n\n---\n\nAnswer the question: {question}"
    )
    rag_chain = (
        {"context": retriever, "question": RunnablePassthrough()} | prompt | llm | StrOutputParser()
    )
    total_time = 0
    for query in TEST_QUERIES:
        start_time = time.time()
        rag_chain.invoke(query)
        end_time = time.time()
        total_time += (end_time - start_time)
    average_latency = total_time / len(TEST_QUERIES)
    print(f"Average RAG pipeline latency for {db_type.upper()}: {average_latency:.4f} seconds.")
    return average_latency

def main():
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
    llm = ChatOpenAI(
        model_name="mistralai/mistral-7b-instruct", temperature=0,
        base_url="https://openrouter.ai/api/v1", api_key=os.getenv("OPENROUTER_API_KEY")
    )
    
    local_chroma_ingestion = run_ingestion("local-chroma")
    pinecone_ingestion = run_ingestion("pinecone")
    cloud_chroma_ingestion = run_ingestion("chroma-cloud")
    
    local_chroma_latency = run_rag_pipeline_benchmark("local-chroma", embeddings, llm)
    pinecone_latency = run_rag_pipeline_benchmark("pinecone", embeddings, llm)
    cloud_chroma_latency = run_rag_pipeline_benchmark("chroma-cloud", embeddings, llm)
    
    print("\n" + "="*50)
    print("           BENCHMARKING REPORT")
    print("="*50)
    print("\n--- INGESTION TIME ---")
    print(f"Local Chroma:   {local_chroma_ingestion:.2f} seconds")
    print(f"Pinecone:       {pinecone_ingestion:.2f} seconds")
    print(f"Chroma Cloud:   {cloud_chroma_ingestion:.2f} seconds")
    print("\n--- AVERAGE END-TO-END RAG LATENCY ---")
    print(f"Local Chroma:   {local_chroma_latency:.4f} seconds")
    print(f"Pinecone:       {pinecone_latency:.4f} seconds")
    print(f"Chroma Cloud:   {cloud_chroma_latency:.4f} seconds")
    print("\n" + "="*50)

if __name__ == "__main__":
    main()