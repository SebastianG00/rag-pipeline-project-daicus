# RAG Pipeline

## Prerequisites

Before running this project, make sure you have Python 3 installed on your system.

## Setup

#### a) Create Virtual Environment

First, create and activate a Python virtual environment:

```bash
python3 -m venv venv
source venv/bin/activate
```
#### b) Install Dependencies

Install the required dependencies using pip:

```bash
pip install -r requirements.txt
```
### c) Development Notes

##### Updating Dependencies

If you make changes to the project dependencies, you **MUST** update the `requirements.txt` file by running:

```bash
pip freeze > requirements.txt
```
***

### 1. Project Overview
This project implements a Retrieval-Augmented Generation (RAG) pipeline designed to answer questions based on a collection of documents. In this case, I arbitrarily decided to use insurance policy documents. Think of it as a smart digital librarian: it reads and understands a library of documents (ingestion.py), and then helps a user find precise answers within them (app.py).

The system is designed to be database-agnostic, supporting local (Chroma), serverless (Pinecone), and cloud-based (Chroma Cloud) vector stores. We also have robust scripts to evaluate both the accuracy and performance of the pipeline.

***

### 2. File Breakdown
The project is split into four main Python scripts, each with a specific job.

📜 ingestion.py - The Librarian
    This is the starting point for all data. Its job is to read raw documents, process them, and store them intelligently in a vector database.

    Purpose: To build and populate the knowledge base (vector store).

    Workflow:

        Load: Scans the data/raw/ directory for PDF and HTML files.

        Standardize Metadata: It crucially takes each document's filename, converts it to lowercase (e.g., AllStatePolicy.PDF becomes allstatepolicy.pdf), and stores this clean name in the document's metadata under the key sourceHelper. This prevents case-sensitivity errors during retrieval.

        Chunk: Breaks the loaded documents into smaller, overlapping text chunks. This is essential for the retriever to find specific, relevant passages.

    How to Run:
        python3 src/ingestion.py --db local-chroma
        python3 src/ingestion.py --db pinecone
        python3 src/ingestion.py --db chroma-cloud

💬 app.py - The Search Desk
    This is a simple, interactive application to ask questions to your RAG pipeline. It's designed for quick, qualitative testing.

    Purpose: To provide a simple way to chat with your documents and see the RAG pipeline in action.

    Key Components:
        It connects only to the local Chroma database stored in the vector_db directory.

        It uses a ChatPromptTemplate to structure the context and question for the LLM.

        It builds a "RAG chain" that automatically retrieves relevant documents, adds them to the prompt, and sends them to the LLM for an answer.
    
    How to Run:
        python3 src/app.py

✅ accuracy_benchmark.py - The Fact-Checker
    This script measures the quality and correctness of your RAG pipeline's retrieval step. It answers the question: "Is our retriever finding the right documents?"

    Purpose: To quantitatively measure the Precision and Recall of the retriever against a "ground truth" dataset.

    Key Components:
        GROUND_TRUTH Dataset: A predefined list of questions and the exact source documents that are expected to contain the answers.

        Router Chain: A small LLM chain that first tries to identify which specific document a user's query is about.

        Filtering Logic: It uses the router's output to apply a metadata filter, forcing the retriever to search only within the identified document.

        Scoring: It compares the documents the retriever found against the "expected" documents in the ground truth to calculate scores.
    
    How to Run:
        python3 src/accuracy_benchmark.py

⏱️ benchmark.py - The Time-Trial Runner
    This script measures the performance and speed of the pipeline. It answers the question: "How fast is our system?"

    Purpose: To measure ingestion time and query latency across all supported databases.

    Key Components:

        run_ingestion: A function that calls ingestion.py as a subprocess and times how long it takes to complete.

        run_rag_pipeline_benchmark: A function that connects to a database and times how long it takes to answer a few test queries.

        Reporting: It prints a final, clean report comparing the speed of each database.
    
    How to Run:
        python3 src/benchmark.py

***


### 3. How to Use & Make Improvements

Standard Workflow
    Setup: Create a .env file with your API keys (PINECONE_API_KEY, CHROMA_API_KEY, OPENROUTER_API_KEY, etc.).

    Add Data: Place your PDF/HTML documents in the data/raw/ folder. Best Practice: Use all-lowercase filenames.

    Ingest: Run ingestion.py to populate your chosen vector database.

    Test: Use app.py for quick manual tests.

    Evaluate: Run accuracy_benchmark.py and benchmark.py to get quantitative scores on the pipeline's quality and speed.

Areas for Experimentation: To improve the accuracy scores (Precision and Recall), interns should focus on these key areas:

    Chunking Strategy (ingestion.py): The chunk_size and chunk_overlap in the chunk_documents function are the most important parameters to tune. Experiment with different values to see how they impact the scores.

    Embedding Model (ingestion.py & accuracy_benchmark.py): The EMBEDDING_MODEL_NAME can be changed to a more powerful model from the Hugging Face Hub. This requires re-ingesting all data.

    Add a Reranker (accuracy_benchmark.py): A powerful technique is to retrieve more documents initially (e.g., k=10) and then use a "reranker" model to pick the best 3-5 before sending them to the LLM.

***


### 4. Project Resources
For more context on this project's development, goals, and future direction, please see the following resources:

* **[Presentation on Next Steps & Future Improvements]([https://www.your-presentation-link-here.com](https://sebastiang00.github.io/rag-presentation/))** - First Attempt accuracy results (old version)
* **[Project Brief & Technical Document (Google Doc)]([https://docs.google.com/document/d/your-doc-id-here/edit](https://docs.google.com/document/d/1JYfVTEHsrkdxZ9z_a2A9AuOFbfPWZ4turDQvX0wsJD0/edit?usp=sharing))** - A detailed document covering the project's initial scope, challenges, and key learnings. As well as outlining the strategic path to improve the RAG pipeline's accuracy to over 90%.
