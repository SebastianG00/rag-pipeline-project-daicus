from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI 
from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser
from dotenv import load_dotenv
import os

CHROMA_DIR = "vector_db"
load_dotenv()

PROMPT_TEMPLATE = """
Answer the question based only on the following context:

{context}

---

Answer the question based on the above context: {question}
"""


def main():
    """
    Main function to load vector store, create a retreiver and query for relevant documents.
    """

    print("---Loading vector store...---")

    #initialize the embeddings model
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    #load vector store from the CHROMA_DIR directory
    vector_db = Chroma(
        persist_directory=CHROMA_DIR,
        embedding_function=embeddings
    )
    # create a retriever from the vector store
    # top K most relevant docs
    retriever = vector_db.as_retriever(search_kwargs={"k": 3})
    print(f"Vector store loaded with {vector_db._collection.count()} documents.")

    #------TESTING--------
    prompt = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)

    llmModel = ChatOpenAI(
        model="mistralai/Mistral-7B-Instruct-v0.2",
        temperature=0.0,
        max_tokens=1000,
        openai_api_key=os.getenv("OPENROUTER_API_KEY"),  # Use your OpenRouter API key
        openai_api_base="https://openrouter.ai/api/v1",  # OpenRouter endpoint
    )
    
    #Settig up the RAG chain
    """
    It wires together the retriever(from where to get the relevant documents aka from our vector database),
    the prompt template (which formats the question and context),
    question as a passthrough (which basically passes the query into the question field without any modification),
    and the LLM (which generates the answer based on the context).

    The chain can be used to answer questions based on the context retrieved from the vector store.

    """
    rag_chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llmModel
        | StrOutputParser()
    )

    print("---RAG CHAIN READY---")
    
    query = "What are the requirements for a good driver discount?"
    results = rag_chain.invoke(query)
    
    if not results:
        print("No relevant documents found.")
        return

    print(f"Found {len(results)} relevant documents.")

    print(f"Query: {query}")
    print(f"Answer: {results}")


if __name__ == "__main__":
    main()
    print("\n--- Finished ---")

