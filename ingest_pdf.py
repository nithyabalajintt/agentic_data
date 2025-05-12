import os
import pdfplumber
from langchain_core.documents import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from openai import AzureOpenAI
from chromadb import PersistentClient
from chromadb.config import Settings
from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction
from dotenv import load_dotenv
 
# Load environment variables
load_dotenv()
 
# GPT config
model_name = os.environ.get("AZURE_OPENAI_MODEL_NAME")
endpoint_url = os.environ.get("AZURE_OPENAI_ENDPOINT")
api_key = os.environ.get("AZURE_OPENAI_API_KEY")
deployment_name = os.environ.get("AZURE_OPENAI_DEPLOYMENT")
version_number = os.environ.get("API_VERSION_GA")
 
embedding_model_name = os.environ.get("ADA_AZURE_OPENAI_MODEL_NAME")
embedding_endpoint_url = os.environ.get("ADA_AZURE_OPENAI_ENDPOINT")
embedding_api_key = os.environ.get("ADA_AZURE_OPENAI_API_KEY")
embedding_deployment_name = os.environ.get("ADA_AZURE_OPENAI_DEPLOYMENT")

client = AzureOpenAI(
    api_key=api_key,
    api_version=version_number,
    azure_endpoint=endpoint_url
)
 
# PDF reading and processing
def extract_text_and_tables(pdf_path):
    documents = []
    with pdfplumber.open(pdf_path) as pdf:
        for i, page in enumerate(pdf.pages):
            text = page.extract_text() or ""
            tables = page.extract_tables()
            doc_text = f"Page {i+1}:\n{text.strip()}" if text else f"Page {i+1}: No text"
            
            documents.append(Document(
                page_content=doc_text,
                metadata={"source": pdf_path, "page": i + 1}
            ))
            
            for table in tables:
                table_text = convert_table_to_text(table)
                documents.append(Document(
                    page_content=f"Table on page {i+1}:\n{table_text}",
                    metadata={"source": pdf_path, "page": i + 1}
                ))
    return documents
 
def convert_table_to_text(table):
    prompt = f"Convert this table into a readable paragraph:\n{table}"
    try:
        response = client.chat.completions.create(
            model=deployment_name,
            messages=[
                {"role": "system", "content": "You summarize tables into a readable financial description."},
                {"role": "user", "content": prompt}
            ]
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"LLM summarization failed: {e}")
        return str(table)
 
# Split and embed documents
def chunk_documents(documents):
    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
    return splitter.split_documents(documents)
 
def save_to_chroma(docs, persist_directory="chroma_finance_docs"):
    print("Starting Chroma embedding...")
 
    embedding_function = OpenAIEmbeddingFunction(
        api_key=os.environ.get("ADA_AZURE_OPENAI_API_KEY"),
        azure_endpoint=os.environ.get("ADA_AZURE_OPENAI_ENDPOINT"),
        deployment_name=os.environ.get("ADA_AZURE_OPENAI_DEPLOYMENT"),
        api_version=version_number,
        model_name="text-embedding-ada-002"  # Ensure this is specified
    )
 
    db = PersistentClient(path=persist_directory, settings=Settings(allow_reset=True))
    collection = db.get_or_create_collection(name="finance_docs", embedding_function=embedding_function)
 
    for i, doc in enumerate(docs):
        if doc.page_content.strip():
            # Filter out LangChain-internal keys like `_type` to prevent Chroma errors
            cleaned_metadata = {k: v for k, v in doc.metadata.items() if not k.startswith("_")}
            
            collection.add(
                documents=[doc.page_content],
                metadatas=[cleaned_metadata],
                ids=[f"doc_{i}"]
            )
    print(f"Saved {len(docs)} chunks to ChromaDB at: {persist_directory}")
 
def generate_data_store(pdf_folder="data/pdf_files"):
    all_documents = []
    for filename in os.listdir(pdf_folder):
        if filename.endswith(".pdf"):
            print(f"Processing {filename}")
            path = os.path.join(pdf_folder, filename)
            docs = extract_text_and_tables(path)
            all_documents.extend(docs)
 
    print(f"Loaded {len(all_documents)} raw docs from PDFs.")
    chunks = chunk_documents(all_documents)
    print(f"Split into {len(chunks)} chunks.")
    save_to_chroma(chunks)
 
if __name__ == "__main__":
    generate_data_store()
 