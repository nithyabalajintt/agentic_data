import camelot
from IPython.display import display
import os
from agno.agent import Agent, RunResponse
from agno.models.azure import AzureOpenAI as AOI
from langchain.vectorstores.chroma import Chroma
from langchain_openai import AzureOpenAIEmbeddings
from langchain.docstore.document import Document  # <-- added import
import shutil
from dotenv import load_dotenv
import warnings

# Suppress warnings
warnings.filterwarnings("ignore")

# Configure Chroma path for vector store
CHROMA_PATH = "chroma_finance_docs"

# Load environment variables from a .env file
load_dotenv()

#######################################
### Load Azure OpenAI Configurations ###
#######################################
# GPT-4o
model_name = os.environ.get("AZURE_OPENAI_MODEL_NAME")
endpoint_url = os.environ.get("AZURE_OPENAI_ENDPOINT")
api_key = os.environ.get("AZURE_OPENAI_API_KEY")
deployment_name = os.environ.get("AZURE_OPENAI_DEPLOYMENT")
version_number = os.environ.get("API_VERSION_GA")

# ADA
embedding_model_name = os.environ.get("ADA_AZURE_OPENAI_MODEL_NAME")
embedding_endpoint_url = os.environ.get("ADA_AZURE_OPENAI_ENDPOINT")
embedding_api_key = os.environ.get("ADA_AZURE_OPENAI_API_KEY")
embedding_deployment_name = os.environ.get("ADA_AZURE_OPENAI_DEPLOYMENT")

#########################################
### Azure OpenAI and LangChain Setup ###
#########################################
# Embeddings for Chroma
embedder = AzureOpenAIEmbeddings(
    model=embedding_model_name,
    api_key=embedding_api_key,
    azure_endpoint=embedding_endpoint_url,
    api_version="2024-10-21"
)

# Agent setup for summarization
agent = Agent(
    model=AOI(
        id=model_name,
        api_key=api_key,
        azure_endpoint=endpoint_url,
        azure_deployment=deployment_name
    ),
    description="You are a finance analyst that researches income, balance sheet, and cash flow statements, and annual reports of companies. Generate concise expert summaries of the provided tables, including all key details and numerics.",
    instructions=["Summarize the table/dataframe into a single paragraph without missing any important details and include all the numericals."],
    show_tool_calls=True,
    markdown=True,
    debug_mode=True
)

############################################
### Function to Process PDF and Summarize ###
############################################
def process_pdfs_in_directory(directory_path):
    """
    Processes all PDF files in a directory, extracts tables, summarizes them, and stores them in a Chroma database.

    Args:
      directory_path (str): Path to the directory containing PDF files.

    Returns:
      None
    """
    all_summaries = []  # Collect all summaries from all PDFs

    # Loop through all PDF files in the directory
    for filename in os.listdir(directory_path):
        if filename.endswith(".pdf"):  # Only process .pdf files
            pdf_file_path = os.path.join(directory_path, filename)
            print(f"\nProcessing PDF: {filename}")

            # Step 1: Extract tables using Camelot
            tables = camelot.read_pdf(pdf_file_path, pages="all", flavor="stream")
            filtered_tables = []

            for i, table in enumerate(tables):
                num_columns = table.df.shape[1]
                if num_columns > 1:  # Only consider tables with multiple columns
                    print(f"\nTable {i + 1} from {filename} (Columns: {num_columns}):")
                    display(table.df)  # Use Jupyter display for pretty printing
                    filtered_tables.append(table.df)

            # Step 2: Generate summaries from tables 
            for table_df in filtered_tables:
                finance_narrative_response: RunResponse = agent.run(table_df.to_string())
                print(f"\nSummary generated for Table {i + 1} in {filename}:\n{finance_narrative_response.content}")
                all_summaries.append(finance_narrative_response.content)  # Collect the summary

    # Save all summaries to Chroma
    save_to_chroma(all_summaries)

def save_to_chroma(summaries):
    """
    Save the given list of summaries (strings) to a Chroma database.

    Args:
      summaries: List of summaries (string format).
    """
    # Convert summaries into LangChain Document objects
    documents = [Document(page_content=summary, metadata={}) for summary in summaries]

    # Clear out the existing database directory, if it exists
    if os.path.exists(CHROMA_PATH):
        shutil.rmtree(CHROMA_PATH)

    # Create a new Chroma database from summaries using OpenAI embeddings
    db = Chroma.from_documents(
        documents,
        embedder,
        persist_directory=CHROMA_PATH
    )

    # Persist the database to disk
    db.persist()
    print(f"\nSaved {len(documents)} summaries to Chroma at {CHROMA_PATH}.")

##################################
### Execution Begins Here ###
##################################
if __name__ == "__main__":
    # Path to directory containing PDF files
    pdf_directory_path = "data/pdf_files"  # Change to your directory

    # Process PDFs and summarize
    process_pdfs_in_directory(pdf_directory_path)