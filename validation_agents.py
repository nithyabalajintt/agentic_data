from pydantic import BaseModel, Field
from collections import defaultdict
from langchain_community.document_loaders import PyPDFDirectoryLoader
import re
from dotenv import load_dotenv
from langchain_openai import AzureOpenAIEmbeddings
from openai import AzureOpenAI
import os
import json

load_dotenv()  # Load env variables from .env file

# Paths
DATA_PATH = "data/pdf_files"

# GPT-4 Model configuration
model_name = os.environ.get("AZURE_OPENAI_MODEL_NAME")
endpoint_url = os.environ.get("AZURE_OPENAI_ENDPOINT")
api_key = os.environ.get("AZURE_OPENAI_API_KEY")
deployment_name = os.environ.get("AZURE_OPENAI_DEPLOYMENT")
version_number = os.environ.get("API_VERSION_GA")

# ADA Embeddings configuration
embedding_model_name = os.environ.get("ADA_AZURE_OPENAI_MODEL_NAME")
embedding_endpoint_url = os.environ.get("ADA_AZURE_OPENAI_ENDPOINT")
embedding_api_key = os.environ.get("ADA_AZURE_OPENAI_API_KEY")

# Client setup
client = AzureOpenAI(
    azure_endpoint=endpoint_url,
    azure_deployment=deployment_name,
    api_key=api_key,
    api_version=version_number,
)

embedder = AzureOpenAIEmbeddings(
    model=embedding_model_name,
    api_key=embedding_api_key,
    azure_endpoint=embedding_endpoint_url,
    api_version="2024-10-21",
)

class Name_Validate(BaseModel):
    company_names: list[str] = Field(..., description="List of correct company name, and alternative names and abbreviations")
    

class Docs_Validate(BaseModel):
    response: str  = Field(..., description="Must be Yes if company name is matching with the content, else No")

# Financial keywords
ANNUAL_REPORT_KEYWORDS = [
    "financial statements", "balance sheet", "income statement", "statement of cash flows",
    "statement of changes in equity", "net income", "revenue", "profit", "earnings", "expenses",
    "dividends", "shareholders' equity", "assets", "liabilities", "gross margin", "operating income", "EBITDA",
    "business overview", "company profile", "corporate structure", "organizational structure", "company history",
    "principal activities", "subsidiaries", "risk factors", "internal controls", "management's discussion and analysis",
    "MD&A", "audit report", "independent auditors", "compliance", "regulatory", "board of directors",
    "executive officers", "corporate governance", "management", "compensation", "shareholder information",
    "shareholder meeting", "dividend policy", "stock performance", "major shareholders", "legal proceedings",
    "litigation", "environmental matters", "controls and procedures", "forward-looking statements", "outlook",
    "future plans", "expectations", "guidance", "corporate social responsibility", "sustainability",
    "environmental, social, and governance", "ESG", "certification", "signed", "attestation",
    "section 302 certifications", "table of contents", "letter to shareholders", "financial highlights",
    "selected financial data", "report of independent registered public accounting firm", "notes to financial statements",
    "corporate governance", "risk factors", "legal proceedings", "controls and procedures", "exhibits"
]

# Helper function: Count keyword occurrences
def count_keyword_occurrences(company_names, docs, keywords):
    """
    Count keyword occurrences, including the company name, in the documents.
    """
    # keywords.append(company_names)
    # print(company_names)
    company_names = json.loads(company_names)
    # print(type(company_names))

    print(company_names["company_names"])
    keywords+=company_names["company_names"]
    patterns = {kw: re.compile(r'\b' + re.escape(kw) + r'\b', re.IGNORECASE) for kw in keywords}
    kw_counts = defaultdict(int)
    for doc in docs:
        for kw, pat in patterns.items():
            matches = pat.findall(doc)
            kw_counts[kw] += len(matches)
    print(dict(kw_counts))
    return dict(kw_counts)

# Name Validation Agent
class NameValidationAgent:
    def process_name(self, company_name: str) -> str:
        prompt = [
            {
                "role": "system",
                "content": "You are a validation agent that corrects the name of an enterprise company entered by the user, along with any abbreviations or alternate names for the company. The response must be a list of names, e.g., ['Punjab National Bank', 'PNB', 'PNBIL']."
            },
            {
                "role": "user",
                "content": f"COMPANY NAME:\n{company_name}"
            }
        ]
        response = client.beta.chat.completions.parse(
            messages=prompt,
            model=model_name,
            response_format= Name_Validate
        )
        return response.choices[0].message.content

# Document Validation Agent
class ParentValidationAgent:
    def process_documents(self, documents, company_names: list, keywords) -> str:

        keyword_counts = count_keyword_occurrences(company_names, documents, keywords)
        prompt = [
            {
                "role": "system",
                "content": "You are a validation agent that validates if a document content matches the company name given. Use the keyword occurrence dictionary to decide. If the company names occur multiple times in the dictionary and other terms in the dictionary also appear more than once, respond with 'Yes' which means the content is relevant, else say 'No' which means that the document content is not related to the company name."
            },
            {
                "role": "user",
                "content": f"COMPANY NAME:\n{company_names}\n\nDOCUMENT CONTENT COUNTS:\n{keyword_counts}"
            }
        ]
        response = client.beta.chat.completions.parse(
            messages=prompt,
            model=model_name,
            response_format=Docs_Validate
        )
        return response.choices[0].message.content

# PDF Loading
loader = PyPDFDirectoryLoader(
    path=DATA_PATH,
    glob="**/[!.]*.pdf"
)
docs = loader.load()

documents = [doc.page_content for doc in docs]  # Extract document content
print(documents)

# MAIN FUNCTION EXECUTION
if __name__ == "__main__":
    # Validate company name
    company_name = input("Enter Company Name: ")
    name_agent = NameValidationAgent()
    name_validation_response = name_agent.process_name(company_name)

    # Process parent validation with validated names
    print("Validated Company Names:", name_validation_response)
    parent_agent = ParentValidationAgent()
    parent_validation_response = parent_agent.process_documents(
        documents=documents,
        company_names=name_validation_response,
        keywords=ANNUAL_REPORT_KEYWORDS
    )

    print("Parent Validation Response:", parent_validation_response)