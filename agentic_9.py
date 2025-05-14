# Import agnodata Agent and sample tools.
from agno.agent import Agent
# from agno.tools.duckduckgo import DuckDuckGoTools
from agno.tools.yfinance import YFinanceTools 
from agno.agent import Agent, RunResponse
# from agno.knowledge.pdf_url import PDFUrlKnowledgeBase
from agno.knowledge.text import TextKnowledgeBase
from validation_agents import NameValidationAgent, ParentValidationAgent, ANNUAL_REPORT_KEYWORDS
from agno.vectordb.chroma import ChromaDb
from agno.embedder.azure_openai import AzureOpenAIEmbedder
from agno.models.azure import AzureOpenAI as AOI
from validation_agents import ParentValidationAgent, NameValidationAgent
from agno.document.chunking.recursive import RecursiveChunking
from pydantic import BaseModel, Field
import requests
from rule_decision import rule_function  # You must define this function
import os
from datetime import datetime
import joblib
import shutil
import sklearn
import numpy as np
from dotenv import load_dotenv
from agno.utils.pprint import pprint_run_response
from openai import AzureOpenAI
from temp_table import process_pdfs_in_directory
from pathlib import Path
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import pickle
import json
from langchain_community.vectorstores import Chroma
from langchain_openai import AzureOpenAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from agno.knowledge.langchain import LangChainKnowledgeBase
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFDirectoryLoader
import warnings
warnings.filterwarnings("ignore")

load_dotenv()  # Load env variables from .env file

#GPT-4o
model_name = os.environ.get("AZURE_OPENAI_MODEL_NAME")
endpoint_url = os.environ.get("AZURE_OPENAI_ENDPOINT")
api_key = os.environ.get("AZURE_OPENAI_API_KEY")
deployment_name = os.environ.get("AZURE_OPENAI_DEPLOYMENT")
version_number = os.environ.get("API_VERSION_GA")

#ADA
embedding_model_name = os.environ.get("ADA_AZURE_OPENAI_MODEL_NAME")
embedding_endpoint_url = os.environ.get("ADA_AZURE_OPENAI_ENDPOINT")
embedding_api_key = os.environ.get("ADA_AZURE_OPENAI_API_KEY")
embedding_deployment_name = os.environ.get("ADA_AZURE_OPENAI_DEPLOYMENT")
# version_number = os.environ.get("API_VERSION_GA")

# Azure OpenAI
client = AzureOpenAI(
        azure_endpoint=endpoint_url,
        azure_deployment=deployment_name,
        api_key=api_key,
        api_version=version_number,
    )

# Initialize embeddings (Azure OpenAI)
embeddings = AzureOpenAIEmbeddings(
  api_key = embedding_api_key,  
  api_version = "2024-10-21",
  azure_endpoint =embedding_endpoint_url
)
# Initialize an empty knowledge base globally

# Function to set up the knowledge base
# Function to set up the knowledge base
# Declare the global knowledge_base variable with an initial value of None
knowledge_base = None
def setup_knowledge_base():
    """
    Set up the global knowledge base initially with None retriever.
    """
    global knowledge_base
    if knowledge_base is None:  # Initialize the knowledge base only once
        knowledge_base = LangChainKnowledgeBase(retriever=None)  # No data loaded yet
        print("Initialized empty knowledge base.")
setup_knowledge_base()
# Function to dynamically add documents to Chroma

def validate_and_process_file(file_path, company_name):
    """
    Validate the uploaded file against the company name before processing.
    Returns a tuple of (success: bool, message: str)
    """
    try:
        # Load the PDF file
        loader = PyPDFDirectoryLoader(
            path=os.path.dirname(file_path),
            glob=os.path.basename(file_path),
            silent_errors=False
        )
        docs = loader.load()
        
        if not docs:
            return False, "No content could be extracted from the PDF."
            
        # Extract document content
        documents = [doc.page_content for doc in docs]
        
        # Validate company name
        name_agent = NameValidationAgent()
        name_validation_response = name_agent.process_name(company_name)
        print(f"Validated Company Names: {name_validation_response}")
        
        # Process parent validation with validated names
        parent_agent = ParentValidationAgent()
        parent_validation_response = parent_agent.process_documents(
            documents=documents,
            company_names=name_validation_response,
            keywords=ANNUAL_REPORT_KEYWORDS
        )
        print(f"Parent Validation Response: {parent_validation_response}")
        
        # Check validation response
        validation_obj = json.loads(parent_validation_response)
        if validation_obj["response"] == "Yes":
            # File is valid, proceed with embedding
            add_file_result = add_file_to_chroma(file_path)
            print("Processing tables from PDFs....")
            pdf_directory_path = "data/pdf_files"
            process_pdfs_in_directory(pdf_directory_path)
            return True, name_validation_response
        else:
            # File is not valid for the company
            return False, "The document does not appear to be related to the specified company."
            
    except Exception as e:
        print(f"Error in validation process: {str(e)}")
        return False, f"Validation failed: {str(e)}"

def add_file_to_chroma(file_path):
    """
    Process the uploaded PDF file, extract embeddings, and update the retriever in the knowledge base.
    """
    try:
        global knowledge_base
       
        # Load the PDF file 
        loader = PyPDFDirectoryLoader(
                path = "data\pdf_files",
                glob = "**/[!.]*.pdf",
                silent_errors = False,
                load_hidden = False,
                recursive = False,
                extract_images = False
        )
        docs = loader.load()
 
        # Split text into chunks for embedding
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        docs = text_splitter.split_documents(docs)
 
        # Initialize Chroma vector store and add documents
        db = Chroma(embedding_function=embeddings, persist_directory="chroma_db")
        db.add_documents(docs)
 
        # Create a retriever from the Chroma database
        retriever = db.as_retriever()
 
        # Update the global knowledge base with the retriever
        knowledge_base.retriever = retriever  # Associate the updated retriever
       
        print(f"Knowledge base updated with {len(docs)} documents.")  # Debugging log
        return {"message": "Document processed and added to embeddings successfully!"}
 
    except Exception as e:
        print(f"Error in embedding process: {str(e)}")
        return {"error": f"Embedding failed: {str(e)}"}
 
# loader = PyPDFDirectoryLoader(
#     path = "data\pdf_files",
#     glob = "**/[!.]*.pdf",
#     silent_errors = False,
#     load_hidden = False,
#     recursive = False,
#     extract_images = False,
#     # password = None,
#     # mode = "page",
#     # images_to_text = None,
#     # headers = None,
#     # extraction_mode = "plain",
#     # extraction_kwargs = None,
# )
# docs = loader.load()

# # Split text into chunks
# text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
# docs = text_splitter.split_documents(docs)

# embeddings = AzureOpenAIEmbeddings(
#   api_key = embedding_api_key,  
#   api_version = "2024-10-21",
#   azure_endpoint =embedding_endpoint_url
# )

# # Create Chroma vector store
# db = Chroma(embedding_function=embeddings, persist_directory="./chroma_db")
# db.add_documents(docs)

# # Set up the retriever
# retriever = db.as_retriever()

# Create a knowledge base
# knowledge_base = LangChainKnowledgeBase(retriever=retriever)

# embeddings = client.embeddings.create(
#     input = "Your text string goes here",
#     model= embedding_model_name
# )

# knowledge_base = PDFKnowledgeBase(
#     path="/content/RIL_IAR_2024.pdf",
#     vector_db=ChromaDb(collection="finances", embedder = embeddings),
# )
# knowledge_base.load(recreate=True)

# embeddings = AzureOpenAIEmbedder(id=embedding_model_name, api_key = embedding_api_key, azure_deployment= embedding_deployment_name, azure_endpoint=embedding_endpoint_url, api_version="2024-10-21")

# knowledge_base = TextKnowledgeBase(
#     path="data/txt_files",
#     vector_db=ChromaDb(collection="finance_docs", embedder = embeddings),
#     chunking_strategy=RecursiveChunking(chunk_size = 1000, overlap = 100),
# )
# knowledge_base.load(recreate=False)

# embeddings = AzureOpenAIEmbeddings(
#   api_key = embedding_api_key,  
#   api_version = "2024-10-21",
#   azure_endpoint =embedding_endpoint_url
# )

# # Create Chroma vector store
# db = Chroma(embedding_function=embeddings, persist_directory="./chroma_db")
# db.add_documents(docs)

# # Set up the retriever
# retriever = db.as_retriever()

# # Create a knowledge base
# knowledge_base = LangChainKnowledgeBase(retriever=retriever)

# embeddings = client.embeddings.create(
#     input = "Your text string goes here",
#     model= embedding_model_name
# )

# knowledge_base = PDFKnowledgeBase(
#     path="/content/RIL_IAR_2024.pdf",
#     vector_db=ChromaDb(collection="finances", embedder = embeddings),
# )
# knowledge_base.load(recreate=False)

# embeddings = AzureOpenAIEmbedder(id=embedding_model_name, api_key = embedding_api_key, azure_deployment= embedding_deployment_name, azure_endpoint=embedding_endpoint_url, api_version="2024-10-21")

# knowledge_base = TextKnowledgeBase(
#     path="data/txt_files",
#     vector_db=ChromaDb(collection="finance_docs", embedder = embeddings),
#     chunking_strategy=RecursiveChunking(chunk_size = 1000, overlap = 100),
# # )
# knowledge_base.load(recreate=False)

# ---------------------------------------------------------------------
# Finance Agent: Uses multiple tools to generate a risk financial narrative.
# ---------------------------------------------------------------------
risk_analysis_finance_agent = Agent(
    model=AOI(id=model_name,
    api_key=api_key,
    azure_endpoint=endpoint_url,
    azure_deployment=deployment_name
    ),
    description="You are an agent that drafts a financial analysis draft for a corporate loan for a company. The context provided to you to will help you understand the company's financial figures, sentiment, news and stock market data if any",
    tools=[YFinanceTools(key_financial_ratios = True, stock_fundamentals = True, company_news=True)], #add ml tool also
    # run_id=run_id,
    # user_id=user,
    # knowledge=knowledge_base,
    instructions="""Use the context to generate a financial analysis draft of the company.
Provide a well-structured credit narrative that integrates all relevant information from the context.
Include sections under the following headings:
1. Financial and Stock Information
2. Company News & Sentiment

""",
    # add_context_instructions = ""
    # use_tools=True,
    show_tool_calls=True,
    debug_mode=True,
    # markdown=True
)

# ---------------------------------------------------------------------
#  RAG Agent: Uses knowledge base to generate a risk financial narrative.
# ---------------------------------------------------------------------
risk_analysis_rag_agent = Agent(
    model=AOI(id=model_name,
    api_key=api_key,
    azure_endpoint=endpoint_url,
    azure_deployment=deployment_name
    ),
    description="You are an agent that comes up with draft for a corporate loan for a company based on the information in the knowledge base.",
    # tools=[YFinanceTools(key_financial_ratios = True, stock_fundamentals = True, company_news=True)], #add ml tool also
    # run_id=run_id,
    # user_id=user,
    knowledge=knowledge_base,
    instructions="""You are tasked with generating an analysis. Query the knowledge base for:
 
1. Positives - Financial strengths, strategic wins, liquidity improvements, favorable trends.
 
2. Risks - Operational challenges, weakening financials, refinancing issues, regulatory or macro headwinds.
 
3. Major Changes - Capital structure changes, M&A, management shifts, covenant changes, business model updates.

4. Financial Metrics and Ratios - Analyze profitability, liquidity, leverage, efficiency, returns, growth, and cash flow metrics from annual reports and balance sheets to assess financial performance.
 
Output a well-structured credit narrative under the following headings:
 
1. Positives
2. Risks
3. Major Changes
4. Financial Metrics and Ratios
""",
    # add_context_instructions = ""
    # use_tools=True,
    show_tool_calls=True,
    debug_mode=True,
    # markdown=True
)

# ---------------------------------------------------------------------
# Credit Narrative Agent: A simple agent that processes user feedback.
# ---------------------------------------------------------------------
class NarrativeAgent:
    def process_narrative(self, credit_text: str) -> str:
        narrative_chat_completion = client.chat.completions.create(

        messages=[
            {
                "role": "system",
                "content": f"""Based on the context provided to you, generate a comprehensive credit narrative addressing the users query. Provide a well-structured credit narrative that integrates all relevant information from the context.
Include sections under the following headings:
1. financial performance
2. industry position
3.credit risks
4.recommendations"""
            },
            {
                "role": "user",
                "content": f"""CONTEXT:
                {credit_text}""",
            }
        ],

        model=model_name,
        # temperature=0.5,
        )
        return narrative_chat_completion.choices[0].message.content

    
narrative_agent = NarrativeAgent()

# ---------------------------------------------------------------------
# Narrative Calculation Agent: Uses ML score and narrative to calculate application risk score.
# ---------------------------------------------------------------------

class ScoreStructure(BaseModel):
    score: float = Field(..., description="The score you will generate for the loan application based on the narrative and ml model score provided to you, must be in the range of 0-100")


# -----------------------------
# Single Function Implementation
# -----------------------------

def safe_div(numerator, denominator):
    """Safely divide two numbers, return None if denominator is zero or None."""
    try:
        if denominator == 0 or denominator is None:
            return None
        return numerator / denominator
    except:
        return None
def evaluate_loan_risk(company_name: str, loan_value: float, collateral_value: float, credit_score: float) -> str:
    """
    Evaluates loan risk based on company name, loan value, collateral value,
    and credit score in a single function (no nested implementation).
    """
    print(f"Step 1: Evaluating loan risk for company '{company_name}'...")
 
    # -----------------------
    # Step 1: Search Ticker Symbol
    # -----------------------
    print(f"Looking up ticker symbol for '{company_name}'...")
    url = f"https://query1.finance.yahoo.com/v1/finance/search?q={company_name}"
    headers = {"User-Agent": "Mozilla/5.0"}
    response = requests.get(url, headers=headers)
   
    print(f"API Response Status Code: {response.status_code}")
    if response.status_code != 200:
        print(f"Error: Failed to fetch ticker data for {company_name}. Status Code: {response.status_code}")
        return {"error": f"Failed to fetch ticker data for {company_name}. Status Code: {response.status_code}"}
   
    results = response.json().get("quotes", [])
    print(f"Search Results: {results}")
    ticker = None
    for result in results:
        symbol = result.get("symbol", "")
        if symbol.endswith(".NS"):
            ticker = symbol
        elif symbol.endswith(".BO"):
            ticker = symbol[:-3] + ".NS"
        else:
            ticker = symbol + ".NS"
        if ticker:
            break
 
    if not ticker:
        print(f"Error: Could not find a valid ticker for {company_name}.")
        return {"error": f"Could not find a valid ticker for {company_name}."}
   
    print(f"Found ticker: {ticker}")
 
    # -----------------------
    # Step 2: Fetch Financial Data from Excel
    # -----------------------
    print("Fetching financial data from 'Company_Financials_FY2024.xlsx'...")
    try:
        df = pd.read_excel("Company_Financials_FY2024.xlsx")
    except FileNotFoundError:
        print("Error: Financial Excel file not found.")
        return {"error": "Financial Excel file not found. Ensure 'Company_Financials_FY2024.xlsx' is present."}
 
    if "Company" not in df.columns:
        print("Error: Excel file does not have a 'Company' column.")
        return {"error": "Excel file must have a 'Company' column."}
 
    row = df[df['Company'] == ticker]
    if row.empty:
        print(f"Error: Financial data for ticker {ticker} not found in the Excel sheet.")
        return {"error": f"Financial data for ticker {ticker} not found in the Excel sheet."}
   
    row = row.iloc[0]  # Select the first matching row
    print("Financial data fetched successfully.")
 
    data = {
        "Net Profit Margin %": row.get("Net Profit Margin %"),
        "Return on Equity %": row.get("Return on Equity %"),
        "Return on Assets %": row.get("Return on Assets %"),
        "Current Ratio": row.get("Current Ratio"),
        "Asset Turnover Ratio": row.get("Asset Turnover Ratio"),
        "Debt Equity Ratio": row.get("Debt Equity Ratio"),
        "Debt To Asset Ratio": row.get("Debt To Asset Ratio"),
        "Interest Coverage Ratio": row.get("Interest Coverage Ratio"),
        "Loan Value": loan_value,
        "Collateral Value": collateral_value,
        "Credit Score": credit_score,
        "LtC": safe_div(loan_value, collateral_value),  # Loan-to-Collateral Ratio
    }
    print(f"Data for risk analysis: {data}")
 
    # -----------------------
    # Step 3: Compute Risk Score
    # -----------------------
    print("Computing risk scores...")
    try:
        df1 = pd.read_excel("Company_Financials_Synthetic_First100.xlsx")
    except FileNotFoundError:
        print("Error: Synthetic financial data file not found.")
        return {"error": "Synthetic financial data file not found. Ensure 'Company_Financials_Synthetic_First100.xlsx' is present."}
 
    df2 = pd.DataFrame([data])
    df = pd.concat([df1, df2], axis=0)
 
    # Keep only relevant columns
    df = df[[
        "Net Profit Margin %", "Return on Equity %", "Return on Assets %",
        "Current Ratio", "Asset Turnover Ratio", "Debt Equity Ratio", "Debt To Asset Ratio",
        "Interest Coverage Ratio", "Loan Value", "Collateral Value", "Credit Score",
        "LtC"
    ]]
    print("Relevant columns selected.")
 
    # Drop unused columns and fill missing values
    df = df.drop(columns=["Loan Value", "Collateral Value"])
    df = df.fillna(df.median(numeric_only=True))
    print("Data prepared and missing values handled.")
 
    # Scale the data using MinMaxScaler
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(df)
    scaled_df = pd.DataFrame(scaled_data, columns=df.columns)
    scaled_df = 1 + (100 * scaled_df)
    print("Data scaled successfully.")
 
    # Apply weights to calculate scores
    dict_fin_weights = {
        "Net Profit Margin %": 0.25, "Return on Equity %": 0.25, "Return on Assets %": 0.25,
        "Current Ratio": 0.25, "Asset Turnover Ratio": 0.1, "Debt Equity Ratio": 0.1, "Debt To Asset Ratio": -0.2
    }
    dict_repay_weights = {
        "Interest Coverage Ratio": 0.20, "Credit Score": 0.65, "LtC": 0.15
    }
 
    for col, weight in {**dict_fin_weights, **dict_repay_weights}.items():
        scaled_df[col] = scaled_df[col] * weight
    print("Weights applied to scaling.")
 
    scaled_df["Financial Risk Score"] = 100 - scaled_df[list(dict_fin_weights.keys())].sum(axis=1)
    scaled_df["Repayment Risk Score"] = 100 - scaled_df[list(dict_repay_weights.keys())].sum(axis=1)
    scaled_df["Final Risk Score"] = (
        scaled_df["Financial Risk Score"] * 0.3 +
        scaled_df["Repayment Risk Score"] * 0.7
    )
    print("Risk scores calculated.")
 
    # Extract the final risk score and Loan-to-Collateral Ratio for the new data row
    final_score = scaled_df.iloc[-1]["Final Risk Score"]
    ltc_ratio = df.iloc[-1]["LtC"]
    print(f"Final Risk Score: {final_score}, Loan-to-Collateral Ratio: {ltc_ratio}")
    return f"Final Risk Score: {final_score}, Loan-to-Collateral Ratio: {ltc_ratio}"
    # -----------------------
    # Step 4: Return Results
    # -----------------------
    # # result = {
    #     "Company": company_name,
    #     "Ticker": ticker,
    #     "Loan-to-Collateral Ratio": ltc_ratio,
    #     "Final Risk Score": final_score,
    # }
    # print("Result prepared:", result)
    # return result
# -----------------------------
# Single Tool for the Agent
# -----------------------------
def evaluate_company(
    company_name: str,
    loan_value: float,
    collateral_value: float,
    credit_score: float
) -> str:
    """Evaluates loan risk based on company financials and repayment ratio.

    Args: 
    company_name: str,
    loan_value: float,
    collateral_value: float,
    credit_score: float

    returns:
    ltc_ratio, risk_score: str
    """

    return evaluate_loan_risk(company_name, loan_value, collateral_value, credit_score)

risk_score_agent = Agent(
    model=AOI(id=model_name,
    api_key=api_key,
    azure_endpoint=endpoint_url,
    azure_deployment=deployment_name
    ),
    # name="risk_score_agent",
    description="You are an agent that calculates an application risk score based on the custom tool provided to you.",
    instructions="You will recieve a company name, loan ammount, collateral value, credit score. You will use these values in the custom tool which when given these values will return an application risk score.",
    tools=[evaluate_company],
)
 

risk_calculation_narrative_agent = Agent(
    model=AOI(id=model_name,
    api_key=api_key,
    azure_endpoint=endpoint_url,
    azure_deployment=deployment_name
    ),

    description="You are an agent that calculates an application risk score.",
    # tools=[evaluate_company_risk], #add ml tool also
    # run_id=run_id,
    # user_id=user,
    # knowledge=knowledge_base,
    instructions="""You will receive an application risk score from a rule-based model along with the loan amount and loan-to-collateral (LTC) ratio. You will also be provided with a credit risk narrative that outlines key financial information, company performance, and potential credit risks.
 
Your task is to review the financial and long-term stock information in narrative, consider the LTC ratio, and adjust the rule-based model's score using a risk-based scale. The final risk score should range from 0 to 100, where 0 indicates no risk in approving a loan and 100 indicates the highest risk. Adjust the base score as follows:
 
Narrative Adjustments:
- Extremely Positive Signals: Decrease score significantly (e.g., by 25-35 points).
- Moderately Positive Signals: Decrease score slightly (e.g., by 10-15 points).
- Neutral or Balanced Signals: Minimal adjustment (e.g., ±0-10 points).
- Moderately Negative Signals: Increase score slightly (e.g., by 10-15 points).
- Extremely Negative Signals: Increase score significantly (e.g., by 25-35 points).
 
Loan Amount and LTC Ratio Adjustments:
- Low LTC Ratio (<1): Generally low risk, but:
If the loan amount is very small (e.g., under 10), the final score should be extremely low (under 30), regardless of other factors.
Larger loan amounts may warrant moderate scores even with low LTC ratios.
 
- High LTC Ratio (≥1): Indicates higher risk, particularly for large loan amounts:
Assign higher scores for higher LTC ratios and large loan amounts.
Small loan amounts may reduce the risk impact even with high LTC ratios.
 
Adjust the risk score smoothly and proportionally to reflect the strength of financial and long-term stock information in the narrative and financial context. Do not consider sentiment or short-term stock information in the evaluation. Return a single integer between 0 and 100 as the final adjusted risk score.
 """,
    # add_context_instructions = ""
    # use_tools=True,
    show_tool_calls=True,
    response_model=ScoreStructure,
    # temperature=0.4
    # debug_mode=True,
    # markdown=True
)

# ---------------------------------------------------------------------
# Feedback Agent: A simple agent that processes user feedback.
# (It does not require any external tools.)
# ---------------------------------------------------------------------
class FeedbackAgent:
    def process_feedback_narrative(self, feedback_text: str) -> str:
        feedback_chat_completion = client.chat.completions.create(

        messages=[
            {
                "role": "system",
                "content": f"""You are an expert feedback analyzer for credit narratives. Analyze the feedback provided to you and structure it into clear improvement points of missing information to provide as instructions to another agent. Ensure that any gaps, Missing information or other specific areas for improvement are highlighted if required."""
            },
            {
                "role": "user",
                "content": f"""FEEDBACK:
                {feedback_text}""",
            }
        ],

        model=model_name,
        # temperature=0.5,
        )
        return feedback_chat_completion.choices[0].message.content

    def process_feedback_note(self, feedback_text: str) -> str:
        feedback_chat_completion = client.chat.completions.create(

        messages=[
            {
                "role": "system",
                "content": f"""You are an expert feedback analyzer for credit notes. Analyze the feedback provided to you and structure it into clear improvement points of missing information to provide as instructions to another agent. Ensure that any gaps, Missing information or other specific areas for improvement are highlighted if required."""
            },
            {
                "role": "user",
                "content": f"""FEEDBACK:
                {feedback_text}""",
            }
        ],

        model=model_name,
        # temperature=0.5,
        )
        return feedback_chat_completion.choices[0].message.content

feedback_agent = FeedbackAgent()

# ---------------------------------------------------------------------
# Credit Note Agent: Uses a provided template to generate a credit note.
# ---------------------------------------------------------------------
class CreditNoteAgent:
    def __init__(self, template: str):
        self.template = template
    def generate_credit_note(self, narrative: str, user_query: str, loan_details: str, feedback_history: list = None) -> str:
        """
        Generate a formal credit note based on the provided information.
       
        Args:
            query (str): User query or company name.
            loan_amount (float): Loan amount in Cr.
            loan_purpose (str): Purpose of the loan.
            feedback_history (list, optional): List of past feedback. Defaults to None.
 
        Returns:
            str: The generated credit note.
        """
        current_date = datetime.now().strftime("%Y-%m-%d")
       
        # Step 1: Process Feedback Context
        feedback_context = ""
        if feedback_history:
            feedback_context = "\n".join([f"Feedback {i + 1}: {fb}" for i, fb in enumerate(feedback_history)])
        else:
            feedback_context = "No feedback provided yet."

        
        formatted_loan_amount = f"{loan_details} Cr"
 
        # Step 2: Prepare the System Prompt
        system_prompt = f"""You are a financial analyst generating formal credit notes.
        Use the following template to create the credit note:
 
        TEMPLATE:
        {self.template}
       
        Replace placeholders appropriately:
        - [COMPANY_NAME]: {user_query}
        - [AMOUNT]: {formatted_loan_amount}
        - [DATE]: {current_date}
       
        Maintain a professional tone and formatting throughout the credit note.
        Consider the following feedback history while generating the credit note:
        {feedback_context}
        Fill other sections using the narrative below. Maintain professional tone and formatting.
        If any fields are missing or ambiguous, make reasonable assumptions but ensure consistency with the narrative.
        """
 
        # Step 3: Send the request to the LLM
        credit_note_chat_completion = client.chat.completions.create(
            messages=[
                {
                    "role": "system",
                    "content": f"""Based on the credit narrative provided to you, generate a
                    formal credit note using the provided template. Replace all placeholder
                    fields such as [COMPANY_NAME], [AMOUNT], [COLLATERAL], [PURPOSE], [DATE], etc.
                   
                    Ensure the credit note reflects the following feedback history if available:
                    {feedback_context}
                   
                    TEMPLATE:
                    {self.template}"""
                },
                {
                    "role": "user",
                    "content": f"""USER QUERY: {user_query}\n\nCREDIT NARRATIVE:
                    {narrative}
                   
                    Narrative/loan details:
                    - LOAN DETAILS: {loan_details}"""
                }
            ],
            model=model_name,
            temperature=0.4,
        )
 
        # Step 4: Return the generated credit note
        return credit_note_chat_completion.choices[0].message.content
 
#Adjust the template format as required.
credit_note_template = """
Credit Note
**Company Name:** {{COMPANY_NAME}}  
**Date:** {{DATE}}  
**Ref:** Credit Loan for {{AMOUNT}}
 
---
 
## Loan Details
 
| S.No | Field                                      | Description                                                                 |
|------|--------------------------------------------|-----------------------------------------------------------------------------|
| 1    | Borrowers                                  | {{borrowers}}                                                               |
| 2    | Amount                                     | {{amount}}                                                                  |
| 3    | Purpose                                    | {{purpose}}                                                                 |
| 4    | Drawdown                                   | {{drawdown}}                                                                |
| 5    | Term                                       | {{term}}                                                                    |
| 6    | Repayment                                  | {{repayment}}                                                               |
| 7    | Security                                   | {{security}}                                                                |
| 8    | Security Cover                             | {{security_cover}}                                                          |
| 9    | Interest Rate                              | {{interest_rate}}                                                           |
| 10   | Admin Fees                                 | {{admin_fees}}                                                              |
| 11   | Other Conditions                           | {{other_conditions}}                                                        |
| 12   | Representations, Warranties, Covenants     | {{representations_warranties_covenants}}                                   |
| 13   | Events of Default                          | {{events_of_default}}                                                       |
| 14   | Legal Costs                                | {{legal_costs}}                                                             |
| 15   | Law and Jurisdiction                       | {{law_and_jurisdiction}}                                                    |
 
---
 
## Signatures
 
**Authorized Signatory (Lender):**  
Name: {{signatory_1}}  
Designation: {{designation_1}}  
Date: {{date_1}}  
Signature: _____________________________  
 

 
**Authorized Signatory (Borrower):**  
Name: {{signatory_2}}  
Designation: {{designation_2}}  
Date: {{date_2}}  
Signature: _____________________________
"""
credit_note_agent = CreditNoteAgent(credit_note_template)
 
 
# ---------------------------------------------------------------------
# Main pipeline: Combines all the agents and human-in-the-loop interactions.
# ---------------------------------------------------------------------
def main():
    input_data = "Generate me a credit note for a corporate loan for Microsoft"
    # Step 1: Generate and approve risk narrative
    # while True:  
    print("Finance Agent")
    finance_narrative_response: RunResponse = risk_analysis_finance_agent.run(input_data)
    pprint_run_response(finance_narrative_response, markdown=True, show_time=True)
    print("RAG Agent")
    rag_agent_response: RunResponse = risk_analysis_rag_agent.run(input_data)
    pprint_run_response(rag_agent_response, markdown=True, show_time=True)

    print("\n--- Generating Risk Narrative ---")
    narrative_response = narrative_agent.process_narrative(finance_narrative_response.content + "\n\n" + rag_agent_response.content)
    print(narrative_response)
    company = input("Enter company name: ").strip()
    loan = float(input("Enter loan value: "))
    collateral = float(input("Enter collateral value: "))
    credit = int(input("Enter credit score (300-900): "))
    risk_score,ltc = evaluate_company_risk(company, loan, collateral, credit)
    print(risk_score)
    score_response: RunResponse = risk_calculation_narrative_agent.run("rule based model score:" + str(risk_score) + "\n\n"+ "Loan Amount:" + str(loan) + "\n\n" + "Loan to Collateral Ratio:" + str(ltc) + "\n\n" + narrative_response)
    pprint_run_response(score_response, markdown=True, show_time=True)
    print("----------",score_response)
    while True:  # Approval inner loop
        approval = input("Do you approve the narrative? (yes/no): ").strip().lower()
        
        if approval == "yes":
            final_narrative_response = narrative_response  # Store final response
            break  # Exit inner loop

        # Feeback if user isn't happy
        user_feedback_narrative = input("Enter your feedback to improve the narrative: ")
        structured_fb_narrative = feedback_agent.process_feedback_narrative(user_feedback_narrative)
        input_data = input_data + " " + structured_fb_narrative 
        print("Feedback received. Regenerating narrative...\n")

        finance_narrative_response: RunResponse = risk_analysis_finance_agent.run(input_data)
        pprint_run_response(narrative_response, markdown=True, show_time=True)
        rag_agent_response: RunResponse = risk_analysis_rag_agent.run(input_data)
        pprint_run_response(rag_agent_response, markdown=True, show_time=True)

        print("\n--- Generating Risk Narrative ---")
        narrative_response = narrative_agent.process_narrative(input_data + "\n\n" + finance_narrative_response.content + "\n\n" + rag_agent_response.content)
        print(narrative_response)

        score_response: RunResponse = risk_calculation_narrative_agent.run(narrative_response)
        pprint_run_response(score_response, markdown=True, show_time=True)
        # print(score_response)

        # break  # Exit outer loop after approval

    # print(final_narrative_response)
    # Step 2: Generate and approve credit note
    # while True:  
    print("\n--- Generating Credit Note ---")
    credit_note = credit_note_agent.generate_credit_note(final_narrative_response, input_data)
    print("\nGenerated Credit Note:")
    print(credit_note)

    while True:  # Approval loop for credit note
        approval_cn = input("Do you approve the credit note? (yes/no): ").strip().lower()
        
        if approval_cn == "yes":
            break  # Exit credit note loop and complete the flow

        # Feedback if user isnt happy
        print("Feedback received. Regenerating credit note...\n")
        user_feedback_credit = input("Enter your feedback to improve the credit note: ")
        structured_fb_credit = feedback_agent.process_feedback_note(user_feedback_credit)
        input_data = input_data + " " + structured_fb_credit  # Update input with feedback
        # print(input_data)

        credit_note = credit_note_agent.generate_credit_note(final_narrative_response, input_data)
        # print("regen-note")
        print(credit_note)

        # break  #Exit outer loop after approval

    print("\nProcess Complete: Final narrative and credit note have been approved.")

if __name__ == "__main__":
    main()