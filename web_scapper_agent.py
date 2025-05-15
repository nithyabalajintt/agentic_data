Nithya Balaji and Ankam Rishitha, below is the agentic code, refine the prompts accordingly and integrate it with the flow:
 
# Install with pip install firecrawl-py
from firecrawl import FirecrawlApp
import os
from dotenv import load_dotenv
from agno.agent import Agent
from agno.models.azure import AzureOpenAI as AOI
from agno.tools import tool
import shutil
from agno.tools import tool
import warnings
warnings.filterwarnings("ignore")
# Load environment variables from a .env file
load_dotenv()
#GPT-4o
model_name = os.environ.get("AZURE_OPENAI_MODEL_NAME")
endpoint_url = os.environ.get("AZURE_OPENAI_ENDPOINT")
api_key = os.environ.get("AZURE_OPENAI_API_KEY")
deployment_name = os.environ.get("AZURE_OPENAI_DEPLOYMENT")
version_number = os.environ.get("API_VERSION_GA")
@tool
def scraping(url, formats, includeTags):
    """
    Scrape web data from a specified URL using Firecrawl.
    Args:
        url (str): The URL of the web page to scrape.
        formats (list of str): List of output formats to return, e.g., ['markdown', 'text', 'json'].
        includeTags (list of str): CSS selectors to specify which elements to include in the output.
    Returns:
        dict: The Firecrawl API response containing the scraped results.
    Example:
        result = scraping(
            url="https://www.example.com/page",
            formats=["markdown"],
            includeTags=['.indicators-table_row__Q16TJ']
        )
    """
    app = FirecrawlApp(api_key='fc-b80b34b95db745d4ae1aa9ffe71e257b')
    response = app.scrape_url(
        url=url,
        formats=['markdown'],
        includeTags=['.indicators-table_row__Q16TJ']
    )
    
    return str(response)
client = AOI(
        azure_endpoint=endpoint_url,
        azure_deployment=deployment_name,
        api_key=api_key,
        api_version=version_number,
    )
# message_prompt = [
#             {
#                 "role": "system",
#                 "content": f"""Based on the context provided to you, give the following annual ratios for the company:
#                 Net profit margin, ROE, ROA, Current Ratio, Asset Turnover, Debt Equity Ratio """
#             },
#             {
#                 "role": "user",
#                 "content": f"""CONTEXT:
#                 {response}""",
#             }
#         ]
# narrative_chat_completion = client.chat.completions.create(messages=message_prompt,model=model_name)
# print(narrative_chat_completion.choices[0].message.content)
web_scraper_agent = Agent(
    model=AOI(
        id=model_name,
        api_key=api_key,
        azure_endpoint=endpoint_url,
        azure_deployment=deployment_name
    ),
    description="You are a finance analyst that researches into income, balance sheet and cash flow statements, annual reports of the companies.",
    instructions=["You have access to Scraping tool. You will be given a company link in investing.com. Call that tool with format of markdown and the class '.indicators-table_row__Q16TJ'. You have to give me ratios like Net Profit Margin,Return on Equity, Quick Ratio,Return on Assets,Current Ratio,Asset Turnover Ratio,Debt Equity Ratio"],
    tools=[scraping],
    show_tool_calls=True,
    markdown=True,
    debug_mode=True
)
 
web_scraper_agent.print_response("https://www.investing.com/equities/kothari-products-ltd-ratios", debug=True)
 
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
        "Current Ratio", "Asset Turnover Ratio", "Debt Equity Ratio", "Loan Value", 
        "Collateral Value","Credit Score","LtC"
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
        "Net Profit Margin %": 0.25, "Return on Equity %": 0.25, "Return on Assets %": 0.15,
        "Current Ratio": 0.15, "Asset Turnover Ratio": 0.1, "Debt Equity Ratio": 0.1, 
    }
    dict_repay_weights = { "Credit Score": 0.75, "LtC": 0.25
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
