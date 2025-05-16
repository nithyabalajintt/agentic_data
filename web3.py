import json
from dotenv import load_dotenv
from agno.agent import Agent
from agno.models.azure import AzureOpenAI as AOI
from agno.tools import tool
from firecrawl import FirecrawlApp
import os
import warnings
 
warnings.filterwarnings("ignore")
 
# Load environment variables
load_dotenv()
 
# Azure OpenAI API creds
model_name = os.environ.get("AZURE_OPENAI_MODEL_NAME")
endpoint_url = os.environ.get("AZURE_OPENAI_ENDPOINT")
api_key = os.environ.get("AZURE_OPENAI_API_KEY")
deployment_name = os.environ.get("AZURE_OPENAI_DEPLOYMENT")
version_number = os.environ.get("API_VERSION_GA")
 
 
# Define the scraping tool
@tool
def scraping(url, formats, includeTags):
    """
    Scrape web data from a specified URL using Firecrawl.
    """
    print(f"Scraping URL: {url} with formats {formats} and selectors {includeTags}")
    app = FirecrawlApp(api_key='fc-f13f627cdb1445288a8cc91419d65b96')
    try:
        response = app.scrape_url(
            url=url,
            formats=['markdown'],
            includeTags=['.indicators-table_row__Q16TJ']
        )
        print("Scraping response:", response)
        return str(response)
    except Exception as e:
        print(f"Scraping error: {e}")
        return None
 
 
# Define the Azure OpenAI Client and Agent
client = AOI(
    azure_endpoint=endpoint_url,
    azure_deployment=deployment_name,
    api_key=api_key,
    api_version=version_number,
)
 
agent = Agent(
    model=AOI(
        id=model_name,
        api_key=api_key,
        azure_endpoint=endpoint_url,
        azure_deployment=deployment_name
    ),
    description="You are a finance analyst that researches into income, balance sheet, cash flow statements, and annual reports of companies.",
    instructions=[
        "You have access to Scraping tool. You will be given a company link in Investing.com.",
        "Call the scraping tool with the format of markdown and the class '.indicators-table_row__Q16TJ'.",
        "Return ratios like Net Profit Margin, Return on Equity, Quick Ratio, Return on Assets, Current Ratio, Asset Turnover Ratio, Debt Equity Ratio.",
        "Do not include periods (e.g., TTM, MRQ) in your response keys.",
        "The response should be a dictionary with ratio names as keys and corresponding values."
    ],
    tools=[scraping],
    show_tool_calls=True,
    markdown=True,
    debug_mode=True
)
 
 
def fetch_company_ratios(company_name: str) -> dict:
    """
    Fetch financial ratios for the given company and return the result as a dictionary.
 
    Parameters:
        company_name (str): Name of the company to fetch ratios for.
 
    Returns:
        dict: A dictionary containing the financial ratios.
    """
    base_url = "https://www.investing.com/equities/"
    formatted_company_name = company_name.lower().replace(" ", "-")
    url = f"{base_url}{formatted_company_name}-ratios"
 
    print(f"Fetching financial ratios for: {company_name}")
    print(f"Generated URL: {url}")
 
    try:
        # Call the agent to fetch the response
        response = agent.run(url, debug=True)
        print("Agent response:", response)
        return response
 
    except Exception as e:
        print(f"Error during fetch_company_ratios: {e}")
        return {}
 
 
# Main function to trigger the agent only
def main():
    """
    Main function to get user input and run the agent-based scraping and analysis.
    """
    company_name = input("Enter the company name: ").strip()
    fetch_company_ratios(company_name)
 
 
if __name__ == "__main__":
    main()
 