import json
import pandas as pd  # Import pandas for dataframe creation
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
    app = FirecrawlApp(api_key='fc-b80b34b95db745d4ae1aa9ffe71e257b')
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


def safe_div(numerator, denominator):
    """
    Safely perform division, returning None if the denominator is zero.
    """
    return numerator / denominator if denominator else None


def parse_ratios(response: str) -> dict:
    """
    Parse raw response to extract financial ratios with their original terms.

    Parameters:
        response (str): The raw response text containing the financial ratios.

    Returns:
        dict: A dictionary with financial ratios and extracted values.
    """
    ratios = {
        "Net Profit Margin": None,
        "Return on Equity": None,
        "Return on Assets": None,
        "Current Ratio": None,
        "Quick Ratio": None,
        "Asset Turnover Ratio": None,
        "Debt Equity Ratio": None,
    }

    try:
        # Attempt to parse the response as JSON
        print(f"Raw response:\n{response}")
        parsed_response = json.loads(response)  # Assuming response comes as JSON

        # Populate the ratios dictionary with the corresponding values
        for key in ratios.keys():
            if key in parsed_response:
                ratios[key] = parsed_response[key]

    except json.JSONDecodeError as e:
        print(f"JSON parsing error: {e}")
    except Exception as e:
        print(f"Unexpected error while parsing ratios: {e}")
    
    return ratios


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
        response = agent.print_response(url, debug=True)
        print("Agent response:", response)

        if not response or "error" in response.lower():
            print("Error or empty response from the agent.")
            return {}

        # Parse ratios from the received data
        return parse_ratios(response)

    except Exception as e:
        print(f"Error during fetch_company_ratios: {e}")
        return {}


def create_single_row_dataframe(
    company_name: str,
    loan_value: float,
    collateral_value: float,
    credit_score: int
) -> pd.DataFrame:
    """
    Create a single-row dataframe with financial ratios and user-provided input.

    Parameters:
        company_name (str): Name of the company to fetch ratios for.
        loan_value (float): Loan value provided by the user.
        collateral_value (float): Collateral value provided by the user.
        credit_score (int): Credit score provided by the user.

    Returns:
        pd.DataFrame: A single-row dataframe with financial ratios, user data, and calculated fields.
    """
    # Fetch financial ratios as a dictionary
    financial_ratios = fetch_company_ratios(company_name)

    # Build the final dictionary for the DataFrame
    data = {
        "Net Profit Margin": financial_ratios.get("Net Profit Margin"),
        "Return on Equity": financial_ratios.get("Return on Equity"),
        "Return on Assets": financial_ratios.get("Return on Assets"),
        "Current Ratio": financial_ratios.get("Current Ratio"),
        "Quick Ratio": financial_ratios.get("Quick Ratio"),
        "Asset Turnover Ratio": financial_ratios.get("Asset Turnover Ratio"),
        "Debt Equity Ratio": financial_ratios.get("Debt Equity Ratio"),
        "Loan Value": loan_value,
        "Collateral Value": collateral_value,
        "Credit Score": credit_score,
        "LtC": safe_div(loan_value, collateral_value),  # Loan-to-Collateral Ratio
    }
    print(f"Data for risk analysis: {data}")

    # Create and return the single-row DataFrame
    return pd.DataFrame([data])


# Main function to orchestrate the workflow with user inputs
def main():
    """
    Main function to get user inputs and execute the workflow.
    """
    # Get user inputs interactively
    company_name = input("Enter the company name: ").strip()
    try:
        loan_value = float(input("Enter the loan value (e.g., 1000000): ").strip())
        collateral_value = float(input("Enter the collateral value (e.g., 1500000): ").strip())
        credit_score = int(input("Enter the credit score (e.g., 720): ").strip())
    except ValueError:
        print("Invalid input! Please enter numerical values for loan, collateral, and credit score.")
        return

    # Create the consolidated DataFrame
    data = create_single_row_dataframe(company_name, loan_value, collateral_value, credit_score)

    # Show the output
    print("Final DataFrame:")
    print(data)


# Execute the main function when the script is run
if __name__ == "__main__":
    main()