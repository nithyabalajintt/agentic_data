import os
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from agno.agent import Agent
from agno.tools.yfinance import YFinanceTools
from agno.models.azure import AzureOpenAI
from openai import AzureOpenAI
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Azure OpenAI GPT API Configuration
model_name = os.environ.get("AZURE_OPENAI_MODEL_NAME")
endpoint_url = os.environ.get("AZURE_OPENAI_ENDPOINT")
api_key = os.environ.get("AZURE_OPENAI_API_KEY")
deployment_name = os.environ.get("AZURE_OPENAI_DEPLOYMENT")


def fetch_financial_ratios(
    company_name: str,
    api_key: str,
    endpoint_url: str,
    deployment_name: str
) -> pd.DataFrame:
    """
    Fetch financial ratios for a company and return them as a single-row Pandas DataFrame.

    Parameters:
    - company_name (str): Name of the company to fetch financial ratios for.
    - api_key (str): Azure OpenAI API Key.
    - endpoint_url (str): Azure OpenAI endpoint URL.
    - deployment_name (str): Azure OpenAI deployment name.

    Returns:
    - pd.DataFrame: A single-row DataFrame containing the company's financial ratios.
    """
    # Define the Agent with AzureOpenAI Model and YFinanceTools
    risk_analysis_finance_agent = Agent(
        model=AzureOpenAI(
            api_key=api_key,
            azure_endpoint=endpoint_url,
            azure_deployment=deployment_name,
            api_version = "2024-10-21"
        ),
        description=(
            "An AI agent that fetches financial ratios of a company via YFinanceTools."
            "The ratios will be returned as a single-row Pandas DataFrame."
        ),
        tools=[YFinanceTools(key_financial_ratios=True)],
        instructions=f"""
        Fetch the key financial ratios for the company '{company_name}'.
        Organize them into a Pandas DataFrame with column names aligning with the ratio names.
        Return the DataFrame with the following columns:
        - Net Profit Margin %
        - Return on Equity %
        - Return on Assets %
        - Current Ratio
        - Asset Turnover Ratio
        - Debt to Equity Ratio
        - Debt to Asset Ratio
        - Interest Coverage Ratio.
        """
    )

    # Query the Agent to Fetch Financial Ratios
    query = f"Fetch financial ratios for '{company_name}' and return them as a DataFrame."
    response = risk_analysis_finance_agent.run(query)

    # Check if the response is a valid DataFrame
    if isinstance(response, pd.DataFrame):
        return response
    else:
        raise ValueError(f"Error fetching financial ratios: {response}")


def compute_risk_score(
    company_name: str,
    ticker: str,
    financial_ratios_df: pd.DataFrame,
    loan_value: float,
    collateral_value: float,
    credit_score: float
) -> list:
    """
    Compute risk scores based on financial ratios, loan details, and credit score.

    Parameters:
    - company_name (str): Company name for identification.
    - ticker (str): Company's ticker symbol.
    - financial_ratios_df (pd.DataFrame): DataFrame containing financial ratios.
    - loan_value (float): Loan amount requested.
    - collateral_value (float): Collateral value offered.
    - credit_score (float): Credit score of the company.

    Returns:
    - list: Computed risk scores and relevant loan-to-collateral ratio.
    """

    # Combine relevant input data into a dictionary for scoring
    financial_ratios = financial_ratios_df.iloc[0].to_dict()
    loan_to_collateral_ratio = loan_value / (collateral_value or 1)  # Prevent division by zero

    data = {
        **financial_ratios,
        "Loan Value": loan_value,
        "Collateral Value": collateral_value,
        "Credit Score": credit_score,
        "LtC": loan_to_collateral_ratio
    }
    print(f"Data for risk analysis: {data}")

    # Load synthetic dataset
    try:
        df1 = pd.read_excel("Company_Financials_Synthetic_First100.xlsx")
    except FileNotFoundError:
        raise FileNotFoundError("Synthetic financial data file not found.")

    # Combine with the new data
    df2 = pd.DataFrame([data])
    df = pd.concat([df1, df2], axis=0)

    # Preprocess the DataFrame
    columns_to_keep = [
        "Net Profit Margin %", "Return on Equity %", "Return on Assets %",
        "Current Ratio", "Asset Turnover Ratio", "Debt Equity Ratio",
        "Debt To Asset Ratio", "Interest Coverage Ratio", "Loan Value",
        "Collateral Value", "Credit Score", "LtC"
    ]
    df = df[columns_to_keep]
    df = df.drop(columns=["Loan Value", "Collateral Value"])  # Drop unused columns
    df = df.fillna(df.median(numeric_only=True))

    # Scale the data
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(df)
    scaled_df = pd.DataFrame(scaled_data, columns=df.columns)
    scaled_df = 1 + (100 * scaled_df)

    # Apply weights to compute scores
    financial_weights = {
        "Net Profit Margin %": 0.25,
        "Return on Equity %": 0.25,
        "Return on Assets %": 0.25,
        "Current Ratio": 0.25,
        "Asset Turnover Ratio": 0.1,
        "Debt Equity Ratio": 0.1,
        "Debt To Asset Ratio": -0.2
    }
    repayment_weights = {
        "Interest Coverage Ratio": 0.2,
        "Credit Score": 0.65,
        "LtC": 0.15
    }

    for column, weight in {**financial_weights, **repayment_weights}.items():
        scaled_df[column] = scaled_df[column] * weight

    scaled_df["Financial Risk Score"] = 100 - scaled_df[list(financial_weights.keys())].sum(axis=1)
    scaled_df["Repayment Risk Score"] = 100 - scaled_df[list(repayment_weights.keys())].sum(axis=1)
    scaled_df["Final Risk Score"] = (
        scaled_df["Financial Risk Score"] * 0.3 +
        scaled_df["Repayment Risk Score"] * 0.7
    )

    # Extract results
    final_score = scaled_df.iloc[-1]["Final Risk Score"]
    ltc_ratio = df.iloc[-1]["LtC"]
    result = [company_name, ticker, ltc_ratio, final_score]
    print(f"Final Risk Score: {final_score}, Loan-to-Collateral Ratio: {ltc_ratio}")
    return result


if __name__ == "__main__":
    print("Enter Company Loan Evaluation Details:")
    company_name = input("Company Name: ").strip()
    loan_value = float(input("Loan Amount: "))
    collateral_value = float(input("Collateral Amount: "))
    credit_score = float(input("Credit Score (300-900): "))

    try:
        # Fetch ratios
        financial_ratios_df = fetch_financial_ratios(
            company_name=company_name,
            api_key=api_key,
            endpoint_url=endpoint_url,
            deployment_name=deployment_name,
            api_version = "2024-10-21"
        )

        # Compute risk scores
        result = compute_risk_score(
            company_name=company_name,
            ticker="TICKER",  # Replace with logic to get ticker
            financial_ratios_df=financial_ratios_df,
            loan_value=loan_value,
            collateral_value=collateral_value,
            credit_score=credit_score
        )

        print("\nLoan Risk Analysis Result:")
        print(f"Company Name: {result[0]}")
        print(f"Ticker Symbol: {result[1]}")
        print(f"Loan-to-Collateral Ratio: {result[2]}")
        print(f"Final Risk Score: {result[3]}")

    except Exception as e:
        print(f"\nAn error occurred: {e}")