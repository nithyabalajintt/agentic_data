import os
import requests
import pandas as pd
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from sklearn.preprocessing import MinMaxScaler
from openai import AzureOpenAI
 
# Load environment variables
load_dotenv()
 
# Azure OpenAI GPT-4o
model_name = os.environ.get("AZURE_OPENAI_MODEL_NAME")
endpoint_url = os.environ.get("AZURE_OPENAI_ENDPOINT")
api_key = os.environ.get("AZURE_OPENAI_API_KEY")
deployment_name = os.environ.get("AZURE_OPENAI_DEPLOYMENT")
version_number = os.environ.get("API_VERSION_GA")
 
# Azure OpenAI Client
client = AzureOpenAI(
    azure_endpoint=endpoint_url,
    azure_deployment=deployment_name,
    api_key=api_key,
    api_version=version_number,
)
  
class ScoreStructure(BaseModel):
    score: float = Field(..., description="Score for the loan application based on narrative and model.")
 
def search_ticker_by_company_name(company_name):
    url = f"https://query1.finance.yahoo.com/v1/finance/search?q={company_name}"
    headers = {"User-Agent": "Mozilla/5.0"}
    response = requests.get(url, headers=headers)
    response.raise_for_status()
    results = response.json().get("quotes", [])
 
    for result in results:
        symbol = result.get("symbol", "")
        if symbol.endswith(".NS"):
            return symbol
        elif symbol.endswith(".BO"):
            return symbol[:-3] + ".NS"
        else:
            return symbol + ".NS"
    return None
 
def safe_div(numerator, denominator):
    try:
        if denominator == 0 or denominator is None:
            return None
        return numerator / denominator
    except:
        return None
 
def fetch_financial_data_from_excel(ticker, loan_value, collateral_value, credit_score):
    df = pd.read_excel("Company_Financials_FY2024.xlsx")
 
    if "Company" not in df.columns:
        print("Excel file must have a 'Company' column.")
        return None
 
    row = df[df['Company'] == ticker]
    if row.empty:
        print(f"Ticker {ticker} not found in Excel.")
        return None
 
    row = row.iloc[0]
 
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
    }
 
    data["LtC"] = safe_div(loan_value, collateral_value)
    return data
 
def rule_function(data):
    df1 = pd.read_excel("Company_Financials_Synthetic_First100.xlsx")
    df2 = pd.DataFrame([data])
    df = pd.concat([df1, df2], axis=0)
 
    df = df[[
        "Net Profit Margin %", "Return on Equity %", "Return on Assets %",
        "Current Ratio", "Asset Turnover Ratio", "Debt Equity Ratio", "Debt To Asset Ratio",
        "Interest Coverage Ratio", "Loan Value", "Collateral Value", "Credit Score"
    ]]
    
    df["LtC"] = df["Loan Value"] / df["Collateral Value"]
    df = df.drop(columns=["Loan Value", "Collateral Value"])
    df = df.fillna(df.median(numeric_only=True))
 
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(df)
    scaled_df = pd.DataFrame(scaled_data, columns=df.columns)
    scaled_df = 1 + (100 * scaled_df)
 
    # Apply weights
    dict_fin_weights = {
        "Net Profit Margin %": 0.25, "Return on Equity %": 0.25, "Return on Assets %": 0.25,
        "Current Ratio": 0.25, "Asset Turnover Ratio": 0.1, "Debt Equity Ratio": 0.1, "Debt To Asset Ratio": -0.2
    }
    dict_repay_weights = {
        "Interest Coverage Ratio": 0.20, "Credit Score": 0.65, "LtC": 0.15
    }
 
    for col, weight in {**dict_fin_weights, **dict_repay_weights}.items():
        scaled_df[col] = scaled_df[col] * weight
 
    scaled_df["Financial Risk Score"] = 100 - scaled_df[list(dict_fin_weights.keys())].sum(axis=1)
    scaled_df["Repayment Risk Score"] = 100 - scaled_df[list(dict_repay_weights.keys())].sum(axis=1)
    scaled_df["Final Risk Score"] = (
        scaled_df["Financial Risk Score"] * 0.3 +
        scaled_df["Repayment Risk Score"] * 0.7
    )
 
    scaled_df.to_csv("scaled_data.csv", index=False)
    return scaled_df.iloc[-1]["Final Risk Score"], df.iloc[-1]["LtC"]
 
def evaluate_company_risk(company_name, loan_value, collateral_value, credit_score):
    ticker = search_ticker_by_company_name(company_name)
    if not ticker:
        print(f"Ticker not found for company: {company_name}")
        return
 
    print(f"\nCompany: {company_name} | Ticker: {ticker}")
    data = fetch_financial_data_from_excel(ticker, loan_value, collateral_value, credit_score)
    if not data:
        return
 
    risk_score, ltc = rule_function(data)
    print(f"\nFinal Risk Score: {risk_score:.2f}")
    print(f"Loan-to-Collateral (LtC) Ratio: {ltc:.2f}")
    return risk_score, ltc
 
# -----------------------
# Main entry point
# -----------------------
if __name__ == "__main__":
    print("Enter Company Loan Evaluation Details:")
    company_name = input("Company Name: ")
    loan_value = float(input("Loan Amount: "))
    collateral_value = float(input("Collateral Amount: "))
    credit_score = float(input("Credit Score (0-1000): "))
 
    evaluate_company_risk(company_name, loan_value, collateral_value, credit_score)
 