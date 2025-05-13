import os
import requests
import pandas as pd
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from sklearn.preprocessing import MinMaxScaler
from typing import Tuple, Dict, Optional, Any, Union
from langchain.tools import tool
 
# Load environment variables
load_dotenv()
 
# Azure OpenAI GPT-4o client setup (optional for this module)
model_name = os.environ.get("AZURE_OPENAI_MODEL_NAME")
endpoint_url = os.environ.get("AZURE_OPENAI_ENDPOINT")
api_key = os.environ.get("AZURE_OPENAI_API_KEY")
deployment_name = os.environ.get("AZURE_OPENAI_DEPLOYMENT")
version_number = os.environ.get("API_VERSION_GA")
 
class ScoreStructure(BaseModel):
    score: float = Field(..., description="Score for the loan application based on narrative and model.")
 
@tool
def search_ticker_by_company_name(company_name: str) -> Optional[str]:
    """Search for a company's stock ticker using Yahoo Finance."""
    url = f"https://query1.finance.yahoo.com/v1/finance/search?q={company_name}"
    headers = {"User-Agent": "Mozilla/5.0"}
    try:
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
    except Exception as e:
        print(f"Error searching for ticker: {e}")
        return None
 
def safe_div(numerator: Union[float, int], denominator: Union[float, int]) -> Optional[float]:
    try:
        if denominator == 0 or denominator is None:
            return None
        return numerator / denominator
    except:
        return None
 
@tool
def fetch_financial_data_from_excel(
    ticker: str, 
    loan_value: float, 
    collateral_value: float, 
    credit_score: float
) -> Optional[Dict[str, Any]]:
    """Fetch the financial data of a company from an Excel sheet using its ticker."""
    try:
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
    except Exception as e:
        print(f"Error fetching financial data: {e}")
        return None
 
@tool
def rule_function(data: Dict[str, Any]) -> Tuple[float, float]:
    """Apply rule-based scoring logic to calculate risk scores based on financial data."""
    try:
        df1 = pd.read_excel("Company_Financials_Synthetic_First100.xlsx")
        df2 = pd.DataFrame([data])
        df = pd.concat([df1, df2], axis=0)
 
        df = df[[
            "Net Profit Margin %", "Return on Equity %", "Return on Assets %",
            "Current Ratio", "Asset Turnover Ratio", "Debt Equity Ratio", "Debt To Asset Ratio",
            "Interest Coverage Ratio", "Loan Value", "Collateral Value", "Credit Score"
        ]]
        df["LtC"] = df.apply(lambda row: safe_div(row["Loan Value"], row["Collateral Value"]), axis=1)
        df = df.drop(columns=["Loan Value", "Collateral Value"])
        df = df.fillna(df.median(numeric_only=True))
 
        scaler = MinMaxScaler()
        scaled_data = scaler.fit_transform(df)
        scaled_df = pd.DataFrame(scaled_data, columns=df.columns)
        scaled_df = 1 + (100 * scaled_df)
 
        dict_fin_weights = {
            "Net Profit Margin %": 0.25, 
            "Return on Equity %": 0.25, 
            "Return on Assets %": 0.25,
            "Current Ratio": 0.25, 
            "Asset Turnover Ratio": 0.1, 
            "Debt Equity Ratio": 0.1, 
            "Debt To Asset Ratio": -0.2
        }
        dict_repay_weights = {
            "Interest Coverage Ratio": 0.20, 
            "Credit Score": 0.65, 
            "LtC": 0.15
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
    except Exception as e:
        print(f"Error in risk calculation: {e}")
        return 0.0, 0.0
 
@tool
def evaluate_company_risk(
    company_name: str, 
    loan_value: float, 
    collateral_value: float, 
    credit_score: float
) -> Optional[Tuple[float, float]]:
    """Evaluate a company's risk score based on financial data and loan parameters."""
    ticker = search_ticker_by_company_name(company_name)
    if not ticker:
        print(f"Ticker not found for company: {company_name}")
        return None
    print(f"\nCompany: {company_name} | Ticker: {ticker}")
    data = fetch_financial_data_from_excel(ticker, loan_value, collateral_value, credit_score)
    if not data:
        return None
    risk_score, ltc = rule_function(data)
    print(f"\nFinal Risk Score: {risk_score:.2f}")
    print(f"Loan-to-Collateral (LtC) Ratio: {ltc:.2f}")
    return risk_score, ltc
 
# Agent class
class RiskAssessmentAgent:
    def __init__(self):
        pass
 
    def assess_loan_risk(
        self, 
        company_name: str, 
        loan_amount: float, 
        collateral_amount: float, 
        credit_score: float
    ) -> Dict[str, Any]:
        if not isinstance(company_name, str) or not company_name.strip():
            return {"error": "Company name must be a non-empty string"}
        if loan_amount <= 0 or collateral_amount <= 0:
            return {"error": "Loan and collateral values must be positive"}
        if credit_score < 300 or credit_score > 900:
            return {"error": "Credit score must be between 300 and 900"}
 
        ticker = search_ticker_by_company_name(company_name)
        if not ticker:
            return {"error": f"Ticker not found for company: {company_name}"}
 
        data = fetch_financial_data_from_excel(ticker, loan_amount, collateral_amount, credit_score)
        if not data:
            return {"error": "Financial data not found or invalid"}
 
        risk_score, ltc = rule_function(data)
        return {
            "company": company_name,
            "ticker": ticker,
            "risk_score": round(risk_score, 2),
            "ltc_ratio": round(ltc, 2) if ltc is not None else None
        }
if __name__ == "__main__":
    print("\n📊 Company Loan Risk Assessment Tool 📊")
    try:
        company_name = input("Enter the company name: ").strip()
        loan_amount = float(input("Enter the loan amount: "))
        collateral_amount = float(input("Enter the collateral amount: "))
        credit_score = float(input("Enter the credit score (300 to 900): "))
 
        agent = RiskAssessmentAgent()
        result = agent.assess_loan_risk(
            company_name=company_name,
            loan_amount=loan_amount,
            collateral_amount=collateral_amount,
            credit_score=credit_score
        )
 
        print("\n🔍 Risk Assessment Result:")
        for key, value in result.items():
            print(f"{key.capitalize().replace('_', ' ')}: {value}")
 
    except ValueError:
        print("❌ Invalid input. Please enter numeric values for loan, collateral, and credit score.")
    except Exception as e:
        print(f"❌ An error occurred: {e}")
 