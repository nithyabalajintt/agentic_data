from agno.tools.yfinance import YFinanceTools 
from agno.agent import Agent, RunResponse
from openai import AzureOpenAI
from pydantic import BaseModel
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import os
 
# ---------------------------
# Azure OpenAI Credentials
# ---------------------------
model_name = os.environ.get("AZURE_OPENAI_MODEL_NAME")
endpoint_url = os.environ.get("AZURE_OPENAI_ENDPOINT")
api_key = os.environ.get("AZURE_OPENAI_API_KEY")
deployment_name = os.environ.get("AZURE_OPENAI_DEPLOYMENT")
version_number = os.environ.get("API_VERSION_GA")
 
# ---------------------------
# Risk Evaluation Tool
# ---------------------------
def evaluate_loan_risk(company_name: str, loan_value: float, collateral_value: float, credit_score: float) -> str:
    print("Fetching key financial ratios...")
 
    ytools = YFinanceTools(key_financial_ratios=True)
    ratios = ytools.get_key_financial_ratios(ticker=company_name)
 
    if "error" in ratios:
        return {"error": f"Could not fetch ratios for {company_name}"}
 
    try:
        data = {
            "Net Profit Margin %": ratios.get("Net Profit Margin %"),
            "Return on Equity %": ratios.get("Return on Equity %"),
            "Return on Assets %": ratios.get("Return on Assets %"),
            "Current Ratio": ratios.get("Current Ratio"),
            "Asset Turnover Ratio": ratios.get("Asset Turnover Ratio"),
            "Debt Equity Ratio": ratios.get("Debt Equity Ratio"),
            "Debt To Asset Ratio": ratios.get("Debt To Asset Ratio"),
            "Interest Coverage Ratio": ratios.get("Interest Coverage Ratio"),
            "Loan Value": loan_value,
            "Collateral Value": collateral_value,
            "Credit Score": credit_score,
            "LtC": loan_value / collateral_value if collateral_value != 0 else 0
        }
 
        print(f"\n Data for risk analysis:\n{data}")
 
        # ---------------------------
        # Risk Score Computation
        # ---------------------------
        try:
            df1 = pd.read_excel("Company_Financials_Synthetic_First100.xlsx")
        except FileNotFoundError:
            return {"error": "Synthetic data file missing!"}
 
        df2 = pd.DataFrame([data])
        df = pd.concat([df1, df2], axis=0)
 
        df = df[[
            "Net Profit Margin %", "Return on Equity %", "Return on Assets %",
            "Current Ratio", "Asset Turnover Ratio", "Debt Equity Ratio", "Debt To Asset Ratio",
            "Interest Coverage Ratio", "Loan Value", "Collateral Value", "Credit Score",
            "LtC"
        ]]
 
        df = df.drop(columns=["Loan Value", "Collateral Value"])
        df = df.fillna(df.median(numeric_only=True))
 
        scaler = MinMaxScaler()
        scaled_data = scaler.fit_transform(df)
        scaled_df = pd.DataFrame(scaled_data, columns=df.columns)
        scaled_df = 1 + (100 * scaled_df)
 
        # Apply Weights
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
 
        final_score = scaled_df.iloc[-1]["Final Risk Score"]
        ltc_ratio = df.iloc[-1]["LtC"]
 
        return f"\nFinal Risk Score: {final_score:.2f}\nLoan-to-Collateral Ratio: {ltc_ratio:.2f}"
 
    except Exception as e:
        return {"error": str(e)}
 
# ---------------------------
# Agent Wrapper Function
# ---------------------------
def evaluate_company(
    company_name: str,
    loan_value: float,
    collateral_value: float,
    credit_score: float
) -> str:
    return evaluate_loan_risk(company_name, loan_value, collateral_value, credit_score)
 
# ---------------------------
# Create Agent
# ---------------------------
risk_score_agent = Agent(
    model=AzureOpenAI(
        api_key=api_key,
        azure_endpoint=endpoint_url,
        azure_deployment=deployment_name,
        api_version="2024-110-21",        

    ),
    description="Agent that calculates company financial and repayment risk score.",
    instructions="You will receive a company name, loan amount, collateral value, and credit score. Use the tool to calculate the risk score.",
    tools=[evaluate_company],
)
 
# ---------------------------
# CLI Input for Local Execution
# ---------------------------
if __name__ == "__main__":
    print("\nWelcome to the Loan Risk Score Agent\n")
 
    company = input("Enter company name: ").strip()
    loan = float(input("Enter loan value: ").strip())
    collateral = float(input("Enter collateral value: ").strip())
    score = float(input("Enter credit score: ").strip())
 
    print("\nEvaluating...")
    result = evaluate_company(company, loan, collateral, score)
    print(result)
 