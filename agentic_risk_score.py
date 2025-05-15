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
