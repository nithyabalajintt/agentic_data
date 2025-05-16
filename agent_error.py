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
        response = agent.print_response(url, debug=True)
        print("Agent response:", response)
        return response
 
    except Exception as e:
        print(f"Error during fetch_company_ratios: {e}")
        return {}



Agent response: None
