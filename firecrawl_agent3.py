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
agent = Agent(
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
 
agent.print_response("https://www.investing.com/equities/kothari-products-ltd-ratios", debug=True)
 
 