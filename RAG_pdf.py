import os
from langchain.vectorstores.chroma import Chroma 
from langchain_openai import AzureOpenAIEmbeddings
from openai import AzureOpenAI
from dotenv import load_dotenv
 
CHROMA_PATH = "chroma_finance_docs"
load_dotenv()
 
# GPT config
model_name = os.environ.get("AZURE_OPENAI_MODEL_NAME")
endpoint_url = os.environ.get("AZURE_OPENAI_ENDPOINT")
api_key = os.environ.get("AZURE_OPENAI_API_KEY")
deployment_name = os.environ.get("AZURE_OPENAI_DEPLOYMENT")
version_number = os.environ.get("API_VERSION_GA")
 
embedding_model_name = os.environ.get("ADA_AZURE_OPENAI_MODEL_NAME")
embedding_endpoint_url = os.environ.get("ADA_AZURE_OPENAI_ENDPOINT")
embedding_api_key = os.environ.get("ADA_AZURE_OPENAI_API_KEY")
embedding_deployment_name = os.environ.get("ADA_AZURE_OPENAI_DEPLOYMENT")
 
embedder = AzureOpenAIEmbeddings(
    model=embedding_model_name,
    api_key=embedding_api_key,
    azure_deployment=embedding_deployment_name,
    azure_endpoint=embedding_endpoint_url,
    api_version="2024-10-21"
)
 
def query_rag(query_text):
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedder)
    results = db.similarity_search_with_relevance_scores(query_text, k=3)
 
    if len(results) == 0 or results[0][1] < 0.7:
        return "Unable to find matching results.", ""
 
    context_text = "\n\n - -\n\n".join([doc.page_content for doc, _ in results])
 
    client = AzureOpenAI(
        azure_endpoint=endpoint_url,
        azure_deployment=deployment_name,
        api_key=api_key,
        api_version=version_number
    )
 
    rag_chat_completion = client.chat.completions.create(
        model=model_name,
        messages=[
            {
                "role": "system",
                "content": """You are a Retrieval Augmentation Chatbot that only replies using the context provided.
If the answer is not found, say: 'I cannot answer that based on the context provided.'"""
            },
            {
                "role": "user",
                "content": f"VECTOR DATABASE CONTEXT:\n{context_text}\n\nUSER QUERY: {query_text}"
            }
        ]
    )
 
    response_text = rag_chat_completion.choices[0].message.content
    usage = rag_chat_completion.usage
    sources = [doc.metadata.get("source", None) for doc, _ in results]
 
    formatted_response = f"Response: {response_text}\nSources: {sources}\nUsage: {usage}"
    return formatted_response, response_text
 
# Test
query_text = "What are the benefits of perksplus"
formatted_response, response_text = query_rag(query_text)
print(formatted_response)
 