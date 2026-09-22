from tavily import TavilyClient
import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq

load_dotenv()

llm = ChatGroq(model="openai/gpt-oss-120b", api_key=os.environ["GROQ_API_KEY"])

tavily = TavilyClient(api_key=os.environ["TAVILY_API_KEY"])

def web_search(query: str) -> str:
    """Searches the web and returns a summarised set of results"""
    results = tavily.search(query=query, search_depth="basic", max_results=3)
    
    formatted = []
    for r in results["results"]:
        formatted.append(f"Source: {r["title"]} ({r['url']})\n{r['content']}")
        
    return "\n\n".join(formatted)

if __name__ == "__main__":
    result = web_search("current AWS pricing for small startups 2026")
    print(result)