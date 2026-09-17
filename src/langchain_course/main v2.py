from pathlib import Path
import os

from pydantic import BaseModel, Field
from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.tools import tool
from langchain_core.messages import HumanMessage
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI
from tavily import TavilyClient
from langchain_tavily import TavilySearch

class Source(BaseModel):
    """Schema for a source used by the agent."""
    #name: str = Field(description="The name of the source")
    url: str = Field(description="The url of the source")

class AgentResponse(BaseModel):
    """Schema for the response with an answer and sources."""
    
    answer: str = Field(description="The agent's answer to the query")
    sources: list[Source] = Field(default_factory=list, description="List of sources used to answer the query")

#tavily = TavilyClient()

# Project root .env (not .venv/.env). Path is based on this file so CWD does not matter.
load_dotenv(Path(__file__).resolve().parents[2] / ".env")

'''
@tool
def search(query: str) -> str:
    """
    Tool to search the web for information.
    Args:
        query: The query to search for.
    Returns:
        The search results.
    """
    print(f"Search results for {query}")
    #return "Tokyo weather is sunny"
    return tavily.search(query=query)
'''

# gemma3:270m does not support tools; llama3.2 does
llm = ChatOllama(model="llama3.2:3b", temperature=0)
#tools = [search]
tools = [TavilySearch()]
agent = create_agent(model=llm, tools=tools, response_format=AgentResponse)

def main() -> None:
    print("Hello from langchain-course!")
    #result = agent.invoke({"messages": [HumanMessage(content="What is the weather in Tokyo?")]})
    result = agent.invoke({"messages": [HumanMessage(content="Search 3 jobs for AI engieers for remote location even I can work from India?")]})
    print(result)

if __name__ == "__main__":
    main()
