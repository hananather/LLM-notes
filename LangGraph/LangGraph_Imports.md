from typing import TypedDict, Annotated
from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages


from IPython.display import Image, display
from langgraph.graph import StateGraph, START, END


import os
from pprint import pprint
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_openai import AzureChatOpenAI



# invoking the LLM:
llm = AzureChatOpenAI(
    azure_deployment= os.environ["AZURE_OPENAI_DEPLOYMENT"],
    api_version= os.environ["AZURE_OPENAI_API_VERSION"],
)
result = llm.invoke(messages)