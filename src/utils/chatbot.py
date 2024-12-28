import yaml
from typing import Tuple, List, Annotated

from typing_extensions import TypedDict

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_google_genai import ChatGoogleGenerativeAI

from utils.load_config import LoadConfig

APPCFG = LoadConfig()

with open("config/app_config.yml") as cfg:
    app_config = yaml.load(cfg, Loader=yaml.FullLoader)

class State(TypedDict):
    messages:Annotated[list,add_messages]

graph_builder = StateGraph(State)

llm = ChatGoogleGenerativeAI(
        model=APPCFG.llm_engine,
        temperature = 0
)

def chatbot(state: State):
    return {"messages":[llm.invoke(state["messages"])]}

graph_builder.add_node("chatbot", chatbot)
graph_builder.add_edge(START, "chatbot")
graph_builder.add_edge("chatbot", END)
graph = graph_builder.compile()

