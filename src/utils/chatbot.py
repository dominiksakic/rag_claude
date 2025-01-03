import yaml
from typing import Tuple, List, Annotated

from typing_extensions import TypedDict

from langchain_chroma import Chroma
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_google_genai import ChatGoogleGenerativeAI

from utils.load_config import LoadConfig

APPCFG = LoadConfig()

with open("config/app_config.yml") as cfg:
    app_config = yaml.load(cfg, Loader=yaml.FullLoader)

# Load the embedding function
embeddings = APPCFG.embedding_model

# Load the DB
vectordb = Chroma(persist_directory=APPCFG.persist_directory,
                  embedding_function=embeddings)

class State(TypedDict):
    messages:Annotated[list,add_messages]
    retrieved_docs: List[str]

graph_builder = StateGraph(State)

llm = ChatGoogleGenerativeAI(
        model=APPCFG.llm_engine,
        temperature = 0
)

def retrieve_documents(state: State):
    last_message= state["messages"][-1]
    question = last_message.content
    question = "# user new question:\n" + question
    docs = vectordb.similarity_search(question, k=APPCFG.k)
    retrieved_docs = [str(doc.page_content) for doc in docs]
    return {"retrieved_docs":retrieved_docs}

def chatbot(state: State):
    return {"messages": [llm.invoke(state["messages"])]}

memory = MemorySaver() # in Development, for production research sql memory

graph_builder.add_node("retrieve_documents", retrieve_documents)
graph_builder.add_node("chatbot", chatbot)
graph_builder.add_edge(START, "retrieve_documents")
graph_builder.add_edge("retrieve_documents", "chatbot")
graph_builder.add_edge("chatbot", END)
graph = graph_builder.compile(checkpointer =memory)

