from utils.chatbot import graph
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage


def stream_graph_updates(user_input: str, thread_id: str):
    config = {"configurable":{"thread_id": thread_id}}
    events = graph.stream(
         {"messages": [("user", user_input)]}, config, stream_mode="values"
    )
    for event in events:
        last_message = event["messages"][-1]
        if isinstance(last_message, AIMessage):
            print(event["messages"][-1].content)

def main():
    while True:
        try:
            user_input = input("User: ")
            if user_input.lower() in ["quit", "exit", "q"]:
                print("Goodbye!")
                break

            stream_graph_updates(user_input, "1")
        except Exception as e:
            print(f"Error:{e}")
            break

if __name__ == "__main__":
    main()
