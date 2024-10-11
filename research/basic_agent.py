from langchain_anthropic import ChatAnthropic
from langchain_core import MessagesState
from langgraph.graph import END, StateGraph
from langgraph.checkpoint.memory import MemorySaver

llm = ChatAnthropic(model="claude-3-5-sonnet-20240620", temperature=0)


def call_model(state: MessagesState):
    messages = state.messages
    response = llm.invoke(messages)
    return {"messages": state.messages + [response]}


workflow = StateGraph(MessagesState)
workflow.add_node("agent", call_model)

workflow.set_entry_point("agent")
workflow.add_edge("agent", END)

graph = workflow.compile(checkpointer=MemorySaver())


def main():

    user_query = input("Ask a question: ")

    initial_state = {
        "messages": [HumanMessage(content=user_query)]
    }

    final_state = graph.invoke(initial_state)

    ai_response = final_state["messages"][-1]

    print(ai_response.content)
