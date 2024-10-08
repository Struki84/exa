import asyncio
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_anthropic import ChatAnthropic
from typing import Annotated, List, Sequence
from langgraph.graph import END, StateGraph, START
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver
from typing_extensions import TypedDict

prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are an essay assistant tasked with writing excellent 5-paragraph essays."
            " Generate the best essay possible for the user's request."
            " If the user provides critique, respond with a revised version of your previous attempts.",
        ),
        MessagesPlaceholder(variable_name="messages"),
    ]
)
llm = ChatAnthropic(model="claude-3-5-sonnet-20240620", temperature=0)
generate = prompt | llm

request = HumanMessage(content="What do you want to know about the universe?")

reflection_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a teacher grading an essay submission. Generate critique and recommendations for the user's submission."
            " Provide detailed recommendations, including requests for length, depth, style, etc.",
        ),
        MessagesPlaceholder(variable_name="messages"),
    ]
)
reflect = reflection_prompt | llm


class State(TypedDict):
    messages: Annotated[list, add_messages]


async def generation_node(state: State) -> State:
    return {"messages": [await generate.ainvoke(state["messages"])]}


async def reflection_node(state: State) -> State:
    # Other messages we need to adjust
    cls_map = {"ai": HumanMessage, "human": AIMessage}
    # First message is the original user request. We hold it the same for all nodes
    translated = [state["messages"][0]] + [
        cls_map[msg.type](content=msg.content) for msg in state["messages"][1:]
    ]
    res = await reflect.ainvoke(translated)
    # We treat the output of this as human feedback for the generator
    return {"messages": [HumanMessage(content=res.content)]}


def should_continue(state: State):
    if len(state["messages"]) > 4:
        return END
    return "reflect"


checkpoint = MemorySaver()

workflow = StateGraph(State)
workflow.add_node("generate", generation_node)
workflow.add_node("reflect", reflection_node)
workflow.add_edge(START, "generate")
workflow.add_conditional_edges("generate", should_continue)
workflow.add_edge("reflect", "generate")

graph = workflow.compile(checkpointer=checkpoint)

config = {"configurable": {"thread_id": 1}}


async def process_events():
    async for event in graph.astream({"messages": request}, config=config):
        print(event)
        print("---")


asyncio.run(process_events())
