import asyncio

from langchain_anthropic import ChatAnthropic
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.runnables import RunnableLambda
from langchain_core.prompts import PromptTemplate
from typing import List
from langchain_core.tools import tool
from langchain_exa import ExaSearchRetriever
from typing import Annotated
from langgraph.graph import END, StateGraph, START
from langgraph.graph.message import add_messages
from typing_extensions import TypedDict
from langchain_core.messages import HumanMessage

llm = ChatAnthropic(model="claude-3-5-sonnet-20240620", temperature=0)

generator_prompt = PromptTemplate.from_messages(

)

generator = generator_prompt | llm

reflection_prompt = PromptTemplate.from_messages(

)


@tool
def exa_search(query: str) -> List[str]:
    """Function to retrieve usable documents for AI assistant"""

    search = ExaSearchRetriever(k=3, highlights=True, use_autoprompt=True)

    document_prompt = PromptTemplate.from_template(
        """
        <source>
            <url>{url}</url>
            <highlights>{highlights}</highlights>
        </source>
        """
    )

    parse_info = RunnableLambda(
        lambda document: {
            "url": document.metadata["url"],
            "highlights": document.metadata.get("highlights", "No highlights"),
        }
    )

    document_chain = (parse_info | document_prompt)

    search_chain = search | document_chain.map()

    documents = search_chain.invoke(query)
    return documents


reflection = reflection_prompt | llm.bind_tools([exa_search])


class State(TypedDict):
    messages: Annotated[list, add_messages]


async def generation_node(state: State) -> State:
    response = await generator.ainvoke(state["messages"])
    return {"messages": [response]}


async def reflection_node(state: State) -> State:
    response = await reflection.ainvoke(state["messages"])
    return {"messages": [response]}


def stop_reflection(state: State):
    if len(state["messages"]) > 4:
        return "generate"
    return "reflect"


workflow = StateGraph(State)
workflow.add_node("generate", generation_node)
workflow.add_node("reflect", reflection_node)


workflow.add_edge(START, "generate")
workflow.add_edge("generate", "reflect")
workflow.add_conditional_edges("reflect", stop_reflection)
workflow.add_edge("generate", END)

graph = workflow.compile(checkpointer=MemorySaver())

query = HumanMessage(content="When did Napoleon die?")


async def process_events():
    async for event in graph.astream({"messages": query}):
        print(event)
        print("---")


asyncio.run(process_events())
