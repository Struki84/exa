from typing import List,  Dict, Any, Literal
from pydantic import BaseModel

from langchain_anthropic import ChatAnthropic
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.runnables import RunnableLambda
from langchain_core.prompts import PromptTemplate
from langchain_core.tools import tool
from langchain_exa import ExaSearchRetriever
from typing import Annotated
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages
from langchain_core.messages import HumanMessage, AIMessage


class State(BaseModel):
    messages: Annotated[list, add_messages]
    analysis_result: Dict[str, Any] = {}


class AnalysisResult(BaseModel):
    is_hallucination: bool
    confidence: str  # "Low", "Medium", "High", "Certain"
    exa_queries: List[str]
    sources: List[str]
    verified_facts: List[str]
    hallucinated_facts: List[str]


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


@tool
def hallucination_check(text: str):
    """Assess the given text for hallucinations using Exa search.
     1. extract factual claims from the query
     2. for each of the claims, create a search query
     3. perform web search for each of the search queries
     4. verify each of the claims as true or false based on search results
     5. determine if the text is hallucination
     6. determine the confidence of the hallucination assumption
     7. save the used search queries
     8. save the verified facts
     9. save the hallucinated facts
      """


llm = ChatAnthropic(model="claude-3-5-sonnet-20240620", temperature=0)
llm.bind_tools([hallucination_check])


def call_model(state: State):
    messages = state.messages
    response = llm.invoke(messages)
    return {"messages": state.messages + [response]}


def stop_analysis(state: State) -> Literal["agent", "process_result", END]:
    messages = state.messages
    last_message = messages[-1]
    content = last_message.content

    if isinstance(last_message, AIMessage) and "FINAL ANALYSIS" in content:
        return "process_result"

    # Limit to prevent infinite loops
    return "agent" if len(messages) < 4 else END


def process_result(state: State):
    last_message = state.messages[-1].content
    lines = last_message.split("\n")
    result = AnalysisResult(
        is_hallucination="Yes" in lines[1],
        confidence=lines[2].split(": ")[1],
        exa_queries=eval(lines[3].split(": ")[1]),
        sources=eval(lines[4].split(": ")[1]),
        verified_facts=eval(lines[5].split(": ")[1]),
        hallucinated_points=eval(lines[6].split(": ")[1])
    )
    return {"analysis_result": result.dict()}


workflow = StateGraph(State)

workflow.add_node("agent", call_model)
workflow.add_node("process_result", process_result)

workflow.set_entry_point("agent")
workflow.add_conditional_edges(
    "agent",
    stop_analysis,
    {
        "agent": "agent",
        "process_result": "process_result",
        END: END
    }
)
workflow.add_edge("process_result", END)

graph = workflow.compile(checkpointer=MemorySaver())
