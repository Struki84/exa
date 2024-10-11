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
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.prebuilt import ToolNode


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

    # 1. Extract factual claims from the query using LLM
    claims = extract_claims(text)

    # 2 & 3. Create search queries and perform web search for each claim
    exa_queries = []
    search_results = {}
    for claim in claims:
        query = create_search_query(claim)
        exa_queries.append(query)
        search_results[claim] = exa_search(query)

    # 4. Verify each claim as true or false based on search results
    verified_facts = []
    hallucinated_facts = []
    for claim, results in search_results.items():
        if verify_claim(claim, results):
            verified_facts.append(claim)
        else:
            hallucinated_facts.append(claim)

    # 5 & 6. Determine if the text is a hallucination and the confidence
    is_hallucination = len(hallucinated_facts) > 0
    confidence = determine_confidence(verified_facts, hallucinated_facts)

    # 7, 8, 9. Save the used search queries, verified facts, and hallucinated facts
    sources = list(set([result.split('\n')[0].split(': ')[1]
                        for results in search_results.values()
                        for result in results]))

    return {
        "is_hallucination": is_hallucination,
        "confidence": confidence,
        "exa_queries": exa_queries,
        "sources": sources,
        "verified_facts": verified_facts,
        "hallucinated_facts": hallucinated_facts
    }


def extract_claims(text: str) -> List[str]:
    """Extract factual claims from the text using an LLM."""
    system_message = SystemMessage(content="""
    You are an expert at extracting factual claims from text.
    Your task is to identify and list all factual claims present
     in the given text.
    Each claim should be a single, verifiable statement.
    Present the claims as a Python list of strings.
    """)

    human_message = HumanMessage(
        content=f"Extract factual claims from this text: {text}")
    response = llm.invoke([system_message, human_message])

    claims = eval(response.content)
    return claims


def create_search_query(claim: str) -> str:
    """Create a exa search query for the given claim."""
    # generate search strings using llm


def verify_claim(claim: str, results: List[str]) -> bool:
    """Verify if the claim is true or false based on the search results."""
    # based on search results verify is a calim is true or false


def determine_confidence(verified: List[str], hallucinated: List[str]) -> str:
    """Determine the confidence level of the hallucination assessment."""
    total_claims = len(verified) + len(hallucinated)
    if total_claims == 0:
        return "Low"

    hallucination_ratio = len(hallucinated) / total_claims
    if hallucination_ratio == 0:
        return "Certain"
    elif hallucination_ratio < 0.2:
        return "High"
    elif hallucination_ratio < 0.5:
        return "Medium"
    else:
        return "Low"


llm = ChatAnthropic(model="claude-3-5-sonnet-20240620", temperature=0)
llm.bind_tools([hallucination_check])
system_prompt = """
You are an expert on hallucinations.

Use the hallucination_check tool to detect hallucinations and generate an analysis.
Provide your final analysis in this format:

FINAL ANALYSIS
Is Hallucination: [Yes/No]
Confidence: [Low/Medium/High/Certain]
Exa Queries Used: [List of queries]
Sources: [List of relevant URLs]
Verified Facts: [List of facts that were verified]
Hallucinated Facts: [List of points that appear to be hallucinations, if any]

"""


def call_model(state: State):
    messages = state.messages
    response = llm.invoke(messages)
    return {"messages": state.messages + [response]}


def use_analysis(state: State) -> Literal["tools", "process_result"]:
    messages = state.messages
    last_message = messages[-1]
    # content = last_message.content

    # if isinstance(last_message, AIMessage) and "FINAL ANALYSIS" in content:
    #     return "process_result"

    # Limit to prevent infinite loops
    # maybe add a condition based on hallucionation lvls?
    # should we even loop this or just do it once,
    # as in the exa retreival example?
    # return "agent" if len(messages) < 4 else END

    return "tools" if last_message.tool_calls else "process_result"


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
workflow.add_node("tools", ToolNode(hallucination_check))
workflow.add_node("process_result", process_result)

workflow.set_entry_point("agent")
workflow.add_conditional_edges("agent", use_analysis, {
    "process_result": "process_result",
    "tools": "tools"
})
workflow.add_edge("tools", "process_result")
workflow.add_edge("process_result", END)

graph = workflow.compile(checkpointer=MemorySaver())

text = """The Eiffel Tower in Paris was originally constructed as a giant
 sundial in 1822, using the shadow cast by its iron lattice structure to tell
 time for the city's residents."""

initial_state = State(messages=[
    SystemMessage(content=system_prompt),
    HumanMessage(content=text)
])


final_state = graph.invoke(
    initial_state,
    config={"configurable": {"thread_id": 11}},
)

print(final_state["messages"][-1].content)

print("---")

print(final_state["analysis_result"])
