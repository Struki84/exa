from typing import List, Literal, Dict, Any
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda
from langchain_core.tools import tool
from langchain_exa import ExaSearchRetriever
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph
from pydantic import BaseModel

# Define the state


class HallucinationCheckState(BaseModel):
    messages: List[Any]
    hallucination_result: Dict[str, Any] = {}


# Define the output structure
class HallucinationCheckResult(BaseModel):
    is_hallucination: bool
    confidence: str  # "Low", "Medium", "High", "Certain"
    exa_queries: List[str]
    sources: List[str]
    verified_facts: List[str]
    hallucinated_points: List[str]


@tool
def retrieve_web_content(query: str) -> List[str]:
    """Function to retrieve web content for fact-checking"""
    retriever = ExaSearchRetriever(k=3, highlights=True, use_autoprompt=True)
    document_prompt = ChatPromptTemplate.from_messages([
        ("human", "Source information:\nURL: {url}\nHighlights: {highlights}")
    ])
    document_chain = (
        RunnableLambda(lambda document: {
            "highlights": document.metadata.get("highlights", "No highlights"),
            "url": document.metadata["url"],
        })
        | document_prompt
    )
    retrieval_chain = retriever | document_chain.map()
    documents = retrieval_chain.invoke(query)
    return [str(doc) for doc in documents]


# Define and bind the AI model
model = ChatAnthropic(model="claude-3-5-sonnet-20240620", temperature=0).bind_tools(
    [retrieve_web_content]
)

# Determine whether to continue or end


def should_continue(state: HallucinationCheckState) -> Literal["agent", "process_result", END]:
    messages = state.messages
    last_message = messages[-1]
    if isinstance(last_message, AIMessage) and "FINAL ANALYSIS" in last_message.content:
        return "process_result"
    # Limit to prevent infinite loops
    return "agent" if len(messages) < 4 else END

# Function to generate model responses


def call_model(state: HallucinationCheckState):
    messages = state.messages
    response = model.invoke(messages)
    return {"messages": state.messages + [response]}

# Function to process the final result


def process_result(state: HallucinationCheckState):
    last_message = state.messages[-1].content
    lines = last_message.split("\n")
    result = HallucinationCheckResult(
        is_hallucination="Yes" in lines[1],
        confidence=lines[2].split(": ")[1],
        exa_queries=eval(lines[3].split(": ")[1]),
        sources=eval(lines[4].split(": ")[1]),
        verified_facts=eval(lines[5].split(": ")[1]),
        hallucinated_points=eval(lines[6].split(": ")[1])
    )
    return {"hallucination_result": result.dict()}


# Define the workflow graph
def create_hallucination_check_graph():
    workflow = StateGraph(HallucinationCheckState)
    workflow.add_node("agent", call_model)
    workflow.add_node("process_result", process_result)
    workflow.set_entry_point("agent")
    workflow.add_conditional_edges(
        "agent",
        should_continue,
        {
            "agent": "agent",
            "process_result": "process_result",
            END: END
        }
    )
    workflow.add_edge("process_result", END)
    return workflow.compile()


# Main function to check for hallucinations
def check_hallucination(text: str) -> HallucinationCheckResult:
    # Initialize memory checkpointer
    checkpointer = MemorySaver()

    # Create the graph
    app = create_hallucination_check_graph()

    # Run the graph
    initial_state = HallucinationCheckState(
        messages=[
            HumanMessage(content=f"""
            Analyze the following text for hallucinations:

            {text}

            Use the retrieve_web_content tool to verify claims. Provide your final analysis in this format:

            FINAL ANALYSIS
            Is Hallucination: [Yes/No]
            Confidence: [Low/Medium/High/Certain]
            Exa Queries Used: [List of queries]
            Sources: [List of relevant URLs]
            Verified Facts: [List of facts that were verified]
            Hallucinated Points: [List of points that appear to be hallucinations, if any]
            """)
        ]
    )

    final_state = app.invoke(
        initial_state,
        # You can use a unique identifier here
        config={"configurable": {"thread_id": 1}},
    )

    return HallucinationCheckResult(**final_state["hallucination_result"])


# Example usage
if __name__ == "__main__":
    sample_text = "The Eiffel Tower was built in 1887 and is located in Rome, Italy. It stands at a height of 324 meters."
    result = check_hallucination(sample_text)
    print(result)
