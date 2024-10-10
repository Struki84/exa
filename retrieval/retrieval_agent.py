from typing import List, Literal
from dotenv import load_dotenv
from rich.console import Console
from rich.prompt import Prompt
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableLambda
from langchain_core.tools import tool
from langchain_exa import ExaSearchRetriever
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode

load_dotenv()


@tool
def retrieve_web_content(query: str) -> List[str]:
    """Function to retrieve usable documents for AI assistant"""

    retriever = ExaSearchRetriever(k=3, highlights=True, use_autoprompt=True)

    # Define how to extract relevant metadata from the search results
    document_prompt = PromptTemplate.from_template(
        """
        <source>
            <url>{url}</url>
            <highlights>{highlights}</highlights>
        </source>
        """
    )

    # Define how to parse the retrieved documents
    parse_info = RunnableLambda(
        lambda document: {
            "url": document.metadata["url"],
            "highlights": document.metadata.get("highlights", "No highlights"),
        }
    )

    # Create a chain to process the retrieved documents and format it using the prompt
    document_chain = (parse_info | document_prompt)

    # Execute the retrieval and process the results in the document chain
    retrieval_chain = retriever | document_chain.map()

    # Retrieve and return the documents
    documents = retrieval_chain.invoke(query)
    return documents


# Create the model and add the retrieval tool
model = ChatAnthropic(model="claude-3-5-sonnet-20240620",
                      temperature=0).bind_tools([retrieve_web_content])

# Determine whether to continue or end


def should_continue(state: MessagesState) -> Literal["tools", END]:
    messages = state["messages"]
    last_message = messages[-1]
    return "tools" if last_message.tool_calls else END

# Function to generate model responses


def call_model(state: MessagesState):
    messages = state["messages"]
    response = model.invoke(messages)
    return {"messages": [response]}


console = Console()


def main():

    workflow = StateGraph(MessagesState)
    workflow.add_node("agent", call_model)
    workflow.add_node("tools", ToolNode([retrieve_web_content]))
    workflow.set_entry_point("agent")
    workflow.add_conditional_edges("agent", should_continue)
    workflow.add_edge("tools", "agent")

    # Initialize memory
    checkpointer = MemorySaver()

    # Compile the workflow into a runnable
    app = workflow.compile(checkpointer=checkpointer)

    messages = []
    thread_id = 0

    while True:
        try:

            user_query = Prompt.ask(
                "[bold yellow]What do you want to search for?[/bold yellow]",
            )

            initial_state = {
                "messages": [HumanMessage(content=user_query)]
            }

            final_state = app.invoke(
                initial_state,
                config={"configurable": {"thread_id": thread_id}},
            )

            ai_response = final_state["messages"][-1]

            if isinstance(ai_response, AIMessage):
                Console.print(ai_response.content)

            messages = final_state["messages"]

            thread_id += 1

        except KeyboardInterrupt:
            console.print("[bold red]Exiting the program.[/bold red]")
            break
        except Exception as e:
            console.print(f"[bold red]An error occurred:[/bold red] {str(e)}")


if __name__ == "__main__":
    main()
