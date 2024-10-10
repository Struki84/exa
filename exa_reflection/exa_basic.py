from langchain_anthropic import ChatAnthropic
from langchain_core.runnables import RunnableLambda
from langchain_core.prompts import PromptTemplate
from typing import List, Literal
from langchain_core.tools import tool
from langchain_exa import ExaSearchRetriever
from typing import Annotated
from langgraph.graph import END, StateGraph, START
from langgraph.graph.message import add_messages
from typing_extensions import TypedDict

llm = ChatAnthropic(model="claude-3-5-sonnet-20240620", temperature=0)

generator_prompt = ChatPromptTemplate.from_messages(

)

generator = generator_prompt | llm

reflection_prompt = ChatPromptTemplate.from_messages(

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
