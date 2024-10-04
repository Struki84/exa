# import all required packages
import os
import anthropic

from dotenv import load_dotenv
from typing import Any, Dict
from exa_py import Exa
from rich.console import Console
from rich.markdown import Markdown
from rich.prompt import Prompt

# Load environment variables from .env file
load_dotenv()

# create the anthropic client
claude = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

# create the exa client
exa = Exa(api_key=os.getenv("EXA_API_KEY"))

# create the rich console
console = Console()

# define the system message (primer) of your agent
SYSTEM_MESSAGE = "You are an agent that has access to an advanced search engine. Please provide the user with the information they are looking for by using the search tool provided."

# define the tools available to the agent - we're defining a single tool, exa_search
TOOLS = [
    {
        "name": "exa_search",
        "description": "Perform a search query on the web, and retrieve the most relevant URLs/web data.",
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The search query to perform.",
                },
            },
            "required": ["query"],
        },
    }
]

# define the function that will be called when the tool is used and perform the search
# and the retrieval of the result highlights.
# https://docs.exa.ai/reference/python-sdk-specification#search_and_contents-method
def exa_search(query: str) -> Dict[str, Any]:
    return exa.search_and_contents(query=query, type='auto', highlights=True)

# define the function that will process the tool call and perform the exa search
def process_tool_calls(tool_calls):
    search_results = []
    
    for tool_call in tool_calls:
        function_name = tool_call.name
        function_args = tool_call.input
        
        if function_name == "exa_search":
            results = exa_search(**function_args)
            search_results.append(results)
            
            console.print(
                f"[bold cyan]Context updated[/bold cyan] [i]with[/i] "
                f"[bold green]exa_search[/bold green]: ",
                function_args.get("query"),
            )
            
    return search_results


def main():
    messages = []
    
    while True:
        try:
            # create the user input prompt using rich
            user_query = Prompt.ask(
                "[bold yellow]What do you want to search for?[/bold yellow]",
            )
            messages.append({"role": "user", "content": user_query})
            
            # call Claude llm by creating a completion which calls the defined exa tool
            completion = claude.messages.create(
                model="claude-3-sonnet-20240229",
                max_tokens=1024,
                system=SYSTEM_MESSAGE,
                messages=messages,
                tools=TOOLS,
            )
            
            # completion will contain the object needed to invoke your tool and perform the search
            message = completion.content[0]
            tool_calls = [content for content in completion.content if content.type == "tool_use"]
            
            if tool_calls:
                
                # process the tool object created by Claude llm and store the search results
                search_results = process_tool_calls(tool_calls)
                
                # create new message containing the search results and request the Claude llm to process the results
                messages.append({"role": "assistant", "content": f"I've performed a search and found the following results: {search_results}"})
                messages.append({"role": "user", "content": "Please summarize this information and answer my previous query based on these results."})
                
                # call Claude llm again to process the search results and yield the final answer
                completion = claude.messages.create(
                    model="claude-3-sonnet-20240229",
                    max_tokens=1024,
                    system=SYSTEM_MESSAGE,
                    messages=messages,
                )
                
                # parse the agents final answer and print it
                response = completion.content[0].text
                console.print(Markdown(response))
                messages.append({"role": "assistant", "content": response})

            else:
                # in case tool hasn't been used, print the standard agent response
                console.print(Markdown(message.text))
                messages.append({"role": "assistant", "content": message.text})
                
        except Exception as e:
            console.print(f"[bold red]An error occurred:[/bold red] {str(e)}")
            
if __name__ == "__main__":
    main()
