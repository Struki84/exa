# Using Claude's Tool Use Feature with Exa Search Integration
This guide will show you how to use Claude's tool use feature with an Exa search integration.
## What this doc covers
Explain Claude's tool use feature
Show you how to use Exa within the tool call
## Guide
### 1. Pre-requisites and installation
Install the Anthropic and Exa libraries
```python
pip install anthropic exa_py rich
```
### 2. What is Claude tool use?
Claude's tool use feature returns an object with a string, which is the function name defined in _your_ code, and the arguments that the function takes. This does not execute or _call_ functions on Anthropic's side; it only returns the function name and arguments which you will have to parse and call yourself in your code.
For example:
```python
{
  "type": "tool_use",
  "id": "toolu_01A09q90qw90lq917835123",
  "name": "exa_search",
  "input": {"query": "Latest developments in quantum computing"}
}
```
We will use this object to call the `exa_search` function we define with the arguments provided.
### 3. Use Exa Search in your tools
We need to import and initialize the Anthropic and Exa libraries. We'll also use components from the `rich` library to make the output more readable, and import some types to make the code more readable and extensible.
```python
import os
from typing import Any, Dict
import anthropic
from exa_py import Exa
from rich.console import Console
from rich.markdown import Markdown
from rich.prompt import Prompt
claude = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
exa = Exa(api_key=os.getenv("EXA_API_KEY"))
console = Console()
```
<a href="https://dashboard.exa.ai/login?redirect=/api-keys" target="_blank" class="button"><span>Get Exa API Key</span></a>
<style>
.button {
  display: flex;
  align-items: center;
  justify-content: center;
  background-color: #018ef5 !important;
  width: 100%;
  padding: 8px 10px;
  text-decoration: none !important;
  border-radius: 5px;
  font-weight: bold;
}
<br />
.button:hover { 
  background-color: #0180dd !important; 
} 
</style>
We'll define the `SYSTEM` constant here, which we will later use to tell Claude what it is supposed to do:
```python
SYSTEM_MESSAGE = "You are an agent that has access to an advanced search engine. Please provide the user with the information they are looking for by using the search tool provided."
```
Next, we define the function and function schema so that Claude knows that it exists and what arguments our local function takes:
```python
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
```
Create the `exa_search` function that will call Exa's search function with the query:
```python
def exa_search(query: str) -> Dict[str, Any]:
    return exa.search_and_contents(query=query, type='auto', highlights=True)
```
Now we'll create a function to process the tool calls:
```python
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
```
Now we'll create a `main` function to handle user input and interaction with Claude:
```python
def main():
    messages = []
    while True:
        try:
            user_query = Prompt.ask(
                "[bold yellow]What do you want to search for?[/bold yellow]",
            )
            messages.append({"role": "user", "content": user_query})
            completion = claude.messages.create(
                model="claude-3-sonnet-20240229",
                max_tokens=1024,
                system=SYSTEM_MESSAGE,
                messages=messages,
                tools=TOOLS,
            )
            message = completion.content[0]
            tool_calls = [content for content in completion.content if content.type == "tool_use"]
            if tool_calls:
                search_results = process_tool_calls(tool_calls)
                messages.append({"role": "assistant", "content": f"I've performed a search and found the following results: {search_results}"})
                messages.append({"role": "user", "content": "Please summarize this information and answer my previous query based on these results."})
                completion = claude.messages.create(
                    model="claude-3-sonnet-20240229",
                    max_tokens=1024,
                    system=SYSTEM_MESSAGE,
                    messages=messages,
                )
                response = completion.content[0].text
                console.print(Markdown(response))
                messages.append({"role": "assistant", "content": response})
            else:
                console.print(Markdown(message.text))
                messages.append({"role": "assistant", "content": message.text})
        except Exception as e:
            console.print(f"[bold red]An error occurred:[/bold red] {str(e)}")
if __name__ == "__main__":
    main()
```
### 4. Running the code
Remember that you need to have your `ANTHROPIC_API_KEY` and `EXA_API_KEY` set as environment variables or set them when initializing the Anthropic and Exa objects.
The implementation creates a loop that continually prompts the user for search queries, uses Claude's tool use feature to determine when to perform a search, and then uses the Exa search results to provide an informed response to the user's query.
The script uses the rich library to provide a more visually appealing console interface, including colored output and markdown rendering for the responses.
Now you have an advanced search tool that combines the power of Claude's language models with Exa's semantic search capabilities, providing users with informative and context-aware responses to their queries.
### Full code
Here's the complete code for the Claude and Exa integration:
```python
import os
from typing import Any, Dict
import anthropic
from exa_py import Exa
from rich.console import Console
from rich.markdown import Markdown
from rich.prompt import Prompt
claude = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
exa = Exa(api_key=os.getenv("EXA_API_KEY"))
console = Console()
SYSTEM_MESSAGE = "You are an agent that has access to an advanced search engine. Please provide the user with the information they are looking for by using the search tool provided."
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
def exa_search(query: str) -> Dict[str, Any]:
    return exa.search_and_contents(query=query, type='auto', highlights=True)
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
            user_query = Prompt.ask(
                "[bold yellow]What do you want to search for?[/bold yellow]",
            )
            messages.append({"role": "user", "content": user_query})
            completion = claude.messages.create(
                model="claude-3-sonnet-20240229",
                max_tokens=1024,
                system=SYSTEM_MESSAGE,
                messages=messages,
                tools=TOOLS,
            )
            message = completion.content[0]
            tool_calls = [content for content in completion.content if content.type == "tool_use"]
            if tool_calls:
                search_results = process_tool_calls(tool_calls)
                messages.append({"role": "assistant", "content": f"I've performed a search and found the following results: {search_results}"})
                messages.append({"role": "user", "content": "Please summarize this information and answer my previous query based on these results."})
                completion = claude.messages.create(
                    model="claude-3-sonnet-20240229",
                    max_tokens=1024,
                    system=SYSTEM_MESSAGE,
                    messages=messages,
                )
                response = completion.content[0].text
                console.print(Markdown(response))
                messages.append({"role": "assistant", "content": response})
            else:
                console.print(Markdown(message.text))
                messages.append({"role": "assistant", "content": message.text})
        except Exception as e:
            console.print(f"[bold red]An error occurred:[/bold red] {str(e)}")
if __name__ == "__main__":
    main()
```
