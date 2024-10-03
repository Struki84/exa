## What this doc covers
Explain OpenAI's tool calling feature
Show you how to use Exa within the tool call
## Guide
### 1. Pre-requisites and installation
Install OpenAI, rich and the Exa library
```python Python
pip install openai exa_py rich
```
### 2. What is OpenAI tool calling?
OpenAI's [tool calling](https://platform.openai.com/docs/guides/function-calling?lang=python) feature returns an object with a string, which is the function name defined in _your_ code and the arguments that the function takes.
For example:
```python
...
id='call_62136123',
function=Function(
    arguments='{"query":"Latest developments in quantum computing"}',
    name='exa_search',),
type='function'
...
```
We will use this object to - in this case - call the `exa_search` function we define with the arguments provided.
### 3. Use Exa Search in your tools
We have to import and initialize the OpenAI and Exa libraries (also the JSON library because we will be converting the arguments string to a dictionary). We will also use components from the `rich` library to make the output more readable, and import some types to make the code more readable and extensible.
```python
import json
import os
from typing import Any, Dict
from exa_py import Exa
from openai import OpenAI
from rich.console import Console
from rich.markdown import Markdown
from rich.prompt import Prompt
openai = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
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
<br />
<br />
<br />
<br />
<br />
<br />
<br />
<br />
<br />
<br />
<br />
&:hover { 
background-color: #0180dd !important; 
} 
} 
</style>
We will define the `SYSTEM` constant here, which we will later use to tell the AI what it is supposed to do:
```python
SYSTEM_MESSAGE = {
    "role": "system",
    "content": "You are an agent that has access to an advanced search engine. Please provide the user with the information they are looking for by using the search tool provided.",
}
```
Next, we define the function and function schema so that OpenAI knows that it exists and what arguments our local function takes
```python
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "exa_search",
            "description": "Perform a search query on the web, and retrieve the most relevant URLs/web data.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search query to perform.",
                    },
                },
                "required": ["query"],
            },
        },
    }
]
```
Here we define the tool object that OpenAI will use to call the function. The `type` is set to `function` and the `function` object contains the `name` of the function which must match the name of the function in your code. The `parameters` object contains the arguments that the function takes (in this case just the `query`)
Create the `exa_search` function that will call Exa's search function with the query and mode which will be provided by the AI calling the tool. We simply use the `exa_py` method `search_and_contents` here:
```python
def exa_search(query: str) -> Dict[str, Any]:
    return exa.search_and_contents(query=query, type='auto', highlights=True)
```
Because we can have multiple tools, it is a good practice to modularize the code and create a function that will call the correct function based on the tool call object:
```python
def process_tool_calls(tool_calls, messages):
    for tool_call in tool_calls:
        function_name = tool_call.function.name
        function_args = json.loads(tool_call.function.arguments)
        if function_name == "exa_search":
            search_results = exa_search(**function_args)
            messages.append(
                {
                    "role": "tool",
                    "content": str(search_results),
                    "tool_call_id": tool_call.id,
                }
            )
            console.print(
                f"[bold cyan]Context updated[/bold cyan] [i]with[/i] "
                f"[bold green]exa_search ({function_args.get('mode')})[/bold green]: ",
                function_args.get("query"),
            )
    return messages
```
Now we are almost done, we just need to get the user input and actually call the OpenAI completion endpoint with the tool object and print out the result. We will do this in a `main` function to keep the code clean:
```python
def main():
    messages = [SYSTEM_MESSAGE]
    while True:
        try:
            user_query = Prompt.ask(
                "[bold yellow]What do you want to search for?[/bold yellow]",
            )
            messages.append({"role": "user", "content": user_query})
            completion = openai.chat.completions.create(
                model="gpt-4o-mini",
                messages=messages,
                tools=TOOLS,
            )
            message = completion.choices[0].message
            tool_calls = message.tool_calls
            if tool_calls:
                messages.append(message)
                messages = process_tool_calls(tool_calls, messages)
                messages.append(
                    {
                        "role": "user",
                        "content": "Answer my previous query based on the search results.",
                    }
                )
                completion = openai.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=messages,
                )
                console.print(Markdown(completion.choices[0].message.content))
            else:
                console.print(Markdown(message.content))
        except Exception as e:
            console.print(f"[bold red]An error occurred:[/bold red] {str(e)}")
if __name__ == "__main__":
    main()
```
### 4. Running the code
If you have skipped to this part, remember that you will have your `OPENAI_API_KEY` and `EXA_API_KEY` set as environment variables or set them when initializing the OpenAI and Exa objects.
The implementation below creates a loop that continually prompts the user for search queries, uses OpenAI's tool calling feature to determine when to perform a search, and then uses the Exa search results to provide an informed response to the user's query.
The script uses the rich library to provide a more visually appealing console interface, including colored output and markdown rendering for the responses.
Now you have an advanced search tool that combines the power of OpenAI's language models with Exa's semantic search capabilities, providing users with informative and context-aware responses to their queries.
### Full code
```python
import json
import os
from typing import Any, Dict
from exa_py import Exa
from openai import OpenAI
from rich.console import Console
from rich.markdown import Markdown
from rich.prompt import Prompt
openai = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
exa = Exa(api_key=os.getenv("EXA_API_KEY"))
console = Console()
SYSTEM_MESSAGE = {
    "role": "system",
    "content": "You are the world's most advanced search engine. Please provide the user with the information they are looking for by using the tools provided.",
}
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "exa_search",
            "description": "Perform a search query on the web, and retrieve the world's most relevant information.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search query to perform.",
                    },
                },
                "required": ["query"],
            },
        },
    }
]
def exa_search(query: str) -> Dict[str, Any]:
    return exa.search_and_contents(query=query, type='auto', highlights=True)
def process_tool_calls(tool_calls, messages):
    for tool_call in tool_calls:
        function_name = tool_call.function.name
        function_args = json.loads(tool_call.function.arguments)
        if function_name == "exa_search":
            search_results = exa_search(**function_args)
            messages.append(
                {
                    "role": "tool",
                    "content": str(search_results),
                    "tool_call_id": tool_call.id,
                }
            )
            console.print(
                f"[bold cyan]Context updated[/bold cyan] [i]with[/i] "
                f"[bold green]exa_search ({function_args.get('mode')})[/bold green]: ",
                function_args.get("query"),
            )
    return messages
def main():
    messages = [SYSTEM_MESSAGE]
    while True:
        try:
            user_query = Prompt.ask(
                "[bold yellow]What do you want to search for?[/bold yellow]",
            )
            messages.append({"role": "user", "content": user_query})
            completion = openai.chat.completions.create(
                model="gpt-4o-mini",
                messages=messages,
                tools=TOOLS,
                tool_choice="auto",
            )
            message = completion.choices[0].message
            tool_calls = message.tool_calls
            if tool_calls:
                messages.append(message)
                messages = process_tool_calls(tool_calls, messages)
                messages.append(
                    {
                        "role": "user",
                        "content": "Answer my previous query based on the search results.",
                    }
                )
                completion = openai.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=messages,
                )
                console.print(Markdown(completion.choices[0].message.content))
            else:
                console.print(Markdown(message.content))
        except Exception as e:
            console.print(f"[bold red]An error occurred:[/bold red] {str(e)}")
if __name__ == "__main__":
    main()
```
You’ve now built a search tool leveraging Exa's semantic capabilities and integrated it with OpenAI's GPT-4o through its tool calling feature.
