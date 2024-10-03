# Using OpenAI's "Tool Use" Feature with Exa Search Integration
This guide will show you how to properly set up and use OpenAI's and Exa's API client, and utilize OpenAI's function calling or "tool use" feature to perform Exa search integration. 

### What this guide covers
- installing the prerequisit packages
- setting up API keys as environment variables
- explain how OpenAI's "tool use" feature works
- explain how to use Exa within the tool call

## Guide
### 1. Pre-requisites and installation
Before you can use this guide you will need to have [python3](https://www.python.org/doc/) and [pip](https://pip.pypa.io/en/stable/installation/) installed on your machine.

For the purpose of this guide we will need to install:

- `anthropic` library to perform Claude api calls and completions
- `exa_py` library to perform Exa search
- `rich` library to make the output more readable

Install the libraries.

```python Python
pip install openai exa_py rich
```

To successfully use the Exa search client and OpenAI client you will need to have your `OPENAI_API_KEY` and `EXA_API_KEY` 
set as environment variables.

To get OpenAI API key, you will first need an OpenAI account, visit [OpenAI playground](https://platform.openai.com/api-keys) to generate your API key.

Similary, to get Exa API key, you will first need an Exa account, visit [Exa dashboard](https://dashboard.exa.ai/api-keys) to generate your API key.

> Be safe with your API keys. Make sure they are not hardocded in your code or added in a git repository to prevent leaking them to the public.

You can create an `.env` file in the root of your project and add the following to it:

```bash
OPENAI_API_KEY=insert your Anthropic API key here, without the quotes
EXA_API_KEY=insert your Exa API key here, without the quotes
```

Make sure to add your `.env` file to your `.gitignore` file if you have one.

### 2. What is OpenAI tool calling?
OpenAI LLM's can call a function you have defined in your code, this is called [tool calling](https://platform.openai.com/docs/guides/function-calling?lang=python). To do this you first need to describe the function you want to call to OpenAI's LLM. You can do this by defining a description object of the format:

```json
{
    "name": "my_function_name", # The name of the function
    "description": "The description of my function", # Describe the function so OpenAI knows when and how to use it.
    "input_schema": { # input schema describes the format and the type of parameters OpenAI needs to generate to use the function
        "type": "object", # format of the generated OpenAI reponse
        "properties": { # properties defines the input parameters of the function
            "query": { # the function expects a query parameter
                "type": "string", # of type string
                "description": "The search query to perform.", # describes the paramteres to Calude
            },
        },
        "required": ["query"], # define which parameters are required
    },
}
```

When this description is sent to OpenAI's LLM, it returns an object with a string, which is the function name defined in _your_ code, and the arguments that the function takes. This does not execute or _call_ functions on OpenAI's side; it only returns the function name and arguments which you will have to parse and call yourself in your code.

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

### 3. Use Exa Search as OpenAI tool
First we import and initialize the OpenAI and Exa libraries and load the stored API keys. 

```python

from dotenv import load_dotenv
from exa_py import Exa
from openai import OpenAI

load_dotenv()

openai = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
exa = Exa(api_key=os.getenv("EXA_API_KEY"))
```

Next, we define the function and the function schema so that OpenAI knows how to use it and what arguments our local function takes:

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

Finally we'll define the primer `SYSTEM_MESSAGE`, which explains to OpenAI what it is supposed to do:

```python
SYSTEM_MESSAGE = {
    "role": "system",
    "content": "You are an agent that has access to an advanced search engine. Please provide the user with the information they are looking for by using the search tool provided.",
}
```

We can now start writting the code needed to perfrom the LLM calls and the search. We'll create the `exa_search` function that will call Exa's `search_and_contents` function with the query:

```python
def exa_search(query: str) -> Dict[str, Any]:
    return exa.search_and_contents(query=query, type='auto', highlights=True)
```

Next we create a function to process the tool calls:

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

Lastly we'll create a `main` function to bring it all together, handle the user input and interaction with OpenAI:

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

The implementation creates a loop that continually prompts the user for search queries, uses OpenAI's tool use feature to determine when to perform a search, and then uses the Exa search results to provide an informed response to the user's query.

We also use the rich library to provide a more visually appealing console interface, including colored output and markdown rendering for the responses.

### Full code

```python
import json
import os

from dotenv import load_dotenv
from typing import Any, Dict
from exa_py import Exa
from openai import OpenAI
from rich.console import Console
from rich.markdown import Markdown
from rich.prompt import Prompt

# Load environment variables from .env file
load_dotenv()

# create the openai client
openai = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# create the exa client
exa = Exa(api_key=os.getenv("EXA_API_KEY"))

# create the rich console
console = Console()

# define the system message (primer) of your agent
SYSTEM_MESSAGE = {
    "role": "system",
    "content": "You are the world's most advanced search engine. Please provide the user with the information they are looking for by using the tools provided.",
}

# define the tools available to the agent - we're defining a single tool, exa_search
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

# define the function that will be called when the tool is used and perform the search
# and the retrival of the result highlights.
# https://docs.exa.ai/reference/python-sdk-specification#search_and_contents-method
def exa_search(query: str) -> Dict[str, Any]:
    return exa.search_and_contents(query=query, type='auto', highlights=True)

# define the function that will process the tool call and perform the exa search
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
            # create the user input prompt using rich
            user_query = Prompt.ask(
                "[bold yellow]What do you want to search for?[/bold yellow]",
            )
            messages.append({"role": "user", "content": user_query})
            
            # call openai llm by creating a completion which calls the defined exa tool
            completion = openai.chat.completions.create(
                model="gpt-4o",
                messages=messages,
                tools=TOOLS,
                tool_choice="auto",
            )
            
            # completion will contain the object needed to invoke your tool and perform the search
            message = completion.choices[0].message
            tool_calls = message.tool_calls
            
            if tool_calls:

                messages.append(message)

                # process the tool object created by OpenAI llm and store the search results
                messages = process_tool_calls(tool_calls, messages)
                messages.append(
                    {
                        "role": "user",
                        "content": "Answer my previous query based on the search results.",
                    }
                )
                
                # call OpenAI llm again to process the search results and yield the final answer
                completion = openai.chat.completions.create(
                    model="gpt-4o",
                    messages=messages,
                )
                
                # parse the agents final answer and print it
                console.print(Markdown(completion.choices[0].message.content))
            else:
                console.print(Markdown(message.content))
        except Exception as e:
            console.print(f"[bold red]An error occurred:[/bold red] {str(e)}")
            
            
if __name__ == "__main__":
    main()

```

We have now written an advanced search tool that combines the power of Claude's language models with Exa's semantic search capabilities, providing users with informative and context-aware responses to their queries.

### 4. Running the code

Save the code in a file, ie. `openai_search.py`, and make sure the `.env` file containing the API kyes we previoulsy created is in the same directory as the script.

Then run the script using the following command from your terminal:

```bash
python openai_search.py
```

You should see a prompt: 

```bash
What do you want to search for?
```

Let's test it out.

```bash
What do you want to search for?: Who is Tony Stark?
Context updated with exa_search (None):  Tony Stark
Tony Stark, also known as Iron Man, is a fictional superhero from Marvel Comics. He is a wealthy inventor and businessman, known for creating a powered suit of armor that gives him superhuman abilities. Tony Stark is a founding member of the Avengers and has appeared in various comic book series, animated
television shows, and films within the Marvel Cinematic Universe.

If you're interested in more detailed information, you can visit Tony Stark (Marvel Cinematic Universe) - Wikipedia.
```

That's it, enjoy your search agent!
