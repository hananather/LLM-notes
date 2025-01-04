<iframe src="https://link.excalidraw.com/readonly/oPRzp7AJh8RfqtYBpHYu?darkMode=true" width="100%" height="100%" style="border: none;"></iframe>

# Introduction

Humans are great at messy pattern recognition tasks but they often rely on tools like books, Google search, or a calculator to supplement their prior knowledge before arriving at a conclusion. 

LLMs can be trained to use tools to access real-time information or suggestiona real-world action. 
- model can leverage a database retrieval tool to access specific information like a customers purchase history so it can generated tailored shopping recomendations
- based on users query, a model can make various API calls to send an email reponse to a colleague

To do so, the model must not only have access to a set of externel tools, it needs the ability to plan and execute any task in a self-directed fashion. 

**Agent** = reasoning + logic + access to externel information

# What is an agent? 

- an applicaiton that attempts to achieve a goal by observing the world and acting upon it using the tools that it has at its disposal. 
- autonomous and can act indepently of human intervention, when provided with proper goals or objectives they are meant to achieve

### The model

Model = Laguage model that will be utilized as the **central decision maker** for agent process
- can be one or multiple LMs of anysize (small / large) that a capable of following instructions based on reasoning or logic framework, like ReAct, Chain-of-thought, or Tree-of-thoughts
- models can be general purpose, multimodal, or fine-tuned based on the needs of your specific agent architecture; for best production results you should leverage the model that best fits your desired end application and ideally has been trained of **data signatures** associated wiht the tools yor plan to use in the agents. architecture
  - important: model is typically not trained with the specific configuration settings (i.e., tool choices, orchestration/reasoning setup) of the agent, however its possible to further refine the model for the agents tasks by providing it with examples that showcase the agent’s capabilities, including instances of the agent using specific tools or reasoning steps in various contexts

### The tools

Tools allow agents to interact with externel data and services while unlocking a wider range of actions beyond that of the underlying model alone.

Tools can take a varity of forms but typically align with GET, POST, PATCH, and DELETE.
- update customer information in a database
- fetch weather data to influence a travel recomendation that the agent is providing to the user

### The orchestration layer

**Cyclical process** that governs how the agent takes in information, performs some internal reasoning, and uses the reasoning. 

## Tools

While they go by many names, tools are what create a link between our foundational models and the outside world. 

For instance tools can enable agents to adjust smart home settings, update calendars, fetch user information from a database, or send emails based on specific set of instructions. 
- a LM is only as good as what it has learned from its training data
- regardless of how much data we throw at the model, they still lack the fundamental ability to interact with the outside world

Three primary tool types Google Models are able to interact with: 
- Extensions
- Functions
- Data Stores

###  Extensions

Built an agent with the goal of helping users book flights. You know you want to use the Google Flights API to retrieve flight information. But you are not sure how you’re going to get your agent to make calls to this API end point. 

**Approach: **
- implement custom code that takes the incoming user query, parse the query for relevant information, then make the API call. “I want to book a flight from Austin to Zurich.” 
- **Challenge**: In this scenario,  our custom code solution would need to extract “Austin” and “Zurich” as relevant entities
  - What happens if the user says “I want to book a flight to Zurich”? and never provides a departure city? The API call would fail without the required data
  - This approach is not scalable and could easily break in any scenario that fails outside the implemented custom code

**Solutions**:

Extension: 
1. Teaching the agent how to use API endpoint using examples
2. Teaching the agent what arguments or parameters are needed to successfully call the API endpoint.

## Functions

In programing functions are defined as self-contained module of code that accomplish a specific task and can be reused as needed. We generally create many functions to do various tasks. They also define the logic for when to call function_a versus function_b, as well as the expected inputs and outputs. 

Functions work similarly in the world of agents, but we can replace the software developer with a model. A model can take a set of known functions and decide when to use each function and what arguments the function needs based on its application. 

Functions differ from Extensions in a few ways, most notably:
1. A model outputs a Function and its argument, but doesn’t make a live API call
2. Functions are executed on the **client-side**, while Extensions are executed on the **agent-side.**

The main difference is that neither the Function nor the agent interact directly with the Google Flights API. So how does the API call actually happen?

With functions, the logic and execution of calling the actual API endpoint is offloaded away from the agent and back to the client-side applicaiton. 

This offers the devloper more granular control over the flow of data in the application. There are many reasons why a Developer might choose to use functions over Extensions, but a few common use cases are:
1. API calls need to be made at another layer of the application, outside of the direct agent architeture flow (e.g. a middleware system, a front end framework, etc..)
2. Security or Authentication restrictions that prevent the agent from calling an API directly (e.g. API is not exposed to the internet, or non-accessible by agent infrastructure) 
3. Timing or order-of-operation constraints that prevent te agent from making API calls in real-time (i.e., batch operations, human-in-the-loop review, etc..)

### Use Cases

A model can be used to invoke functions in order to handle complex, client-side execution flows for end users, where the agent Developer might not want the language model to manage the API execution (as is the case with Extensions). 

**Example**: 

Agent is being trained as a travel concierge to interact with users that they to book vacation trips. 

**User**: “I’d like to take a ski trip with my family but I’m not sure where to go.” 

The goal is to get the agent to produce a list of cities that we can use in our middleware application to download images, data, etc. for the user’s trip planning. 

In a typical prompt to the model, the output might look like the following:

> Sure, here’s a list of cities that you can consider for family ski trips:>
> - Crested Butte, Colorado, USA>
> - Whistler, BC, Canada>
> - Zermatt, Switzerland

The above output contains the data we need (city names), the format isn’t ideal for parsing. With function calling we can teach the model to format this output in a structured style (JSON) that is more convenient for another system to parse. 

```
function_call {
name: "display_cities"
  args: {
    "cities": ["Crested Butte", "Whistler", "Zermatt"],
    "preferences": "skiing"
   }
}
```

The JSON payload is generated by the model, and then sent to our Client-side server to do whatever we would like to do with it. 
