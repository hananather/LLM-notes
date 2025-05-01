Code interpreter allows the API to wrtie and run Python Code in a sandboxed execution.

- with code interpreter enabled the assistant can run code iteratively to solve more challanging code, math, and data analysis problems

- when the assistant wrties code that fails to run, it can iterate on this code by modifying and runnning different code until the execution succeeds

- this tool can process files with diverse data and formatting and generate files with data and images of graphs

- you can enable code interpreter via passing `code_interpreter` in the tools parameter

- the mdoel decides to invoke Code Interpreter in a Run based on the nature of user request

### Assistant
Think of an Assistatn as a template or profile of bot we have created. 
It defines:
- model
- tools it can use
- instructions you give it
- optionally: files it can access

Assistant level files are accessible to all "Runs" that use that assistant

### Run
A run is a specific instance of the Assistant being executed within a conversation (a Thread)
- You create a Run when the Assistant responds to a user message
- it refers to the combination of an Assistant and Thread

### Thread
A Thread is the ongoing conversation histroy between a user and Assistant
- A Thread holds all the messages so far (like chat logs)
- its presistent, so you can go back and continue conversations
- you can pass Thread-level files which only that conversation can access


SAS Runtime Availability:  you can run reference code automatically (e.g., via saspy or Viya Job Execution).


How much floating‑point drift is acceptable when comparing SAS vs Python outputs?

Do you need byte‑for‑byte identical tables or statistical equivalence?



“80 % of real‑world SAS code uses 20 % of the language.”

https://tree-sitter.github.io/tree-sitter/

