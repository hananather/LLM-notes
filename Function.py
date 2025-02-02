import marimo

__generated_with = "0.10.19"
app = marimo.App()


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        # Function Calling
        **Function calling** enables us to connect LLMs to external data and systems. 
        We can define a set of functions as tools that the model has access to, and it can use them when appropriate based on the conversation history. We can then execute those functions on the **application** side, and provide the results back to the model. 

        ## How Function Calling Works
        "Function calling" naming is confusing. LLMs don't actually call any functions themselves; they suggest which functions you should call from  pre-defined functions which you provide to the LLM in a prompt. Function calling is a type of structured ouput capability of a LLM. Yet, strctured output enables us to intergate LLMs with classical software systems.

        When you use function calling, the model never actually executes functions itself, instead it simply generates parameters that can be used to call our function.

        ![FunctionCalling.png](attachment:37829098-7147-491a-b691-c9b4e982a9f1.png)

        The starting point for function calling is choosing a function in your own codebase that you'd like to enable the model to generate arguments for. 

        ### Handling model responses
        The model only suggests function calls and generates arguments for the defined functions when appropriate. It is then up to us to decide how our application handles the execution of these suggestions. 

        If the model determines that a function should be called, it will return a `tool_calls` field in the response, which we can use to determine if the model generated a function call and what the argumets were.

        #### Warning
        Before diving any deeper, remember, all an LLM does it next token prediction. Due to the principle LLM architecture all it can possibly output is one token with the highes probability. In documentation OpenAI refers to function calling as a capability. Function calling capability is achieved via fine-tuning of a model, when enables it to output data in a specific way. To oversimplify, a structured-output-capable model was just trained on more JSON. Therefore, function calling is merely JSON structured output which contains the name of a function to call and parameters for it.

        #### Why do we care about function calling?
        Three main reasons:
        1. **Models don't have all the data**. Models acquire their "knowledge" during the training process, and that "knowlege" is stored in the model's weights, we cannot be easily updated on demand. To update "knowlege" you effectively need to update the model's weights and run all the post-training routines.

        2. **Models need to be integrated with other systems**. Strcutured output capability enables output to be integrated with other parts of the system via JSON. We can even treat strctured output-enabled LLM as some sort of API which returns JSON. There is a world of difference between: 

        ```
        Hi, my name is Hanan Ather, I'm 27, and my email is hanan@example.com

        ```
        and 
        ```
        {
            "name": "Hanan Ather",
            "age": 27,
            "email": "hanan@example.com"
        }

        ```
        Strcutured output not only make integrations simpler, but they easily enable tasks which normally required eloborated NLP systems. 


        Many applications require models to call custom functions to trigger actions within the application or interact with external systems.

        - Fetching data: enable a conversational assistant to retrieve databased on conversation, like scheduling meetings or initiating order systems
        - Taking action: allowing a assistant to trigger actions based on the conversation, like scheduling meetings or initiating order returns
        - Building workflows: allow assistants to execute multi-step workflows, like data extraction pipelines or content personalization

        ## Structured Output
        Structured output is a models capability to output JSON, acquired during fine-tuning. 

        **Use case of strcutured output:**
        - implement a natural langugage processing parser that allows users to create grocery lists out of natural langauge input. The use provides a list of groceries in written or spoken form, and the program outputs an formatted list.

          - Without LLMs, this is not such an easy task to tackle. Its easy to build a demo, but not easy to build-high quality prodcut that handles edge cases well. 

        - Today we can accomplish this by: pipe user input into the LLM -> LLM outputs JSON -> Python picks it up and formats the JSON into HTML
        """
    )
    return


@app.cell
def _():
    import os
    from openai import OpenAI
    from openai.types.chat import ChatCompletion

    def eval(prompt: str, message: str, model: str='gpt-4o') -> ChatCompletion:
        client = OpenAI(api_key=os.environ['OPENAI_API_KEY'])
        messages = [{'role': 'system', 'content': prompt}, {'role': 'user', 'content': _message}]
        return client.chat.completions.create(model=model, messages=messages)
    return ChatCompletion, OpenAI, eval, os


@app.cell
def _(eval):
    prompt = '\nYou are a data parsing assistant. \nUser provides a list of groceries. \nYour goal is to output it as JSON.\n'
    _message = "I'd like to buy some bread, pack of eggs, few apples, and a bottle of milk."
    res = eval(prompt=prompt, message=_message)
    _json_data = res.choices[0].message.content
    print(_json_data)
    return prompt, res


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""We can see that the model didn't return JSON, it returned markdown formated string containing JSON. The reason is that we didn't enable strcutured output in the API call.""")
    return


@app.cell
def _(ChatCompletion, OpenAI, os):
    def eval_1(prompt: str, message: str, model: str='gpt-4o') -> ChatCompletion:
        client = OpenAI(api_key=os.environ['OPENAI_API_KEY'])
        messages = [{'role': 'system', 'content': prompt}, {'role': 'user', 'content': _message}]
        return client.chat.completions.create(model=model, messages=messages, response_format={'type': 'json_object'})
    return (eval_1,)


@app.cell
def _(eval_1):
    prompt_1 = '\nYou are a data parsing assistant. \nUser provides a list of groceries. \nYour goal is to output it as JSON.\n'
    _message = "I'd like to buy some bread, pack of eggs, few apples, and a bottle of milk."
    res_1 = eval_1(prompt=prompt_1, message=_message)
    _json_data = res_1.choices[0].message.content
    print(_json_data)
    return prompt_1, res_1


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        Now, running the same code returns plain JSON. This is not only great because we don't need to parse anything extra but, but it also guarantees that the LLM won't include any free-from text such as "Sure, here is your data!{}"

        The problem is, we don't have the data shaped defined; lets call it *schema*. Our schema is now up to the LLM, and it might change based on user input. Lets rephrase the user query to see it in action.  Instead of asking, “I’d like to buy some bread, a pack of eggs, a few apples, and a bottle of milk,” let’s ask, “12 eggs, 2 bottles of milk, 6 sparkling waters.”
        """
    )
    return


@app.cell
def _(eval_1, prompt_1):
    _message = '12 eggs, 2 bottles of milk, 6 sparkling waters'
    res_2 = eval_1(prompt=prompt_1, message=_message)
    _json_data = res_2.choices[0].message.content
    print(_json_data)
    return (res_2,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""Strcutured output not only enables JSON ouput, it also helps with schema. The way we can state and define schema is via prompt engineering.""")
    return


@app.cell
def _(eval_1):
    prompt_2 = '\nYou are data parsing assistant. \nUser provides a list of groceries. \nUse the following JSON schema to generate your response:\n\n{{\n    "groceries": [\n        { "name": ITEM_NAME, "quantity": ITEM_QUANTITY }\n    ]\n}}\n\nName is any string, quantity is a numerical value.\n'
    inputs = ["I'd like to buy some bread, pack of eggs, few apples, and a bottle of milk.", '12 eggs, 2 bottles of milk, 6 sparkling waters.']
    for _message in inputs:
        res_3 = eval_1(prompt=prompt_2, message=_message)
        _json_data = res_3.choices[0].message.content
        print(_json_data)
    return inputs, prompt_2, res_3


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Serialization
        Usually, JSON ouput won't cut it in software systems. Its just a string after all. We have to ensure that the LLM indeed returns correctly formed data.
        """
    )
    return


@app.cell
def _(OpenAI, os):
    from typing import TypeVar, List, Self, Any, Generic, Callable, Optional
    from dataclasses import dataclass
    import json
    from pprint import pprint as pp
    T = TypeVar('T')

    @dataclass(frozen=True)
    class Item:
        name: str
        quantity: int

    @dataclass(frozen=True)
    class Groceries:
        groceries: List[Item]

        @staticmethod
        def serialize(data: Any) -> Self:
            """JSON serialization function."""
            _json_data = json.loads(data)
            items = [Item(**item) for item in _json_data['groceries']]
            return Groceries(groceries=items)

    def eval_2(prompt: str, message: str, schema: Generic[T], serializer: Callable=None, model: str='gpt-4o', client: OpenAI=OpenAI(api_key=os.environ['OPENAI_API_KEY'])) -> Optional[T]:
        messages = [{'role': 'system', 'content': prompt}, {'role': 'user', 'content': _message}]
        completion = client.chat.completions.create(model=model, response_format={'type': 'json_object'}, messages=messages)
        try:
            completion_data = completion.choices[0].message.content
            _json_data = json.loads(completion_data)
            if serializer is not None:
                return serializer(completion_data)
            else:
                return schema(**_json_data)
        except TypeError as type_error:
            return None
        except json.JSONDecodeError as json_error:
            return None
    return (
        Any,
        Callable,
        Generic,
        Groceries,
        Item,
        List,
        Optional,
        Self,
        T,
        TypeVar,
        dataclass,
        eval_2,
        json,
        pp,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        1. First we declare a type variable which will be used to pass schema class as a parameter.
        2. Then we describe our JSON data using the data class
        3. `Gorceries` has `serialize` defined on it to parse nested JSON correctly. Our `eval` function is in principle the same as before. It gets a `schema` and an optional `serializer` for data serialization.
        """
    )
    return


@app.cell
def _(Groceries, eval_2, pp, prompt_2):
    res_4 = eval_2(prompt=prompt_2, message="I'd like to buy some bread, pack of eggs, few apples, and a bottle of milk.", schema=Groceries, serializer=Groceries.serialize)
    pp(res_4)
    return (res_4,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Structured Extraction
        Function calling can be used to extract structured data from natural language input. 
        Structured extraction is fundamentally about mapping unstructured text $\to$ structured data. Think of it as specialized form of feature engineering, but instead of transforming existing structured data, we're extracting it from text. 

        ---
        ### Dataclasses 
        Think of a [dataclass](https://docs.python.org/3/library/dataclasses.html) as a blueprint for storing related pieces of information together, with a lot of features automatically added. Its like a smart container that knows how to handle its contents properly.

        The key benefits of `dataclasses` are 
        1. reduces repetitive code for data-holding classes
        2. enforces type saftey through annotations
        3. handle common operations (equality testing, string representation, etc.)

        They are useful when our class is primarly about storing data rather than behavior and we want immutable data structures. They provide clear type definitions for our data and automatically generate common methods (`__init__`,`__repr__`, `__eq__`). 

        We use the dataclass as a schema to identify what fields to extract and validate the data matches expected formats
        - postal code follows A1A 1A1 pattern
        - province is valid two-letter
        - required fields are present

        The dataclass ensures our extracted data is:
        - complete (all required fields present)
        - correctly typed
        - validated


        ---
        Structured extraction is about converting messy text into clean structured data. **The key idea is to use LLMs as our pattern recognition engine, turning unstructured text into structured JSON that matches our schema.**
        Think of the LLM as a smart pattern recognizer that can understand context and variations in how people write addresses. It's more flexible than regex patterns but needs the structure of our dataclasses to ensure reliable output.

        ---
        The "structured" part needs to be precisely defined - what fields do we want to extract, what types should they be, how should they be validated? This is where dataclasses shine:
        1. Define the structure you want to extract using dataclasses
        2. Use this structure as both a schema and a container for the extracted data
        3. Validate and process the extracted data
        ---

        ## Example: Extracting Addresses
        A common real-world application is extracting business addresses from unstructured text (like web pages, documents, or databases) into a standardized format. Here's how we'd use dataclasses for this:


        ### Schema Definition
        The schema is our contract with the data. It's where we say "this is exactly what we want to get out of our text". Using dataclasses makes this contract explicit and enforceable. Think of it like a form where every field has a specific purpose and type.

        Here's what a schema typically defines:
        1. What fields we want to extract (name, address, etc.)
        2. What type each field should be (string, number, etc.)
        3. Which fields are required vs optional
        4. Any special formats or patterns fields should follow

        We use dataclasses to define what data we want:

        ``` python
        @dataclass(frozen=True)
        class Address:
            name: str
            street_number: str
            street_name: str
            city: str
        ```

        ### Serialization
        We need to convert between JSON and our dataclasses. The `serialize` method handles this conversion:

        ```python
        @staticmethod
        def serialize(data: Any) -> Self:
            json_data = json.loads(data)  # JSON string → Python dict
            addresses = [Address(**addr)]  # dict → Address objects
            return AddressBook(addresses=addresses)
        ```
        ### LLM Integration

        The `eval` function is our bridge to the LLM. It has three main jobs: 
        1. Format our prompt and input for the LLM
        2. Get structured JSON back
        3. Convert that JSON into our dataclass objects

        ### Error Prevention
        The system can fail in several ways:
        - LLM returns malformed JSON
        - JSON doesn't match our schema
        - Data fails validation
        We handle these with:

        ``` python
        try:
            completion_data = completion.choices[0].message.content
            json_data = json.loads(completion_data)
            return serializer(completion_data)
        except TypeError:
            # Schema mismatch
        except json.JSONDecodeError:
            # Bad JSON
        ```
        #### Type Safety
        We use `Generic[T]` to make our function type-safe. We can think of generics like a template or a placeholder for a type that will be specified later. It's like having a box and saying "this box will hold something, but I'll tell you what it holds when I actually use it".

        ```Python
        T = TypeVar("T")

        def eval(
            prompt: str, 
            message: str,
            schema: Generic[T],  # This is our "template" - could be Address, Groceries, etc.
        ) -> Optional[T]:       # We'll return that same type (or None)
            # ... LLM processing ...
            return schema(**json_data)

        # Usage examples:
        address = eval(prompt, message, schema=Address)
        # Python knows address is an Address object

        groceries = eval(prompt, message, schema=Groceries)
        # Python knows groceries is a Groceries object
        ```

        We're using generics to create a flexible but type-safe function that can work with any dataclass schema.
        - **Type Safety**: If you try to use `address.items` when address is an `Address` object, Python will catch this error before you run the code
        - **Code Reuse**: The same eval function works for any schema you define
        - **IDE Support**: Your IDE can provide correct autocomplete because it knows what type you're working with

        Without generics, we'd either need separate functions for each type (`eval_address`, `eval_groceries`, etc.) or a function that returns `Any`, losing all type information



        #### Optional and Hierarchical Fields

        We can also enhance the Address structure to handle optional fields and hierarchical relationships. Here's how:

        ``` python
        @dataclass(frozen=True)
        class ResidentialUnit:
            unit_number: str
            floor: Optional[str] = None

        @dataclass(frozen=True)
        class Address:
            # Required fields
            street_number: str
            street_name: str
            city: str

            # Optional fields
            name: Optional[str] = None
            postal_code: Optional[str] = None

            # Hierarchical - only for apartments/units
            unit: Optional[ResidentialUnit] = None
        ```

        We are using `Optional` for fields that might not exist. We can create nested structures for related fields (like unit details), set sensible defaults where appropriate and use inheritance for different types of addresses if needed. 

        #### The Serialization Process
        Let's break down how the JSON-to-object conversion works. When the LLM returns JSON, it looks like this:
        ``` python
        json_string = \"\"\"
        {
            "addresses": [
                {
                    "name": "Hanan Ather",
                    "street_number": "123",
                    "street_name": "Elm St",
                    "city": "Ottawa"
                },
                {
                    "name": "Serge Godbout",
                    "street_number": "456",
                    "street_name": "Oak Ave",
                    "city": "Gatineau"
                }
            ]
        }
        \"\"\"
        ```
        `json.loads` converts this string to a Python dictionary:
        ```python
        {
            "addresses": [
                {"name": "Hanan Ather", ...},
                {"name": "Serge Godbout", ...}
            ]
        }
        ```
        """
    )
    return


@app.cell
def _(
    Any,
    Callable,
    Generic,
    List,
    OpenAI,
    Optional,
    Self,
    TypeVar,
    dataclass,
    json,
    os,
    pp,
):
    T_1 = TypeVar('T')

    @dataclass(frozen=True)
    class Address:
        name: str
        street_number: str
        street_name: str
        city: str

    @dataclass(frozen=True)
    class Addresses:
        addresses: List[Address]

        @staticmethod
        def serialize(data: Any) -> Self:
            """Convert JSON to Addresses instance.

            [Address(**addr) for addr in json_data["addresses"]] is:
            1. Getting the list of address dictionaries with json_data["addresses"]
            2. For each dictionary, unpacking it into the Address constructor with **addr
            3. Creating a list of Address objects
            """
            _json_data = json.loads(data)
            addresses = [Address(**addr) for addr in _json_data['addresses']]
            return Addresses(addresses=addresses)

    def eval_3(prompt: str, message: str, schema: Generic[T_1], serializer: Callable=None, model: str='gpt-4o', client: OpenAI=OpenAI(api_key=os.environ['OPENAI_API_KEY'])) -> Optional[T_1]:
        messages = [{'role': 'system', 'content': prompt}, {'role': 'user', 'content': _message}]
        completion = client.chat.completions.create(model=model, response_format={'type': 'json_object'}, messages=messages)
        try:
            completion_data = completion.choices[0].message.content
            _json_data = json.loads(completion_data)
            if serializer is not None:
                return serializer(completion_data)
            else:
                return schema(**_json_data)
        except TypeError as type_error:
            return None
        except json.JSONDecodeError as json_error:
            return None
    _text = "\nThe Ottawa Public Library is at 150 Elgin Street, Ottawa.\nDown the street, Sarah Wilson runs her bakery at 240 Laurier Avenue, Ottawa.\nOver in Kanata, Tech Corp's office is at 1385 Terry Fox Drive.\n"
    prompt_3 = '\nYou are address parsing assistant. \nUser provides text containing addresses.\nExtract addresses from the text.\nUse the following JSON schema to generate your response:\n\n\nFormat as JSON matching this example:\n{\n    "addresses": [\n        {\n            "name": PERSON_OR_BUSINESS_NAME,\n            "street_number": BUILDING_NUMBER,\n            "street_name": STREET_NAME,\n            "city": CITY_NAME\n        }\n    ]\n}\n\n'
    res_5 = eval_3(prompt=prompt_3, message=_text, schema=Address, serializer=Addresses.serialize)
    pp(res_5)
    return Address, Addresses, T_1, eval_3, prompt_3, res_5


@app.cell
def _(Address, Addresses, eval_3, pp, prompt_3):
    _text = '\nJohn Doe lives at 123 Elm Street, Springfield. Next to him is Jane Smith, residing at 456 Oak Avenue, Lakeview. Not far away, we find Dr. Emily Ryan at 789 Pine Road, Westwood. Meanwhile, in a different part of town, Mr. Alan Turing can be found at 101 Binary Blvd, Computerville. Nearby, Ms. Olivia Newton stays at 202 Music Lane, Harmony. Also, Prof. Charles Xavier is located at 505 Mutant Circle, X-Town.\n'
    print(_text)
    res_6 = eval_3(prompt=prompt_3, message=_text, schema=Address, serializer=Addresses.serialize)
    pp(res_6)
    return (res_6,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ---
        # Validation Pipeline

        When working with LLM-extracted data, we need a robust validation system. When an LLM tries to extract address data, three things can go wrong:
        1. Field is missing from JSON (raises TypeError)

        ```python
        # Missing Fields: LLM forgets required data
        {
            "street_name": "Elm",
            "city": "Ottawa"
        }  # Missing 'name' field -> TypeError
        ```

        2. Field is null in JSON (becomes None)
        ```python
        {
            "name": "Library",
            "street_name": "Elm",
            "city": "Ottawa",
            "postal_code": null
        }  # Optional field is null -> None
        ```
        3. Field has incorrect/hallucinated data (passes to validation)
        ```python
        {
            "name": "Library",
            "street_number": "ABC",  # Should be numeric!
            "street_name": "Elm",
            "city": "Ottawa"
        }  # Invalid data -> Validation Error

        ```
        To handle these cases, there are three levels we need to think about:


        1. **Field Validation:**Does each piece of data match its expected format?
           - Example: Is the postal code in the right format?
           - Example: Is the phone number actually a phone number?
        2. **Record Validation**: Do the fields make sense together?
            - Example: does the city match the province?
            - Example: Do the street number and name exist
        3. **Collection Validation** Do we have all the data we need?
            - Are there any duplicates?
            - Does the data make sense as a whole?


        Let's build this validation system step by step that tackles these issues. 

        ### Basic Validation
        We start with simple field validation:
        ``` python
        @dataclass(frozen=True)
        class Address:
            name: str
            street_number: str
            street_name: str
            city: str

            def __post_init__(self):
                if not self.street_number.replace(" ", "").isnumeric():
                    raise ValueError("Street number must be numeric")
        ```
        This simple validation already ensures that all required fields are present (`dataclass` handles this). Validates the street number format. Makes the address immutable (frozen=True). 

        Now lets say we need to handle optional fields, we can add them with `Optional`

        ### Adding Address Types
        Next, we want to classify addresses. We can start with a simple approach using `Literals`:

        ``` python
        @dataclass(frozen=True)
        class Address:
            name: str
            street_number: str
            street_name: str
            city: str
            address_type: Literal["residential", "commercial"]  # Start with just two types
        ```
        **When we include `address_type` in our schema, we're asking the LLM to perform a classification task. We're saying "look at this address data and determine if it's residential or commercial". We are using the LLM for both extraction AND classification in the same task.**

        But we can do even better with `Enums`. 

        ```python
        class AddressType(Enum):
            RESIDENTIAL = "residential"
            COMMERCIAL = "commercial"

            def requires_business_license(self) -> bool:
                return self == AddressType.COMMERCIAL

        @dataclass(frozen=True)
        class Address:
            name: str
            street_number: str
            street_name: str
            city: str
            postal_code: str
            address_type: AddressType
        ```

        Upgrading our address types from `Literals` to `Enums` which gives us a number of additional features:
        1. provide type safety and intellisense (IDE shows `RESIDENTIAL`, `COMMERCIAL`)
        2. create a proper namespace for related constants
        3. prevent typos that Literals might miss (with `Literals` - typos only caught at runtime)
        4. can have methods and properties


        We can also add `Optional[str]` field. 

        ``` python
        @dataclass(frozen=True)
        class Address:
            name: str
            street_number: str
            street_name: str
            city: str
            postal_code: str
            address_type: AddressType
            unit_number: Optional[str]
            building_name: Optional[str]
        ```

        The `= None` default value is a dataclass default, this is particularly important with LLMs as they might not always extract every field


        ### Comprehensive Validation
        ``` python
        @dataclass(frozen=True)
        class Address:
            name: str
            street_number: str
            street_name: str
            city: str
            postal_code: str
            address_type: AddressType
            unit_number: Optional[str]
            building_name: Optional[str]


            def __post_init__(self):
                self._validate_street_number()
                if self.postal_code is not None:
                    self._validate_postal_code()

            def _validate_street_number(self) -> None:
                if not self.street_number.replace(" ", "").isnumeric():
                    raise ValueError("Street number must be numeric")

            def _validate_postal_code(self) -> None:
                # Remove any whitespace and convert to uppercase
                postal_code = self.postal_code.strip().upper()

                # Canadian postal code pattern: A1A 1A1
                pattern = r'^[A-Z]\d[A-Z]\d[A-Z]\d$'

                if not re.match(pattern, postal_code.replace(" ", "")):
                    raise ValueError(
                        "Invalid postal code format. Must be in format 'A1A 1A1' "
                        "where A is a letter and 1 is a digit"
                    )

                # Additional Canadian postal code rules:
                if postal_code[0] in 'DFIO':  # First letter can't be D, F, I, O
                    raise ValueError("Invalid first letter in postal code")
                if postal_code[2] in 'DFIOQU':  # Third letter can't be D, F, I, O, Q, U
                    raise ValueError("Invalid third letter in postal code")
                if postal_code[4] in 'DFIOQU':  # Fifth letter can't be D, F, I, O, Q, U
                    raise ValueError("Invalid fifth letter in postal code")

        ```


        ## What We're Testing 
        Lets summarize what we're doing:
        1. We have an LLM that extracts address information from text and outputs JSON
        2. We want to validate this JSON to make sure its accurate
        3. We're using Python's type system and validation to enforce data quality
        """
    )
    return


@app.cell
def _(Addresses, Any_1, List_1, Optional_1, Self_1, dataclass, json):
    from typing import Optional, Any, Self, List
    from enum import Enum
    import re

    class AddressType(Enum):
        RESIDENTIAL = 'residential'
        COMMERCIAL = 'commercial'

        def requires_business_license(self) -> bool:
            return self == AddressType.COMMERCIAL

    @dataclass(frozen=True)
    class Address_1:
        name: str
        street_number: str
        street_name: str
        city: str
        postal_code: str
        address_type: AddressType
        unit_number: Optional_1[str]
        building_name: Optional_1[str]

        def __post_init__(self):
            self._validate_street_number()
            if self.postal_code is not None:
                self._validate_postal_code()

        def _validate_street_number(self) -> None:
            if not self.street_number.replace(' ', '').isnumeric():
                raise ValueError('Street number must be numeric')

        def _validate_postal_code(self) -> None:
            postal_code = self.postal_code.strip().upper()
            pattern = '^[A-Z]\\d[A-Z]\\d[A-Z]\\d$'
            if not re.match(pattern, postal_code.replace(' ', '')):
                raise ValueError("Invalid postal code format. Must be in format 'A1A 1A1' where A is a letter and 1 is a digit")
            if postal_code[0] in 'DFIO':
                raise ValueError('Invalid first letter in postal code')
            if postal_code[2] in 'DFIOQU':
                raise ValueError('Invalid third letter in postal code')
            if postal_code[4] in 'DFIOQU':
                raise ValueError('Invalid fifth letter in postal code')

    @dataclass(frozen=True)
    class Addresses_1:
        addresses: List_1[Address_1]

        @staticmethod
        def serialize(data: Any_1) -> Self_1:
            """Convert JSON to Addresses instance.

            [Address(**addr) for addr in json_data["addresses"]] is:
            1. Getting the list of address dictionaries with json_data["addresses"]
            2. For each dictionary, unpacking it into the Address constructor with **addr
            3. Creating a list of Address objects
            """
            _json_data = json.loads(data)
            addresses = [Address_1(**addr) for addr in _json_data['addresses']]
            return Addresses(addresses=addresses)
    test_addresses = [{'name': 'John Smith', 'street_number': '123', 'street_name': 'Maple Ave', 'city': 'Ottawa', 'postal_code': 'K1A 1A1', 'address_type': 'residential', 'unit_number': '4B', 'building_name': None}, {'name': 'Jane Doe', 'street_name': 'Oak St', 'city': 'Toronto', 'postal_code': 'M5J 2R8', 'address_type': 'residential', 'unit_number': None, 'building_name': None}, {'name': 'Tech Corp', 'street_number': 'ABC', 'street_name': 'Bay St', 'city': 'Toronto', 'postal_code': 'M5J 2R8', 'address_type': 'commercial', 'unit_number': None, 'building_name': None}, {'name': 'Library', 'street_number': '150', 'street_name': 'Elgin St', 'city': 'Ottawa', 'postal_code': 'ABC 123', 'address_type': 'commercial', 'unit_number': None, 'building_name': None}]
    for addr in test_addresses:
        try:
            address = Address_1(**addr)
            print(f'Valid: {addr['name']}')
        except (ValueError, TypeError) as e:
            print(f'Error for {addr['name']}: {str(e)}')
    return (
        AddressType,
        Address_1,
        Addresses_1,
        Any,
        Enum,
        List,
        Optional,
        Self,
        addr,
        address,
        re,
        test_addresses,
    )


@app.cell
def _(
    Address_1,
    Addresses_1,
    Callable,
    Generic,
    OpenAI,
    Optional_1,
    T_1,
    json,
    os,
    pp,
):
    _text = "\nThe Ottawa Public Library is located at 120 Metcalfe Street, Ottawa K1P 5M2. This is a large public building.\n\nSarah Wilson and her family live in unit 4B at 250 Laurier Avenue, Ottawa K1P 6M7.\n\nTechCorp's new headquarters are at 1385 Bank Street in Ottawa K1H 7M5, located in the South Keys Business Park building.\n\nThe Rideau Centre mall can be found at 50 Rideau Street, Ottawa K1N 9J7.\n\nDr. Jones runs his family practice from Suite 302 at 1919 Riverside Drive, Ottawa K1H 1A2. The medical building is called River Medical Centre.\n\nThe Blue Condo tower at 100 Bronson Ave has several units for rent, including apartment 2505. The postal code is K1R 7G6.\n"
    prompt_4 = '\nYou are address parsing assistant. \nUser provides text containing addresses.\nExtract addresses from the text.\nUse the following JSON schema to generate your response:\n\n\nFormat as JSON matching this example:\n{\n    "addresses": [\n        {\n            "name": "string (business or person name)",\n            "street_number": "string (numbers only)",\n            "street_name": "string",\n            "city": "string",\n            "postal_code": "string (Canadian format)",\n            "address_type": "residential or commercial",\n            "unit_number": "string or null",\n            "building_name": "string or null"\n        }\n    ]\n}\nRules:\n- Classify buildings like libraries, malls, and offices as commercial\n- Houses and apartments as residential\n- Include unit numbers if mentioned\n- Include building names if mentioned\n\n'

    def eval_4(prompt: str, message: str, schema: Generic[T_1], serializer: Callable=None, model: str='gpt-4o', client: OpenAI=OpenAI(api_key=os.environ['OPENAI_API_KEY'])) -> Optional_1[T_1]:
        messages = [{'role': 'system', 'content': prompt}, {'role': 'user', 'content': _message}]
        completion = client.chat.completions.create(model=model, response_format={'type': 'json_object'}, messages=messages)
        try:
            completion_data = completion.choices[0].message.content
            _json_data = json.loads(completion_data)
            if serializer is not None:
                return serializer(completion_data)
            else:
                return schema(**_json_data)
        except TypeError as type_error:
            return None
        except json.JSONDecodeError as json_error:
            return None
    res_7 = eval_4(prompt=prompt_4, message=_text, schema=Address_1, serializer=Addresses_1.serialize)
    pp(res_7)
    return eval_4, prompt_4, res_7


@app.cell
def _(Any_2, res_7):
    import pandas as pd
    from dataclasses import asdict
    from typing import Any

    def structured_data_to_df(data: Any_2) -> pd.DataFrame:
        """
        Convert any structured data (like dataclass objects) to pandas DataFrame.
        Works with nested dataclass structures where the top level contains a list/sequence.

        Example inputs:
        - Addresses(addresses=[Address(...), Address(...)])
        - Customers(customers=[Customer(...), Customer(...)])
        - Products(items=[Product(...), Product(...)])
        """
        data_list = next(iter(vars(data).values()))
        rows = [asdict(item) for item in data_list]
        return pd.DataFrame(rows)
    df = structured_data_to_df(res_7)
    df
    return Any, asdict, df, pd, structured_data_to_df


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
