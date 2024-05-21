import json
from openai import OpenAI
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
from dotenv import load_dotenv
import os
load_dotenv()

#insert your Open AI API key here.
open_api_key = os.getenv('OPENAI_API_KEY')

#Setup
client = OpenAI(api_key = open_api_key)
embedding_model = OpenAIEmbeddings(openai_api_key = open_api_key)
metadata_id = 0

#Initializing Chroma DB
placeholder = ""
doc = Document(page_content=placeholder, metadata={"_id_": f"{metadata_id}"})
db = Chroma.from_documents([doc], embedding_model)

#VectorDB Utils
def store_text(text):
    global metadata_id
    current_metadata = metadata_id
    metadata = {"id":f"{current_metadata}"}
    metadata_id +=1

    doc = Document(page_content=text, metadata=metadata)
    db.add_documents([doc])

def query_text(query, num_suggestions = 4):
    global db
    search_result = db.similarity_search(query=query, k=num_suggestions)
    return search_result

#Response Generation LLM Calls (3 Types)
STORAGE_SYSTEM_MESSAGE = {
"role":"system",
"content":"""
You are a helpful Journal Assistant that generates an acknowledgement notification about information that you have been told to remember or store.
You will be given an input and you need to generate a suitable acknowledgement notification output for this input. Your output has to have a reference
to the input text, but don't replicate the input text as it is.
"""
}

storage_few_shot_examples = [{"role":"user","content":"Remind me to buy eggs when I'm at the supermarket next."},
                     {"role":"assistant","content":"Sure, I'll remind you about buying eggs when you visit the supermarket next."},
                     {"role":"user","content":"I need to be reminded to do my math assignment and book my flight tickets when I open my laptop next."},
                     {"role":"assistant","content":"The next time you open your laptop, I'll be sure to remind you about your assignment and flight tickets!"}]
storage_messages = [STORAGE_SYSTEM_MESSAGE]
storage_messages.extend(storage_few_shot_examples)

def generate_llm_storage_response(text):
    global client
    global storage_messages

    messages = storage_messages.copy()
    messages.append({"role":"user", "content":text})

    response = client.chat.completions.create(
    model="gpt-3.5-turbo-0125",  
    messages=messages
    )

    return response    


IRRELEVANT_SYSTEM_MESSAGE = {
"role":"system",
"content":"""
You are a helpful Journal Assistant that generates an error message when given an input that is unrelated to your purpose as a journal app.
You will be given an input and you need to generate a suitable error notification output for this input. Your output has to have a reference to the fact
that you are just a journal application, the domain of the input, and how the domain of the input is greatly different from the expertise of a journal
application. 
"""
}

irrelevant_few_shot_examples = [{"role":"user","content":"What's 2+2?"},
                     {"role":"assistant","content":"I'm just a journal application, and cannot perform mathematical calculations."},
                     {"role":"user","content":"Who is the president of India?"},
                     {"role":"assistant","content":"I do not have extensive general knowledge, because I'm just a journal application."}]
irrelevant_messages = [IRRELEVANT_SYSTEM_MESSAGE]
irrelevant_messages.extend(irrelevant_few_shot_examples)

def generate_llm_irrelevant_response(text):
    global client
    global irrelevant_messages

    messages = irrelevant_messages.copy()
    messages.append({"role":"user", "content":text})
    
    response = client.chat.completions.create(
    model="gpt-3.5-turbo-0125",  
    messages=messages
    )

    return response 


RETRIEVAL_SYSTEM_MESSAGE = {
"role":"system",
"content":"""
You are a helpful Journal Assistant that generates a description o
You will be given multiple inputs-
1. The first will be a Query.
2. Some of the following inputs will be sentences related to the query- they will be the answer to the query. 
   Some of the other inputs will not be related to the query-they will not be the answer to the query.

You need to generate an output which combines all the input sentences which are related to the input query and presents them as an 
answer to the input query. You must not include the unrelated input sentences in your answer to the input query.
You could have a large number of such input sentences. You need to judge which input sentences are related and should be included in your 
answer to the input query, and which input sentences are not related and shouldn't be included in your answer.

Here are some examples:
-<inputQuery>I'm at the supermarket now. What should I buy?</inputQuery>
 <inputSentence>Remind me to buy eggs when I'm at the supermarket next.</inputSentence> 
 <inputSentence>I need to buy a birthday card for my mom from the supermarket.</inputSentence>
 <inputSentence>Remind me to get a haircut tomorrow.</inputSentence>
  Response- "You should buy eggs, and a birthday card for your mom from the supermarket."

-<inputQuery>I just opened my laptop, what should I do?</inputQuery>
 <inputSentence>Remind me to email the job recruiter at Google, when I'm on my laptop next.</inputSentence>
 <inputSentence>Remind me about that meeting I have with John tomorrow.</inputSentence>
 <inputSentence>I need to work on my Math Assignment and book flight tickets on my laptop</inputSentence>
  Response- "You should email the job recruiter at Google, work on your Math Assignment and book flight tickets using your laptop."
"""
}

def generate_llm_retrieval_response(query, search_results):
    global RETRIEVAL_SYSTEM_MESSAGE
    global client

    query_message ={"role": "user", "content": query}
    search_messages = [{"role": "user", "content": search_result.page_content} for search_result in search_results]

    messages = [RETRIEVAL_SYSTEM_MESSAGE]
    messages.append(query_message)

    for search_message in search_messages:
        messages.append(search_message)
    
    response = client.chat.completions.create(
    model="gpt-3.5-turbo-0125",  
    messages=messages)

    return response 

#Functions Using Classifier Output to Make Appropriate LLM Call, and Generate Output
def store_information(text):
    store_text(text)
    response = generate_llm_storage_response(text)
    return response

def retrieve_information(query):
    search_results = query_text(query)
    response = generate_llm_retrieval_response(query, search_results)
    return response

def handle_irrelevant(text):
    response = generate_llm_irrelevant_response(text)
    return response

# Dispatcher function
def call_function(args,name):
    if name == "store":
        return store_information(args['text'])
    elif name == "retrieve":
        return retrieve_information(args['text'])
    elif name == "irrelevant":
        return handle_irrelevant(args['text'])
    else:
        raise ValueError("Unknown function")

#Classifier LLM (outputting a JSON with Appropriate Function Arguments)
PROMPT_CLASSIFIER_SYSTEM_MESSAGE = {
    "role": "system",
    "content": """
You are a journal app. You can either store the query in your memory or retrieve information from your memory.

You will strore informations like grocery lists, reminders, incidents, future tasks, daily entries, dreams, things that you need to do in the future etc.

If you get a query that could ask for a information that should be stored in a journal app, you should say that the query is for retrieval.
If the query is not asking general information, but asking for tasks, past incidents, future plans, etc., you should say that the query is for retrieval.

If you feel the query is irrelevant for a journal app, you can say that the query is irrelevant.                                          
                                                 
Read the query carefully and think whether a journal app should store the query, retrieve information from the query, or if the query is irrelevant.


Here are some examples:

1. Storage:
- "Remind me to buy eggs when I'm at the supermarket next."
- "Note that I have a meeting with John at 3 PM tomorrow."
- "I was stuck in traffic today for a very long time."

2. Retrieval:
- "I'm at the supermarket now. What should I buy?"
- "What meetings do I have scheduled for tomorrow?"
- "What's on my to-do list today?"

3. Irrelevant:
- "What's 2+2?"
- "Tell me a joke."
- "What is the capital of Delhi"
"""
}


def generate_classifier_response(prompt):
    tools = [
        {
            "type":"function",
            "function":{
            "name": "store",
            "description": "Stores the input text for future retrieval.",
            "parameters": {
                "type": "object",
                "properties": {
                    "text": {"type": "string", "description": "The actual text to store"}
                },
                "required": ["text"]
            }
            }
        },
        {
            "type":"function",
            "function":{
            "name": "retrieve",
            "description": "Retrieves previously stored information based on the input text.",
            "parameters": {
                "type": "object",
                "properties": {
                    "text": {"type": "string", "description": "The input text to retrieve information for"}
                },
                "required": ["text"]
            }
            }
        },
        {   
            "type":"function",
            "function":{
            "name": "irrelevant",
            "description": "Handles inputs that are irrelevant to storage or retrieval. If you feel it is not store or retrieve, you can say that the input is irrelevant.",
            "parameters": {
                "type": "object",
                "properties": {
                    "text": {"type": "string", "description": "The irrelevant input text"}
                },
                "required": ["text"]
            }
            }
        }
    ]

    tool_call = "required"

    global PROMPT_CLASSIFIER_SYSTEM_MESSAGE

    response = client.chat.completions.create(
        model="gpt-3.5-turbo-0125",  
        messages=[
            PROMPT_CLASSIFIER_SYSTEM_MESSAGE,
            {"role": "user", "content": prompt}
        ],
        tools=tools,
        tool_choice=tool_call    
    )

    return response.choices[0].message.tool_calls[0].function

#Generating Suitable Response Depending on Classifier Output
def generate_overall_model_response(prompt):
    classifier_response = generate_classifier_response(prompt)
    
    if classifier_response:
        arguments = classifier_response.arguments
        name = classifier_response.name
        arguments = json.loads(arguments)
        result = call_function(arguments, name)
        final_response_text = result.choices[0].message.content
        
        return final_response_text
    else:
        error_message = "No function call was made by the model."
        display_message = "Sorry, I'm unable to answer your request at this moment."
        return display_message