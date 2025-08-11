import os
from openai import OpenAI
from langgraph.graph import MessagesState
from typing_extensions import TypedDict

###############################################################################
# Utils Functions
###############################################################################

def generate_embeddings(text:str) -> tuple[int, list]:
    client = OpenAI(api_key=os.environ["OPENAPI_KEY"])

    response = client.embeddings.create(
        input=text,
        model=os.environ["OPENAI_EMB_MODEL"]
    )

    tokens = response.usage.total_tokens
    embedding = response.data[0].embedding
    
    return tokens, embedding


###############################################################################
# Utils Classes
###############################################################################

class SimpleState(MessagesState):
    next: str

class QueriesOutputs(TypedDict):
    products_query_output: list
    coocurrences_query_output: list



