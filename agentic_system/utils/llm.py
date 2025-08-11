import os
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()

# LLM Definitions
# llm for tool selector (1st connection)
tool_llm = ChatOpenAI(
    model=os.environ["OPENAI_TOOL_MODEL"],
    api_key=os.environ["OPENAPI_KEY"],
    max_tokens=1000,
    temperature=0,
)

# llm for inference (2nd connection)
infer_llm = ChatOpenAI(
    model=os.environ["OPENAI_INFER_MODEL"],
    api_key=os.environ["OPENAPI_KEY"],
    max_tokens=1000,
    temperature=0.2,
)