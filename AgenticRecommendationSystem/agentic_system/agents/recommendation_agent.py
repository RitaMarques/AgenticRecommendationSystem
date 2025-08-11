from typing import Annotated, Literal, List
from langchain_core.tools import tool
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import create_react_agent
from langchain_core.messages import AIMessage
from langgraph.graph import END
from langgraph.types import Command
from datetime import datetime


from agentic_system.utils.llm import tool_llm, infer_llm
from agentic_system.utils.utils import QueriesOutputs, SimpleState


@tool
def recommendation_engine_tool(
    query: Annotated[str, "The usr query"],
    queried_info: Annotated[List[QueriesOutputs], "Queried product data"]
) -> Annotated[str, "Product(s) recommendation"]:
    """Uses the background and research sources to write a newspaper article with the given headeline"""

    print ("Writing recommendation")

    messages = [
        ("system", """You are an experienced Nintendo Switch product recommendation specialist. Your role is to interpret user queries accurately and provide tailored, data-driven recommendations for Nintendo Switch consoles, games, and accessories.
         
        The information you need to answer the user query is the queried data from the database.
        Provide a tailored product recommendation.
        """),
        (
            "AIMessage",
            f"""
            Queried data: {queried_info}
            User query: {query}
            """,
        ),
    ]
    response = infer_llm.invoke(messages)
    print ("RESP", response)
    return response


recommendation_agent = create_react_agent(
    tool_llm, 
    tools=[recommendation_engine_tool], 
    checkpointer = MemorySaver(),
    prompt="""
    You are an experienced Nintendo Switch product recommendation specialist.
    Your role is to interpret user requests accurately and provide tailored, data-driven product recommendations for Nintendo Switch products.

    Data usage rules:
    - You may only recommend products that appear in the queried database results provided to you.
    - You must only use the information contained in the retrieved data — no external knowledge, no assumptions, no made-up details.
    - If the requested type of product is not present in the retrieved data, state that no suitable recommendations are available.

    Recommendation guidelines:
    - Provide recommendations that are directly relevant to the user’s expressed needs, preferences, or constraints.
    - Prioritize accuracy, clarity, and relevance over quantity — better to give fewer, more targeted suggestions than a long generic list.
    - Use only attributes present in the retrieved data (e.g., name, type, category, franchise, release_date, min_age, times_sold) to justify your recommendations.
    - When possible, explain briefly why each recommended product is a good fit, using only retrieved attributes.
    - If multiple suitable products are found, order them logically (e.g., by relevance, popularity, or release date) based on available fields in the retrieved data.

    Output requirements:
    - Output only the product recommendation(s) — no extra commentary or meta-text.
    - Do not include raw database rows or unrelated products.
    - Use clear, natural language suitable for a customer-facing recommendation.
    """,
    name='recommendation_agent'
)

def recommendation_specialist_node(simple_state: SimpleState) -> Command[Literal["recommendation_supervisor_node", END]]:
    print(f"Supervisor Node: {datetime.now()}")

    result = recommendation_agent.invoke(simple_state)
    
    return Command(
        update={
            "messages": [
                AIMessage (content=result["messages"][-1].content, name="recommendation_specialist_node")  
            ]
        },
        goto=END,
    )