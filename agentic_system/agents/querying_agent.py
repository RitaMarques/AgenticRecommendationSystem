from typing import Annotated, Literal
from langchain_core.tools import tool
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import create_react_agent
from langchain_core.messages import AIMessage
from langgraph.types import Command
from sqlalchemy import text
from datetime import datetime
from langchain.agents import create_openai_functions_agent
from langchain.agents import AgentExecutor
from langchain_core.prompts import PromptTemplate

from agentic_system.utils.llm import tool_llm
from agentic_system.db.db_conn import session_scope
from agentic_system.utils.utils import generate_embeddings, SimpleState


###############################################################################
# Tools
###############################################################################

@tool
def distinct_products_tool() -> Annotated[list[str], "A list of unique product names"]:
    
    """Fetches all unique products from products table."""
    print("Fetching distinct products from products table...")

    query = """
        SELECT DISTINCT name AS product FROM dbo.products
    """
    sql = text(query)
    
    with session_scope() as session:
        results = session.execute(sql).fetchall()

    # Flatten the results into a Python list
    products = [row[0] for row in results]

    print(f"Found {len(products)} distinct products: {products}.")
    return products


@tool
def cooccurrences_query_tool(
    product_name: Annotated[str, "Exact product name for co-occurrence lookup (match 'name' column in products table)"],
    limit: Annotated[int, "Number of co-ocurrences to return"]=15,
)-> Annotated[list, "List of co-occurring products with counts. Each item has: product1, product2, cooccurrence_count"]:
    
    """
    Queries the `dbo.cooccurrences` table to get the top related products
    given a product_name.

    Parameters
    ----------
    product_name : str
        Exact match of product name to search for (matches `product1` or `product2` column).
    limit : int
        Number of results to return (default: 5).

    Returns
    -------
    list of dict
        Each dict contains: product1, product2, cooccurrence_count.

    Example input:
    {
        "product_name": "The Legend of Zelda: Breath of the Wild",
        "limit": 15
    }
    """
    print("Querying co-occurrences...")

    sql = text(f"""
            SELECT product1, product2, cooccurrence_count
            FROM dbo.cooccurrences
            WHERE product1 = :p OR product2 = :p
            ORDER BY cooccurrence_count DESC
            LIMIT :limit
        """)
    
    with session_scope() as session:

        session.query()
        rows = session.execute(
            sql, 
            {
                "p": product_name, 
                "limit": limit
            }
        ).mappings().all()

        print([dict(r) for r in rows])
        return [dict(r) for r in rows]


@tool
def product_search_tool(
    query: Annotated[str, "The search query to find Nintendo Switch products"]
) -> Annotated[list, "List of matching products with distance score"]:
    """
    Searches the Nintendo Switch product database.

    This tool is helpful to know detailed information about products, their categories and specificities. Here you can also find in which stores products are sold.

    Parameters
    ----------
    query : str
        Search query text to be converted into an embedding.

    Returns
    -------
    list of dict
        Each dict contains all columns from dbo.products plus a `distance` score.

    Example input:
    {
        "query": "Mario party multiplayer game"
    }
    """
    print("Running embedding search...")

    _, embedding_vector = generate_embeddings(query)
    
    # operator <=> for cosine distance
    sql = text(f"""
        SELECT name, release_date, times_sold, store_a, store_b, store_c, type, category, franchise, min_age, major_category
        FROM dbo.products
        ORDER BY embedding <#> :embedding_vector ASC
        LIMIT 10
    """)
        
    with session_scope() as session:
        results = session.execute(
            sql, 
            {"embedding_vector": str(embedding_vector)}
        ).mappings().all()

        print([dict(r) for r in results])
        return [dict(r) for r in results]


###############################################################################
# Agent
###############################################################################

prompt_template = PromptTemplate(
    input_variables=["input", "agent_scratchpad"],
    template="""You are a database querying agent for Nintendo Switch product recommendations.

    Database schema:
    You have two available dataset objects

    - Products: Product information, sold quantities per store
    dbo.products(
        id, name, release_date, times_sold, store_a, store_b, store_c,
        type, category, franchise, min_age, major_category, text, tokens, embedding
    )

    - Co-occurrences: Frequency that two products were sold together
    dbo.cooccurrences(
        id, product1, product2, cooccurrence_count
    )

    Available tools:
    1. product_search_tool(query: str)
    - Performs semantic similarity search on dbo.products available at specific stores.
    - query is free text describing what the user wants.
    - This tool is useful to get to know the specifities of the products and which stores sell those products.

    2. cooccurrences_query_tool(product_name: str, limit: int=15)
        - Finds products frequently bought together with the given product.
        - product_name must match exactly a value from dbo.products.name.
        - limit sets how many rows to return.

    Your task:
    - Read the user request
    - Decide which tools to call and in which order to collect the maximum relevant data.
    - You can call multiple tools if needed.
    - When calling multiple tools, output a JSON array of tool calls exactly as:
    [
    {{ "name": "product_search_tool", "arguments": {{"query": "..."}} }},
    {{ "name": "cooccurrences_query_tool", "arguments": {{ "product_name": "..."}} }}
    ]

    Return the combined result as a JSON object with keys "products" and "cooccurrences".
    {{
      "products": [... results from product_search_tool if called ...],
      "cooccurrences": [{{'productX:' [... results from cooccurrences_query_tool if called ...]}}]
    }}
    - If a tool is not called, return an empty list for that field.
    - Always output valid JSON — no extra text, no trailing commas.

    Important rules:
    - Maximize information coverage: combine semantic search and co-occurrence data whenever it can improve recommendations.
    - When recommending, retrieving co-occurrences is highly relevant to get the top related products.

    User query:
    {input}

    {agent_scratchpad}
    """
)

# agent supports multiple tool calls per LLM output
querying_agent = create_openai_functions_agent(
    tool_llm,
    tools=[product_search_tool, cooccurrences_query_tool],
    prompt=prompt_template
)

###############################################################################
# Node
###############################################################################

agent_executor = AgentExecutor(
    agent=querying_agent,
    tools=[product_search_tool, cooccurrences_query_tool],
    max_iterations=5,
    verbose=True,
    handle_parsing_errors=True
)

def querying_node(simple_state: SimpleState) -> Command[Literal["recommendation_supervisor_node"]]:
    print(f"Querying Node: {datetime.now()}")

    #result = querying_agent.invoke(simple_state)
    result = agent_executor.invoke({"input": simple_state["messages"][-1].content})
    
    return Command(
        update={
            "messages": [
                AIMessage (content=result["output"], name="querying_node")  
            ]
        },
        goto="recommendation_supervisor_node",
    )