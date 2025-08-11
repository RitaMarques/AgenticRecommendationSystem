from typing import Literal
from typing_extensions import TypedDict
from langchain_core.messages import AIMessage
from langgraph.graph import END
from langgraph.types import Command
from datetime import datetime

from agentic_system.utils.llm import tool_llm, infer_llm
from agentic_system.utils.utils import SimpleState


recommendation_system_members = ["querying_node", "recommendation_specialist_node"]
recommendation_system_options = recommendation_system_members + ["FINISH"]

system_prompt = f"""
You are a specialized Recommendation Engine Supervisor for Nintendo Switch products.

Your role:
1. Interpret the user’s request.
2. Plan and coordinate the actions of your available workers to produce accurate, tailored Nintendo Switch product recommendations that meet all user constraints.

Rules:
- Workers available to you are: {recommendation_system_options}.
- Each worker has a specific, clearly defined task and may only be used for that task.
- All recommendations must be based solely on the data returned by the querying agent — no external knowledge, no guessing.

Strict scope guardrails:
- You may only handle requests directly related to Nintendo Switch products.
- Do not answer or process questions unrelated to Nintendo Switch products.
- If the request is outside scope, respond poletly saying that you are not allowed to answer.
- You must never provide information, opinions, or recommendations outside the queried database data.

When you have a final response, return FINISH.
"""

class Router(TypedDict):
    """Worker to route to next. If there is a headline, route to FINISH."""
    next: Literal[*recommendation_system_options]


def recommendation_supervisor_node(state: SimpleState) -> Command[Literal[*recommendation_system_members, END]]:
    print('STATE MESSAGES')
    print(state["messages"])
    print('\n')

    messages = [
        {"role": "system", "content": system_prompt},
    ] + state["messages"]

    print(f"Supervisor: {datetime.now()}")
    print(f"len(messages): {len(messages)}")
    print ("==========================================\n",messages[-1],"\n\n")

    response = tool_llm.with_structured_output(Router).invoke(messages)

    print ("\nRESPONSE",response,"\n")

    # goto = response["next"]
    # if goto == "FINISH":
    #     for item in reversed(messages):
    #         if isinstance(item, AIMessage):
    #             print('IS AI MESSAGE')
    #             print(item)
    #             break
    #     goto = END

    # return Command(goto=goto, update={"next": goto})

    goto = response["next"]
    if goto == "FINISH":
        if len(messages) == 2:
            user_message = state["messages"][-1].content
        
            refusal_prompt = f"""
            You are a friendly AI assistant that can only answer questions about Nintendo Switch products.
            The user asked: "{user_message}"
            Politely explain in natural language that you cannot answer because it is outside your scope,
            and (optionally) invite them to ask something about Nintendo Switch products.
            """
            
            refusal_text = infer_llm.invoke([{"role": "system", "content": refusal_prompt}]).content
            goto=END

            return Command(
                goto=goto,
                update={
                    "messages": state["messages"] + [AIMessage(content=refusal_text)],
                    "next": goto
                }
            )
        else:
            goto = END
        
    return Command(goto=goto, update={"next": goto})