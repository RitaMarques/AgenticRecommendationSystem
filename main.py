import sys
from dotenv import load_dotenv
from langgraph.graph import StateGraph, START

from agentic_system.utils.utils import SimpleState
from agentic_system.agents.querying_agent import querying_node
from agentic_system.agents.recommendation_agent import recommendation_specialist_node
from agentic_system.agents.supervisor_agent import recommendation_supervisor_node



###############################################################################
# Recommendation System Builder (Graph)
###############################################################################

def system_builder_graph():
    system_builder = StateGraph(SimpleState)
    system_builder.add_edge(START, "recommendation_supervisor_node")
    system_builder.add_node("recommendation_supervisor_node", recommendation_supervisor_node)
    system_builder.add_node("querying_node", querying_node)
    system_builder.add_node("recommendation_specialist_node", recommendation_specialist_node)

    system_builder_graph = system_builder.compile()

    #print(system_builder_graph.get_graph().draw_ascii())

    return system_builder_graph 


###############################################################################
# Call Agent
###############################################################################
 
def call_recommendation_system(user_query):

    # user_query ="I want to know how many zelda games are because I want to start a collection"

    inputs = {"messages": [("user", f"{user_query}")]}

    sbg_instance = system_builder_graph()
    graph_output = sbg_instance.invoke(inputs)

    #print ("DONE")

    final_msg = graph_output["messages"][-1] 

    #print(f"FINAL MESSAGE: \n {final_msg.content}")
    return final_msg


if __name__ == "__main__":

    user_query = sys.argv[1]
    
    print(f"User query received: {user_query}")
    print("\n\n--------------------------------debug prints--------------------------------\n")

    answer = call_recommendation_system(user_query)
    
    print("\n\n--------------------------------ended debug prints--------------------------------\n")
    print(f"Answer: {answer.content}")

    # user_query = "I want a pepperoni pizza with extra cheese please."
    # call_recommendation_system(user_query)

    #user_query = "I want to buy a game for my nephew, at Store A, who is 5 years old. We loved Super Mario Odyssey, but I cannot buy a game from this family as he already has all Super Mario games."
    #call_recommendation_system(user_query)

