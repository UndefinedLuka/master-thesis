import random
import string
from typing import Dict, List
import networkx as nx



def construct_database_graph(table_with_foreign_keys: Dict[str, List[str]]) -> nx.DiGraph[str]:
    graph: nx.DiGraph[str] = nx.DiGraph()
    for table_name, foreign_keys in table_with_foreign_keys.items():
        for foreign_key in foreign_keys:
            graph.add_edge(table_name, foreign_key)

    return graph

def has_circular_foreign_key(table_with_foreign_keys: Dict[str, List[str]]) -> bool:
    graph = construct_database_graph(table_with_foreign_keys=table_with_foreign_keys)

    try:
        # This will throw if it does not find a cycle
        nx.find_cycle(G=graph)  # type: ignore

        return True
    except:
        return False

def find_root_node(table_with_foreign_keys: Dict[str, List[str]]) -> List[str]:
    graph = construct_database_graph(table_with_foreign_keys=table_with_foreign_keys)

    # 2. Identify the Hub (Highest In-Degree)
    in_degrees = dict(graph.degree())
    root = max(in_degrees, key=in_degrees.get)

    return [root]
