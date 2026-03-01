from typing import Dict, List
import networkx as nx


def has_circular_foreign_key(table_with_foreign_keys: Dict[str, List[str]]) -> bool:
    G: nx.DiGraph[str] = nx.DiGraph()
    for table_name, foreign_keys in table_with_foreign_keys.items():
        for foreign_key in foreign_keys:
            G.add_edge(table_name, foreign_key)

    try:
        # This will throw if it does not find a cycle
        nx.find_cycle(G=G)  # type: ignore

        return True
    except:
        return False
