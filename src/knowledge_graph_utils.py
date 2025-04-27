import networkx as nx

def load_graph_from_triples(triples):
    graph = nx.DiGraph()
    for subject, predicate, obj in triples:
        graph.add_edge(subject, obj, label=predicate)
    return graph

def get_neighbors(graph, node):
    return list(graph.neighbors(node))

def get_edge_label(graph, source, target):
    if graph.has_edge(source, target):
        return graph[source][target].get('label', None)
    return None

def shortest_path(graph, source, target):
    try:
        return nx.shortest_path(graph, source=source, target=target)
    except nx.NetworkXNoPath:
        return None