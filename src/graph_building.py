import networkx as nx

class GraphBuilder:
    def __init__(self):
        self.graph = nx.DiGraph()

    def add_triple(self, subject, predicate, obj):
        self.graph.add_edge(subject, obj, label=predicate)

    def add_triples(self, triples):
        for subject, predicate, obj in triples:
            self.add_triple(subject, predicate, obj)

    def get_graph(self):
        return self.graph