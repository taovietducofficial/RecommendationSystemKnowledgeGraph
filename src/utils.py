import json
import networkx as nx
import numpy as np

def save_graph(graph, file_path):
    data = nx.node_link_data(graph)
    with open(file_path, 'w') as f:
        json.dump(data, f)

def load_graph(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)
    return nx.node_link_graph(data)

def save_embeddings(embeddings, file_path):
    np.savetxt(file_path, embeddings, delimiter=',')

def load_embeddings(file_path):
    return np.loadtxt(file_path, delimiter=',')

def save_recommendations(recommendations, file_path):
    with open(file_path, 'w') as f:
        json.dump(recommendations, f)

def load_recommendations(file_path):
    with open(file_path, 'r') as f:
        return json.load(f)