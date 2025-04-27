import numpy as np
from sklearn.decomposition import PCA

class EmbeddingModel:
    def __init__(self, graph):
        self.graph = graph
        self.embeddings = None

    def generate_embeddings(self, dimensions=128):
        nodes = list(self.graph.nodes)
        adjacency_matrix = np.zeros((len(nodes), len(nodes)))
        node_index = {node: idx for idx, node in enumerate(nodes)}
        for edge in self.graph.edges(data=True):
            source = node_index[edge[0]]
            target = node_index[edge[1]]
            adjacency_matrix[source][target] = 1
        pca = PCA(n_components=dimensions)
        self.embeddings = pca.fit_transform(adjacency_matrix)

    def get_embedding(self, node):
        if self.embeddings is None:
            raise ValueError("Embeddings chưa được tạo. Vui lòng gọi generate_embeddings() trước.")
        nodes = list(self.graph.nodes)
        node_index = {node: idx for idx, node in enumerate(nodes)}
        if node not in node_index:
            raise ValueError("Node không tồn tại trong đồ thị.")
        return self.embeddings[node_index[node]]