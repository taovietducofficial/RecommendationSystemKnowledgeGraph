import networkx as nx
import matplotlib.pyplot as plt

def visualize_graph(graph, node_color='lightblue', edge_color='gray', node_size=500, font_size=10):
    pos = nx.spring_layout(graph)
    edge_labels = nx.get_edge_attributes(graph, 'label')
    nx.draw(
        graph, pos, with_labels=True, node_color=node_color, edge_color=edge_color,
        node_size=node_size, font_size=font_size
    )
    nx.draw_networkx_edge_labels(graph, pos, edge_labels=edge_labels, font_size=font_size)
    plt.show()

def visualize_embeddings(embeddings, labels=None):
    from sklearn.decomposition import PCA
    import numpy as np

    pca = PCA(n_components=2)
    reduced_embeddings = pca.fit_transform(embeddings)
    x = reduced_embeddings[:, 0]
    y = reduced_embeddings[:, 1]

    plt.figure(figsize=(8, 6))
    plt.scatter(x, y, c='blue', alpha=0.7, label='Nodes')
    if labels is not None:
        for i, label in enumerate(labels):
            plt.annotate(label, (x[i], y[i]), fontsize=8, alpha=0.75)
    plt.title("Node Embeddings Visualization")
    plt.xlabel("PCA Component 1")
    plt.ylabel("PCA Component 2")
    plt.legend()
    plt.show()