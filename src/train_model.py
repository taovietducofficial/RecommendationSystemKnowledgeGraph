from recommender import KnowledgeGraphRecommender

def train_knowledge_graph_recommender(data_path, embedding_dimensions=128, n_neighbors=5):
    recommender = KnowledgeGraphRecommender(
        data_path=data_path,
        embedding_dimensions=embedding_dimensions,
        n_neighbors=n_neighbors
    )
    recommender.build_knowledge_graph()
    recommender.generate_embeddings()
    recommender.setup_recommender()
    return recommender

if __name__ == "__main__":
    data_path = "data/knowledge_graph_data.csv"
    embedding_dimensions = 128
    n_neighbors = 5

    recommender = train_knowledge_graph_recommender(data_path, embedding_dimensions, n_neighbors)
    print("Hệ thống gợi ý đã được huấn luyện thành công.")