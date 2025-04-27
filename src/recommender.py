from data_preprocessing import DataPreprocessor
from graph_building import GraphBuilder
from embedding_model import EmbeddingModel
from knn_based_recommender import KNNRecommender

class KnowledgeGraphRecommender:
    def __init__(self, data_path, embedding_dimensions=128, n_neighbors=5):
        self.data_path = data_path
        self.embedding_dimensions = embedding_dimensions
        self.n_neighbors = n_neighbors
        self.graph = None
        self.embeddings = None
        self.recommender = None

    def build_knowledge_graph(self):
        preprocessor = DataPreprocessor(self.data_path)
        preprocessor.load_data()
        preprocessor.preprocess_data()
        builder = GraphBuilder()
        builder.add_triples(preprocessor.graph_data)
        self.graph = builder.get_graph()

    def generate_embeddings(self):
        model = EmbeddingModel(self.graph)
        model.generate_embeddings(dimensions=self.embedding_dimensions)
        self.embeddings = model.embeddings

    def setup_recommender(self):
        self.recommender = KNNRecommender(self.embeddings, n_neighbors=self.n_neighbors)
        self.recommender.fit()

    def recommend(self, node):
        if self.recommender is None:
            raise ValueError("Recommender chưa được thiết lập. Vui lòng gọi setup_recommender() trước.")
        return self.recommender.recommend(node)