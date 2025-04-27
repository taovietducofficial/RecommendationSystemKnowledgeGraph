import pandas as pd
import networkx as nx

class DataPreprocessor:
    def __init__(self, data_path):
        self.data_path = data_path
        self.raw_data = None
        self.graph_data = None

    def load_data(self):
        try:
            self.raw_data = pd.read_csv(self.data_path)
            print("Dữ liệu đã được tải thành công.")
        except Exception as e:
            print(f"Lỗi khi tải dữ liệu: {e}")

    def preprocess_data(self):
        if self.raw_data is None:
            raise ValueError("Dữ liệu chưa được tải. Vui lòng gọi load_data() trước.")
        self.graph_data = []
        for _, row in self.raw_data.iterrows():
            subject = row['subject']
            predicate = row['predicate']
            obj = row['object']
            self.graph_data.append((subject, predicate, obj))
        print("Dữ liệu đã được tiền xử lý.")

    def build_graph(self):
        if self.graph_data is None:
            raise ValueError("Dữ liệu chưa được tiền xử lý. Vui lòng gọi preprocess_data() trước.")
        graph = nx.DiGraph()
        for subject, predicate, obj in self.graph_data:
            graph.add_edge(subject, obj, label=predicate)
        print("Đồ thị tri thức đã được xây dựng.")
        return graph