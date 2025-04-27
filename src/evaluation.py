import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score

class Evaluator:
    def __init__(self, ground_truth, predictions):
        self.ground_truth = ground_truth
        self.predictions = predictions

    def precision(self):
        return precision_score(self.ground_truth, self.predictions, average='macro')

    def recall(self):
        return recall_score(self.ground_truth, self.predictions, average='macro')

    def f1(self):
        return f1_score(self.ground_truth, self.predictions, average='macro')

    def evaluate(self):
        return {
            "precision": self.precision(),
            "recall": self.recall(),
            "f1_score": self.f1()
        }