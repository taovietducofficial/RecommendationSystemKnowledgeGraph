from data_preprocessing import preprocess_data
from embedding_model import generate_embeddings
from knn_based_recommender import knn_recommend
from utils import load_items
from evaluation import evaluate_recommendations

def main():
    # Step 1: Load and preprocess data
    print("Loading and preprocessing data...")
    items = load_items()
    preprocessed_data = preprocess_data(items)

    # Step 2: Generate embeddings
    print("Generating embeddings...")
    embeddings = generate_embeddings(preprocessed_data)

    # Step 3: Get user input and make recommendations
    user_input = input("Enter your preferences: ")
    print("Generating recommendations...")
    recommendations = knn_recommend(user_input, embeddings, items)

    # Step 4: Evaluate recommendations (optional)
    print("Evaluating recommendations...")
    evaluation_score = evaluate_recommendations(recommendations)
    print(f"Evaluation Score: {evaluation_score}")

    # Step 5: Display recommendations
    print("Recommendations:")
    for rec in recommendations:
        print(f"{rec['name']} - {rec['description']}")

if __name__ == "__main__":
    main()