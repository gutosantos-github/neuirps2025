import os
import pickle
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split

# Import your dataset classes
from datasets_classes.unsw_nb15_class import Unsw_nb15
from datasets_classes.cic_ids_2017_class import Cic_ids_2017
from datasets_classes.bccc_cic_ids2017_class import Bccc_cic_ids2017
from datasets_classes.botNetIot_L01_class import BotNetIot

# Define datasets
datasets = {
    # "unsw_nb15": Unsw_nb15(),
    # "cic_ids_2017": Cic_ids_2017(),
    "bccc_cic_ids2017": Bccc_cic_ids2017(),
    "botNetIot": BotNetIot()
}

# Machine Learning Models
ml_models = {
    "GaussianNB": GaussianNB(),
    "RandomForest": RandomForestClassifier(max_depth=10, random_state=100),
    "AdaBoost": AdaBoostClassifier(),
    "LogisticRegression": LogisticRegression(),
    "DecisionTree": DecisionTreeClassifier(criterion="gini", random_state=100, max_depth=10, min_samples_leaf=5),
    "KNN": KNeighborsClassifier(n_neighbors=5)
}


def get_ml_model_size(model):
    """Returns the size of a scikit-learn model in MB."""
    return len(pickle.dumps(model)) / (1024 ** 2)


# Create output directory
os.makedirs("model_sizes", exist_ok=True)

# Evaluate ML models
for dataset_name, dataset in datasets.items():
    print(f"\n--- Dataset: {dataset_name} ---")

    try:
        # Get features and labels
        features = dataset.features
        labels = dataset.labels

        # Convert to numpy arrays if they're pandas objects
        if isinstance(features, pd.DataFrame):
            features = features.values
        if isinstance(labels, pd.Series):
            labels = labels.values

        # Print shapes for debugging
        print(f"Features shape: {features.shape}")
        print(f"Labels shape: {labels.shape}")

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            features, labels, test_size=0.2, random_state=42
        )

        with open(f"model_sizes/{dataset_name}_ml_model_sizes.txt", "w") as f:
            for model_name, model in ml_models.items():
                model.fit(X_train, y_train)
                size = get_ml_model_size(model)

                f.write(f"\nModel: {model_name}\n")
                f.write(f"Total model size: {size:.6f} MB\n")
                print(f"{model_name}: {size:.6f} MB")

    except Exception as e:
        print(f"Error processing {dataset_name}: {str(e)}")
        print("Available data attributes:")
        print(f"- features: {type(dataset.features)} (shape: {dataset.features.shape})")
        print(f"- labels: {type(dataset.labels)} (shape: {dataset.labels.shape})")
        if hasattr(dataset, 'targets'):
            print(f"- targets: {type(dataset.targets)} (shape: {dataset.targets.shape})")