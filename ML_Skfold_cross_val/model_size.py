import os
import pickle
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split

# Import your dataset classes
from datasets_classes.cic_ids_2017_class import Cic_ids_2017
from datasets_classes.botNetIot_L01_class import BotNetIot
from datasets_classes.unsw_nb15_class import Unsw_nb15
from datasets_classes.ids_class_malicious import IntrusionDetection
from datasets_classes.bccc_cic_ids2017_class import Bccc_cic_ids2017
from datasets_classes.nsl_kdd_class import Nslkdd

# Define datasets with proper data loading
datasets = {
    # "malicious": {
    #     "loader": IntrusionDetection(),
    #     "load_method": lambda x: (x.features, x.labels)  # Adjust based on actual attributes
    # },
    # "nsl_kdd": {
    #     "loader": Nslkdd(),
    #     "load_method": lambda x: (x.features, x.labels)  # Adjust based on actual attributes
    # },
    # Add other datasets similarly
    "unsw_nb15": {
        "loader": Unsw_nb15(),
        "load_method": lambda x: (x.X, x.y)  # Adjust if attribute names are different
    },
    # "cic_ids_2017": {
    #     "loader": Cic_ids_2017(),
    #     "load_method": lambda x: (x.features, x.labels)  # Adjust as needed
    # },
    "bccc_cic_ids2017": {
        "loader": Bccc_cic_ids2017(),
        "load_method": lambda x: (x.X, x.Y)  # Adjust as needed
    },
    "botNetIot": {
        "loader": BotNetIot(),
        "load_method": lambda x: (x.data, x.target)  # Adjust as needed
    }
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

# Then modify the evaluation section to handle cases where features/labels might be None:
for dataset_name, dataset_info in datasets.items():
    print(f"\n--- Dataset: {dataset_name} ---")
    dataset = dataset_info["loader"]
    load_method = dataset_info["load_method"]

    try:
        # Load data using the specified method
        features, labels = load_method(dataset)

        # Check if data was loaded correctly
        if features is None or labels is None:
            raise ValueError(f"Could not load data for {dataset_name} - features or labels is None")

        # Print shapes for debugging
        print(f"Features shape: {features.shape if hasattr(features, 'shape') else len(features)}")
        print(f"Labels shape: {labels.shape if hasattr(labels, 'shape') else len(labels)}")

        # Split data (if not already split)
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
        print(f"Available attributes: {dir(dataset)}")
        # Print which attributes actually contain data
        print("Data-containing attributes:")
        for attr in dir(dataset):
            if not attr.startswith('__') and not callable(getattr(dataset, attr)):
                val = getattr(dataset, attr)
                if val is not None:
                    print(f"- {attr}: {type(val)}")

