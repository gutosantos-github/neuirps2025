from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

from datasets_classes.nsl_kdd_class import Nslkdd
from datasets_classes.unsw_nb15_class import Unsw_nb15
from datasets_classes.cic_ids_2017_class import Cic_ids_2017
from datasets_classes.bccc_cic_ids2017_class import Bccc_cic_ids2017
from datasets_classes.botNetIot_L01_class import BotNetIot
from datasets_classes.ids_class_malicious import IntrusionDetection

from codecarbon import EmissionsTracker
import warnings
warnings.filterwarnings("ignore")
from time import time
import os

# Create emissions folder if it doesn't exist
if not os.path.exists("emissions"):
    os.makedirs("emissions")

test_size = 0.3
k_folds = 10

kf = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=0)

datasets = {
    # "malicious": IntrusionDetection(),
    # "nsl_kdd": Nslkdd(),
    # "unsw_nb15": Unsw_nb15(),
    # "cic_ids_2017": Cic_ids_2017(),
    # "bccc_cic_ids2017": Bccc_cic_ids2017(),
    "botNetIot": BotNetIot()
}

models = {
    "GaussianNB": GaussianNB(),
    "KNN": KNeighborsClassifier(n_neighbors=5),
    "RandomForest": RandomForestClassifier(max_depth=10, random_state=100),
    "AdaBoost": AdaBoostClassifier(),
    "LogisticRegression": LogisticRegression(),
    "DecisionTree": DecisionTreeClassifier(criterion="gini", random_state=100, max_depth=10, min_samples_leaf=5)
}

for name_dataset, dataset in datasets.items():
    print(f"\n######## Dataset {name_dataset} ########")

    X = dataset.features
    y = dataset.labels

    for name_model, model_class in models.items():
        print(f"\n======== MODEL: {name_model} on DATASET: {name_dataset} ========")

        tracker = EmissionsTracker(log_level="error",
                                   project_name=f"{name_model}_{name_dataset}_experiment",
                                   output_dir="emissions", pue=1.59)
        tracker.start()

        acc_list = []
        precision_list = []
        recall_list = []
        f1_list = []
        roc_auc_list = []
        train_time_list = []
        test_time_list = []

        for fold, (train_idx, val_idx) in enumerate(kf.split(X, y)):
            print(f"Fold {fold + 1}/{k_folds}")
            X_train, X_test = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[val_idx]

            model = model_class

            # --- Training ---
            start_train = time()
            model.fit(X_train, y_train)
            end_train = time()
            train_time = end_train - start_train
            train_time_list.append(train_time)

            # --- Testing ---
            start_test = time()
            y_pred = model.predict(X_test)
            end_test = time()
            test_time = end_test - start_test
            test_time_list.append(test_time)

            # --- Metrics ---
            acc_list.append(round(accuracy_score(y_test, y_pred), 4))
            precision_list.append(round(precision_score(y_test, y_pred, average='weighted', zero_division=0), 4))
            recall_list.append(round(recall_score(y_test, y_pred, average='weighted', zero_division=0), 4))
            f1_list.append(round(f1_score(y_test, y_pred, average='weighted', zero_division=0), 4))
            roc_auc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1]) if hasattr(model, "predict_proba") else 0.0
            roc_auc_list.append(round(roc_auc, 4))

        emissions = tracker.stop()  # in kg

# Save the results
        results_text = f"""
======== Evaluation Results ========
Model name: {name_model}
Dataset name: {name_dataset}

Accuracy results: {acc_list}
Average Accuracy: {sum(acc_list) / k_folds:.4f}
Precision results: {precision_list}
Average Precision: {sum(precision_list) / k_folds:.4f}
Recall results: {recall_list}
Average Recall: {sum(recall_list) / k_folds:.4f}
F1 Score results: {f1_list}
Average F1 Score: {sum(f1_list) / k_folds:.4f}
ROC AUC results: {roc_auc_list}
Average ROC AUC: {sum(roc_auc_list) / k_folds:.4f}
Average Training Time per Fold: {sum(train_time_list) / k_folds:.4f} seconds
Average Testing Time per Fold: {sum(test_time_list) / k_folds:.4f} seconds
Total CO₂ emissions: {emissions:.6f} kg

========================================================
"""
        filename = f"{name_model}_{name_dataset}_results.txt"
        with open(filename, "w") as f:
            f.write(results_text)

        print(f"Results saved to {filename}")
