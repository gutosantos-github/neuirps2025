import torch
import torchmetrics
from codecarbon import EmissionsTracker

from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader
from binhd.datasets import BaseDataset

import warnings
warnings.filterwarnings("ignore")
from time import time

from HDC_classifiers.hd_models_classifiers import NeuralHD, OnlineHD, BinHD, AdaptHD, DistHD, CompHD
from datasets_classes.nsl_kdd_class import Nslkdd
from datasets_classes.unsw_nb15_class import Unsw_nb15
from datasets_classes.cic_ids_2017_class import Cic_ids_2017
from datasets_classes.bccc_cic_ids2017_class import Bccc_cic_ids2017
from datasets_classes.botNetIot_L01_class import BotNetIot
from datasets_classes.ids_class_malicious import IntrusionDetection

import os

# Create emissions folder if it doesn't exist
if not os.path.exists("emissions"):
    os.makedirs("emissions")

# Use the GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using {} device".format(device))

dimension = 1024
num_levels = 100
batch_size = 10000
test_size = 0.3
k_folds = 10

kf = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=0)

#Datasets
datasets = {
    # "malicious": IntrusionDetection(),
    # "nsl_kdd": Nslkdd(),
    # "unsw_nb15": Unsw_nb15(),
    "cic_ids_2017": Cic_ids_2017(),
    # "bccc_cic_ids2017": Bccc_cic_ids2017(),
    "botNetIot": BotNetIot()
}

acc_list = []
precision_list = []
recall_list = []
f1_list = []
roc_auc_list = []
train_time_list = []
test_time_list = []

for name_dataset, dataset in datasets.items():

    print(f"######## Dataset {name_dataset} ########")

    min_val, max_val = dataset.get_min_max_values()

    X = dataset.features
    X = torch.tensor(X.values).to(dtype=torch.float32, device=device)
    y = dataset.labels
    y = torch.tensor(y).squeeze().to(device)

    for name_model, model_class in {
        "BinHD": BinHD,
        "AdaptHD": AdaptHD,
        "DistHD": DistHD,
        "CompHD": CompHD,
        "NeuralHD": NeuralHD,
        "OnlineHD": OnlineHD
    }.items():

        print(f"\n======== MODEL: {name_model} on DATASET: {name_dataset} ========")

        tracker = EmissionsTracker(log_level="error",
                                   project_name=f"{name_model}_{name_dataset}_experiment",
                                   output_dir="emissions", pue=1.59)
        tracker.start()

        # Reset metrics per model
        acc_list = []
        precision_list = []
        recall_list = []
        f1_list = []
        roc_auc_list = []
        train_time_list = []
        test_time_list = []

        for fold, (train_idx, val_idx) in enumerate(kf.split(X.cpu(), y.cpu())):
            print(f"\nFold {fold + 1}/{k_folds}")

            # Reinitialize the model for each fold
            model = model_class(dataset.num_features, dimension, dataset.num_classes)

            # Initialize metrics for each fold
            accuracy = torchmetrics.Accuracy(task="binary", num_classes=dataset.num_classes).to(device)
            precision = torchmetrics.Precision(task="binary", num_classes=dataset.num_classes).to(device)
            recall = torchmetrics.Recall(task="binary", num_classes=dataset.num_classes).to(device)
            f1 = torchmetrics.F1Score(task="binary", num_classes=dataset.num_classes).to(device)
            roc_auc = torchmetrics.AUROC(task="binary", num_classes=dataset.num_classes).to(device)

            # Split dataset
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]

            train_dataset = BaseDataset(X_train, y_train)
            val_dataset = BaseDataset(X_val, y_val)

            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
            val_loader = DataLoader(val_dataset, batch_size=batch_size, num_workers=4)

            with torch.no_grad():
                # --- Training ---
                start_train = time()
                for samples, labels in train_loader:
                    samples, labels = samples.to(device), labels.to(device)
                    model.fit(samples, labels)
                end_train = time()

                train_duration = end_train - start_train
                train_time_list.append(train_duration)

                # --- Validation ---
                start_test = time()
                for samples, labels in val_loader:
                    samples, labels = samples.to(device), labels.to(device)
                    predictions = model.predict(samples)

                    accuracy.update(predictions, labels)
                    precision.update(predictions, labels)
                    recall.update(predictions, labels)
                    f1.update(predictions, labels)
                    roc_auc.update(predictions, labels)
                end_test = time()

                test_duration = end_test - start_test
                test_time_list.append(test_duration)

                # Compute fold metrics
                acc_list.append(round(accuracy.compute().item(), 4))
                precision_list.append(round(precision.compute().item(), 4))
                recall_list.append(round(recall.compute().item(), 4))
                f1_list.append(round(f1.compute().item(), 4))
                roc_auc_list.append(round(roc_auc.compute().item(), 4))

        emissions = tracker.stop()
        # --- Save the model results after all folds ---
        results_text = f"""
======== Evaluation Results ========
HDC Model name: {name_model}
Dataset name: {name_dataset}
Device: {device}
Min Value: {min_val:.2f}, Max Value: {max_val:.2f}

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
        with open(filename, "w") as file:
            file.write(results_text)

        print(f"Results saved to {filename}")

