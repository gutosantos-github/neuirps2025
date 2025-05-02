import sys
import os
import joblib
import csv
import numpy as np
from tqdm import tqdm
from time import time
from codecarbon import EmissionsTracker
from memory_profiler import memory_usage
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# === Add path to where hd_models_classifiers is located ===
sys.path.append("/mnt/sdb4/threathd/machine_learning")

n_runs = 1
eval_file = "eval_list.txt"

MODEL_DIR = "/mnt/sdb4/threathd/machine_learning/saved_trained_AM_models"
TESTDATA_DIR = "/mnt/sdb4/threathd/machine_learning/saved_X_y_test_models_datasets"
OUTPUT_DIR = "evaluation_results"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Cria o diretório de logs de carbono se ele não existir
carbon_logs_dir = "carbon_logs"
os.makedirs(carbon_logs_dir, exist_ok=True)

with open(eval_file, "r") as file:
    eval_pairs = [
        line.strip().split()
        for line in file
        if line.strip() and not line.startswith("#")
    ]

total_evals = len(eval_pairs) * n_runs

with tqdm(total=total_evals, desc="Evaluating Models", unit="eval") as pbar:
    for dataset_name, model_name in eval_pairs:
        csv_filename = f"{model_name}_{dataset_name}.csv"
        csv_path = os.path.join(OUTPUT_DIR, csv_filename)

        results = []

        for run in range(1, n_runs + 1):
            model_path = os.path.join(MODEL_DIR, f"{model_name}_{dataset_name}_.pkl")
            X_test_path = os.path.join(TESTDATA_DIR, f"{dataset_name}_X_test.pkl")
            y_test_path = os.path.join(TESTDATA_DIR, f"{dataset_name}_y_test.pkl")

            X_test = joblib.load(X_test_path)
            y_test = joblib.load(y_test_path)
            model = joblib.load(model_path)

            # Inicia medição de carbono
            tracker = EmissionsTracker(
                log_level="error",
                project_name=f"{model_name}_{dataset_name}_run{run}",
                output_dir=carbon_logs_dir,
                pue=1.59
            )
            tracker.start()

            # Medição de tempo e memória durante a predição
            start = time()
            test_memory_usage = memory_usage((model.predict, (X_test,)), max_usage=True)
            y_pred = model.predict(X_test)
            end = time()

            emissions = tracker.stop()
            inference_time = end - start

            acc = accuracy_score(y_test, y_pred) * 100
            prec = precision_score(y_test, y_pred, average='macro', zero_division=0) * 100
            rec = recall_score(y_test, y_pred, average='macro', zero_division=0) * 100
            f1 = f1_score(y_test, y_pred, average='macro', zero_division=0) * 100

            results.append([
                acc, prec, rec, f1,
                emissions, inference_time, test_memory_usage
            ])

            pbar.update(1)

        # Escreve os resultados no CSV
        with open(csv_path, mode="w", newline="") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow([
                "Accuracy (%)", "Precision (%)", "Recall (%)", "F1-Score (%)",
                "Carbon Emissions (kgCO2eq)", "Inference Time (s)", "Test Mem (MiB)"
            ])
            writer.writerows(results)
