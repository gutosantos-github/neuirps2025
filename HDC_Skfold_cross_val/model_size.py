import torch
import os
from HDC_classifiers.hd_models_classifiers import NeuralHD, OnlineHD, BinHD, AdaptHD, DistHD, CompHD
from datasets_classes.cic_ids_2017_class import Cic_ids_2017
from datasets_classes.botNetIot_L01_class import BotNetIot
from datasets_classes.unsw_nb15_class import Unsw_nb15
from datasets_classes.ids_class_malicious import IntrusionDetection
from datasets_classes.bccc_cic_ids2017_class import Bccc_cic_ids2017
from datasets_classes.nsl_kdd_class import Nslkdd

# Configurações
dimension = 1024
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Função para calcular o tamanho de um tensor em MB
def tensor_size_mb(tensor):
    return tensor.numel() * tensor.element_size() / (1024 ** 2)

# Datasets
datasets = {
    "malicious": IntrusionDetection(),
    "nsl_kdd": Nslkdd(),
    "unsw_nb15": Unsw_nb15(),
    "cic_ids_2017": Cic_ids_2017(),
    "bccc_cic_ids2017": Bccc_cic_ids2017(),
    "botNetIot": BotNetIot()
}

# Modelos
models = {
    "BinHD": BinHD,
    "AdaptHD": AdaptHD,
    "DistHD": DistHD,
    "CompHD": CompHD,
    "NeuralHD": NeuralHD,
    "OnlineHD": OnlineHD
}

# Pasta para salvar resultados
os.makedirs("model_sizes", exist_ok=True)

for dataset_name, dataset in datasets.items():
    print(f"\n--- Dataset: {dataset_name} ---")
    with open(f"model_sizes/{dataset_name}_model_sizes_no_encoding.txt", "w") as f:
        for model_name, ModelClass in models.items():
            model = ModelClass(dataset.num_features, dimension, dataset.num_classes).to(device)
            total_size = 0.0

            f.write(f"\nModel: {model_name}\n")

            # Skip encoder weights when computing total size
            for name, param in model.named_parameters():
                size = tensor_size_mb(param)
                # Log all parameters, but only add non-encoder ones to total_size
                f.write(f"  {name:<25}: {size:.6f} MB\n")
                if not name.startswith("encoder."):  # Skip encoder weights
                    total_size += size

            # Include extra attributes (classes_counter, classes_hv)
            for attr in ['classes_counter', 'classes_hv']:
                if hasattr(model, attr):
                    tensor = getattr(model, attr)
                    if tensor is not None and isinstance(tensor, torch.Tensor):
                        size = tensor_size_mb(tensor)
                        total_size += size
                        f.write(f"  {attr:<25}: {size:.6f} MB\n")

            f.write(f"Total model size (no encoding): {total_size:.6f} MB\n")
            print(f"Model: {model_name} | Size (no encoding): {total_size:.6f} MB")
