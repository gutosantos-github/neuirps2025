from binhd.datasets.dataset import Dataset
import os

import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import StandardScaler, LabelEncoder

class IntrusionDetection(Dataset):
    """Dataset normalizado com StandardScaler"""

    name = "dataset"
    categorical_features = []
    numeric_features = ['frame.time_delta', 'tcp.time_delta', 'tcp.flags.ack',
                        'tcp.flags.push', 'tcp.flags.reset', 'mqtt.hdrflags',
                        'mqtt.msgtype', 'mqtt.qos', 'mqtt.retain', 'mqtt.ver']

    def __init__(self):
        base_dir = "/mnt/sdb4/new_binHD++/binhd_iterative_fit/dataset"
        # base_dir = "dataset"
        # Nome da coluna de rótulo (última coluna do dataset)
        self.target_col = "label"  # Defina explicitamente o nome da coluna de destino

        # Verifica se todos os arquivos existem
        for i in range(4):
            # file_path = os.path.join(base_dir, f"ids_{i}.csv") # original Leandro
            file_path = os.path.join(base_dir, f"ids_{i}.csv")
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"Arquivo não encontrado: {file_path}")

        # Carrega os arquivos
        df1 = pd.read_csv(os.path.join(base_dir, "ids_0.csv"))
        df2 = pd.read_csv(os.path.join(base_dir, "ids_1.csv"))
        df3 = pd.read_csv(os.path.join(base_dir, "ids_2.csv"))
        df4 = pd.read_csv(os.path.join(base_dir, "ids_3.csv"))

        # Combina os arquivos
        df = pd.concat([df1, df2, df3, df4], ignore_index=True)
        print(f"Shape: {df.shape}")

        # Separar os dados e o label
        features = df.iloc[:, :-1]  # Todas as colunas exceto a última
        labels = df.iloc[:, -1]  # Última coluna

        # Normalizar as features
        scaler = StandardScaler()
        features_normalized = scaler.fit_transform(features)

        # Reconstruir o dataset com os labels
        df = pd.DataFrame(features_normalized, columns=features.columns)

        # print(df.memory_usage(deep=True).sum() / (1024 ** 2), "MB")
        df[self.target_col] = labels.values

        # Define atributos da classe
        self.features = df.iloc[:, :-1] #retorna as features sem label
        self.targets = df[[self.target_col]] #retorna o label
        self.num_features = self.features.shape[1]

        # print(df.memory_usage(deep=True).sum() / (1024 ** 2), "MB")
        self.gen_class_ids()

    def get_model_size(self, model) -> float:
        component_sizes = {}
        total_bytes = 0

        # Coletar buffers específicos (classes_counter e classes_hv)
        for name, buffer in model.named_buffers():
            # for name, buffer in model.named_buffers():
            #     print(f"Buffer: {name}, Shape: {tuple(buffer.shape)}, Dtype: {buffer.dtype}")
            if name in ["classes_counter", "classes_hv"]:
                size = buffer.nelement() * buffer.element_size()
                component_sizes[name] = size
                total_bytes += size

        # Coletar parâmetro do encoder: encoder.projection
        for name, param in model.named_parameters():
            if name == "encoder.projection.weight":
                size = param.nelement() * param.element_size()
                component_sizes["encoder.projection.weight"] = size
                total_bytes += size

        for name, param in model.named_parameters():
            # for name, param in model.named_parameters():
            #     print(f"Parameter: {name}, Shape: {tuple(param.shape)}, Dtype: {param.dtype}")
            if name == "encoder.projection.bias":
                size = param.nelement() * param.element_size()
                component_sizes["encoder.projection.bias"] = size
                total_bytes += size

        # Impressão formatada
        print("Model component sizes:")
        for name in ["classes_counter", "classes_hv", "encoder.projection.weight", "encoder.projection.bias"]:
            size = component_sizes.get(name, 0)
            print(f"  {name:20s}: {size / (1024 ** 2):.4f} MB")

        print(f"\nTotal model size: {total_bytes / (1024 ** 2):.4f} MB")
        return total_bytes / (1024 ** 2)



