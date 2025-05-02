from dataset.dataset import Dataset
import pandas as pd
import os
from sklearn.preprocessing import StandardScaler


class Unsw_nb15(Dataset):
    """Dataset normalizado com StandardScaler"""

    name = "unsw_nb15"
    categorical_features = []
    numeric_features = ['ct_state_ttl', 'sttl', 'dttl', 'smean', 'dmean', 'rate', 'dinpkt',
       'tcprtt', 'dur', 'sload', 'dpkts', 'dload', 'sinpkt', 'sjit', 'djit',
       'spkts', 'state', 'ct_dst_ltm', 'ct_srv_src', 'ct_src_ltm']

    def __init__(self):
        # base_dir = "/binhd-ids/binhd-ids/datasets/dataset"
        base_dir = "/mnt/sdb4/new_binHD++/binhd_iterative_fit/dataset_unsw_nb15"

        # Nome da coluna de rótulo (última coluna do dataset)
        self.target_col = "label"  # Defina explicitamente o nome da coluna de destino

        # Verifica se todos os arquivos existem
        df = pd.read_csv(os.path.join(base_dir, "unsw_nb15.csv"))
        file_path = os.path.join(base_dir, f"unsw_nb15.csv")
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Arquivo não encontrado: {file_path}")
        print(f"Shape: {df.shape}")

        # Separar os dados e o label
        features = df.iloc[:, :-1]  # Todas as colunas exceto a última
        labels = df.iloc[:, -1]  # Última coluna

        # Normalizar as features MinMaxScaler
        scaler = StandardScaler()
        features_normalized = scaler.fit_transform(features)

        # Reconstruir o dataset com os labels
        df = pd.DataFrame(features_normalized, columns=features.columns)

        df[self.target_col] = labels.values

        # Define atributos da classe
        self.features = df.iloc[:, :-1] #retorna as features sem label
        self.targets = df[[self.target_col]] #retorna o label
        self.num_features = self.features.shape[1]

        # print(df.memory_usage(deep=True).sum() / (1024 ** 2), "MB")
        self.gen_class_ids()




