from dataset.dataset import Dataset
import pandas as pd
import os
from sklearn.preprocessing import StandardScaler


class Cic_ids_2017(Dataset):
    """Dataset normalizado com StandardScaler"""

    name = "cic_ids_2017_class"
    categorical_features = []
    numeric_features = ['destination port', 'fwd packet length max', 'init_win_bytes_forward', 'init_win_bytes_backward', 'flow bytes/s',
'bwd packets/s', 'min_seg_size_forward', 'fwd packet length min', 'total fwd packets']

    def __init__(self):
        # base_dir = "/binhd-ids/binhd-ids/datasets/dataset"
        base_dir = "/mnt/sdb4/new_binHD++/binhd_iterative_fit/dataset_cic_ids2017"

        # Nome da coluna de rótulo (última coluna do dataset)
        self.target_col = "label"  # Defina explicitamente o nome da coluna de destino

        # Verifica se todos os arquivos existem
        df = pd.read_csv(os.path.join(base_dir, "cic_ids2017.csv"))
        file_path = os.path.join(base_dir, f"cic_ids2017.csv")
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




