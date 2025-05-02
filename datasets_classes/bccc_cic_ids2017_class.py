from dataset.dataset import Dataset
import pandas as pd
import os
from sklearn.preprocessing import StandardScaler


class Bccc_cic_ids2017(Dataset):
    """Dataset normalizado com StandardScaler"""

    name = "bccc_cic_ids2017"
    categorical_features = []
    numeric_features = ['payload_bytes_max', 'dst_port', 'fwd_total_payload_bytes', 'duration', 'packets_count', 'std_header_bytes', 'packets_rate', 'fwd_payload_bytes_variance',
               'bwd_bytes_rate', 'down_up_rate', 'bytes_rate', 'max_header_bytes', 'protocol', 'fwd_packets_iat_mean', 'fwd_min_header_bytes', 'psh_flag_counts', 'packets_iat_mean',
               'payload_bytes_min', 'fwd_psh_flag_counts', 'bwd_packets_iat_mean']

    def __init__(self):
        # base_dir = "/binhd-ids/binhd-ids/datasets/dataset"
        base_dir = "/mnt/sdb4/new_binHD++/binhd_iterative_fit/bccc_cic_ids2017"

        # Nome da coluna de rótulo (última coluna do dataset)
        self.target_col = "label"  # Defina explicitamente o nome da coluna de destino

        # Verifica se todos os arquivos existem
        df = pd.read_csv(os.path.join(base_dir, "bccc_cic_ids2017.csv"))
        file_path = os.path.join(base_dir, f"bccc_cic_ids2017.csv")
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




