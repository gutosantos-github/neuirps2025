from dataset.dataset import Dataset
import pandas as pd
import os
from sklearn.preprocessing import StandardScaler


class BotNetIot(Dataset):
    """Dataset normalizado com StandardScaler"""

    name = "botnetiot"
    categorical_features = []
    numeric_features = ['mi_dir_l0.1_weight', 'mi_dir_l0.1_mean', 'hh_l0.1_weight',
       'hh_l0.1_mean', 'hh_l0.1_std', 'hh_l0.1_covariance', 'hh_l0.1_pcc',
       'hh_jit_l0.1_mean', 'hh_jit_l0.1_variance', 'hphp_l0.1_weight',
       'hphp_l0.1_radius', 'hphp_l0.1_covariance', 'hphp_l0.1_pcc']

    def __init__(self):
        base_dir = "/mnt/sdb4/new_binHD++/binhd_iterative_fit/dataset_botNeTIoT-L01"

        # Nome da coluna de rótulo (última coluna do dataset)
        self.target_col = "label"  # Defina explicitamente o nome da coluna de destino

        # Verifica se todos os arquivos existem
        df = pd.read_csv(os.path.join(base_dir, "botNeTIoT-L01.csv"))
        file_path = os.path.join(base_dir, f"botNeTIoT-L01.csv")
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




