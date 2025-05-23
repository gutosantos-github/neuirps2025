from typing import Dict
import numpy as np
from sklearn.preprocessing import LabelEncoder
from torch.utils import data
import torch.nn as nn

class Dataset():
    isimage = False
    transform = None
    target_transform = None

    def get_unique_categories_values(self) -> Dict[str, list]:
        all_unique_cat_values = {}

        for feature in self.categorical_features:
            unique_values = np.sort(self.features[feature].unique()).tolist()
            all_unique_cat_values[feature] = unique_values
        return all_unique_cat_values

    def get_range_numeric_values(self):
        all_range_values = {}

        for feature in self.numeric_features:
            minVal = min(self.features[feature])
            maxVal = max(self.features[feature])
            all_range_values[feature] = (minVal, maxVal)

        return all_range_values

    def get_min_max_values(self):
        all_range_values = {"min": [], "max": []}

        for feature in self.numeric_features:
            minVal = min(self.features[feature])
            maxVal = max(self.features[feature])
            all_range_values["min"].append(minVal)
            all_range_values["max"].append(maxVal)

        return min(all_range_values["min"]), max(all_range_values["max"])

    def gen_class_ids(self):
        # Generating class ids
        self.classes = self.targets[self.target_col].unique()
        self.num_classes = len(self.classes)

        if isinstance(self.classes[0], str):
            label_encoder = LabelEncoder()
            self.labels = label_encoder.fit_transform(self.targets[self.target_col])
        else:
            self.labels_id = self.classes
            self.labels = self.targets[self.target_col]

    def __len__(self):
        return len(self.dataset.labels)

    def __getitem__(self, index: int):
        sample = self.features.iloc[index, :]
        label = self.labels[index]

        if self.transform:
            sample = self.transform(sample)

        if self.target_transform:
            label = self.target_transform(label)

        return sample, label

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

class BaseDataset(data.Dataset):
    def __init__(self, features, labels):
        self.features = features
        self.labels = labels

    def __len__(self):
        return len(self.features)

    def __getitem__(self, index: int):
        return self.features[index], self.labels[index]

