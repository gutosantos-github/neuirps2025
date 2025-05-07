import torch
import torch.nn as nn
import torch.nn.functional as F
import torchhd.functional as functional
from torch import Tensor
import math
from tqdm import trange
from torchhd.classifiers import Centroid
from torchhd.embeddings import Sinusoid, Random, Level, Projection
# from hd_embeddings import RandomProjectionEncoder
from embedding.new_embedding import RandomProjectionEncoder
import scipy.linalg

class BinHD(nn.Module):
    def __init__(
            self,
            n_features: int,
            n_dimensions: int,
            n_classes: int,
            *,
            epochs: int = 30,
            device: torch.device = None,
    ) -> None:
        super().__init__()
        self.n_features = n_features
        self.n_dimensions = n_dimensions
        self.n_classes = n_classes
        self.epochs = epochs
        self.classes_counter = torch.empty((n_classes, n_dimensions), device=device, dtype=torch.int8)
        self.classes_hv = None
        self.reset_parameters()
        self.encoder = RandomProjectionEncoder(n_features, n_dimensions)

    def reset_parameters(self) -> None:
        nn.init.zeros_(self.classes_counter)

    def fit(self, input: Tensor, target: Tensor):
        encoded = self.encoder.forward_bin(input)

        encoded = 2 * encoded - 1
        self.classes_counter.index_add_(0, target, encoded)
        self.classes_hv = self.classes_counter.clamp(min=0, max=1)

    def fit_adapt(self, input: Tensor, target: Tensor):
        variance_evolution = []
        for _ in trange(0, self.epochs, desc="fit"):
            self.adapt(input, target)

    def adapt(self, input: Tensor, target: Tensor):
        # reinforces the correct class (by adding) and penalizes the incorrect class
        pred = self.predict(input)
        input = self.encoder.forward_bin(input)

        is_wrong = target != pred

        # cancel update if all predictions were correct
        if is_wrong.sum().item() == 0:
            return

        input = input[is_wrong]
        input = 2 * input - 1
        target = target[is_wrong]
        pred = pred[is_wrong]

        self.classes_counter.index_add_(0, target, input, alpha=1)
        self.classes_counter.index_add_(0, pred, input, alpha=-1)
        self.classes_hv = torch.where(self.classes_counter >= 0, 1, 0)

    def forward(self, samples: Tensor) -> Tensor:
        encoded = self.encoder.forward_bin(samples)
        response = torch.empty((self.n_classes, samples.shape[0]), dtype=torch.int32)

        for i in range(self.n_classes):
            # Hamming Distance = SUM(XOR(a, b))
            response[i] = torch.sum(torch.bitwise_xor(encoded, self.classes_hv[i]), dim=1)  # Hamming distance

        return response.transpose_(0, 1)

    def predict(self, samples: Tensor) -> Tensor:
        return torch.argmin(self(samples), dim=-1)

    def print_tensor_size_MB(self, name: str, tensor: torch.Tensor) -> None:
        if tensor is not None:
            size_bytes = tensor.numel() * tensor.element_size()
            size_MB = size_bytes / (1024 ** 2)
            print(f"{name:<25}: {size_MB:.4f} MB")
        else:
            print(f"{name:<25}: None")

    def print_model_size(self, model):
        total = 0
        print("Model component sizes:")

        # Print registered parameters (e.g., encoder weights)
        for name, param in model.named_parameters():
            size_bytes = param.numel() * param.element_size()
            total += size_bytes
            print(f"  {name:<25}: {size_bytes / (1024 ** 2):.4f} MB")

        # Include custom components manually if they exist
        extra_components = {
            'classes_counter': getattr(model, 'classes_counter', None),
            'classes_hv': getattr(model, 'classes_hv', None),
        }

        for name, tensor in extra_components.items():
            if tensor is not None and isinstance(tensor, torch.Tensor):
                size_bytes = tensor.numel() * tensor.element_size()
                total += size_bytes
                print(f"  {name:<25}: {size_bytes / (1024 ** 2):.4f} MB")

        print(f"\nTotal model size: {total / (1024 ** 2):.4f} MB")


class NeuralHD(nn.Module):
    r"""Implements `Scalable edge-based hyperdimensional learning system with brain-like neural adaptation <https://dl.acm.org/doi/abs/10.1145/3458817.3480958>`_.

    Args:
        n_features (int): Size of each input sample.
        n_dimensions (int): The number of hidden dimensions to use.
        n_classes (int): The number of classes.
        regen_freq (int, optional): The frequency in epochs at which to regenerate hidden dimensions.
        regen_rate (int, optional): The fraction of hidden dimensions to regenerate.
        epochs (int, optional): The number of iteration over the training data.
        lr (float, optional): The learning rate.
        device (``torch.device``, optional):  the desired device of the weights. Default: if ``None``, uses the current device for the default tensor type (see ``torch.set_default_tensor_type()``). ``device`` will be the CPU for CPU tensor types and the current CUDA device for CUDA tensor types.
        dtype (``torch.dtype``, optional): the desired data type of the weights. Default: if ``None``, uses ``torch.get_default_dtype()``.
    """

    model: Centroid
    encoder: Sinusoid

    # Initialize class hypervectors as floating-point vectors
    def __init__(
        self,
        n_features: int,
        n_dimensions: int,
        n_classes: int,
        *,
        regen_freq: int = 20,
        regen_rate: float = 0.04,
        epochs: int = 120,
        lr: float = 0.37,
        device: torch.device = None,
        dtype: torch.dtype = None
    ) -> None:
        super().__init__()

        self.n_features = n_features
        self.n_dimensions = n_dimensions
        self.n_classes = n_classes
        self.regen_freq = regen_freq
        self.regen_rate = regen_rate
        self.epochs = epochs
        self.lr = lr

        # Initialize class hypervectors as floating-point vectors
        self.encoder = Sinusoid(n_features, n_dimensions, device=device, dtype=dtype)
        self.model = Centroid(n_dimensions, n_classes, device=device, dtype=dtype)

    def fit(self, input: Tensor, target: Tensor):
        encoded = self.encoder(input)
        n_regen_dims = math.ceil(self.regen_rate * self.n_dimensions)

        self.model.add(encoded, target)#2

        for epoch_idx in trange(1, self.epochs, desc="fit"):
            encoded = self.encoder(input)#3
            self.model.add_adapt(encoded, target, lr=self.lr)

            # Regenerate feature dimensions
            if (epoch_idx % self.regen_freq) == (self.regen_freq - 1):
                weight = F.normalize(self.model.weight, dim=1)
                scores = torch.var(weight, dim=0)

                regen_dims = torch.topk(scores, n_regen_dims, largest=False).indices
                self.model.weight.data[:, regen_dims].zero_()
                self.encoder.weight.data[regen_dims, :].normal_()
                self.encoder.bias.data[:, regen_dims].uniform_(0, 2 * math.pi)

        return self

    def forward(self, samples: Tensor) -> Tensor:
        return self.model(self.encoder(samples))

    def predict(self, samples: Tensor) -> Tensor:
        return torch.argmax(self(samples), dim=-1)

    # def print_model_size(self, model):
    #     total = 0
    #     print("Model component sizes:")
    #     for name, param in model.named_parameters():
    #         size_bytes = param.numel() * param.element_size()
    #         total += size_bytes
    #         print(f"  {name:<25}: {size_bytes / (1024 ** 2):.4f} MB")
    #     print(f"\nTotal model size: {total / (1024 ** 2):.4f} MB")

    def print_model_size(self, model):
        total = 0
        print("Model component sizes:")

        # Print registered parameters (e.g., encoder weights)
        for name, param in model.named_parameters():
            size_bytes = param.numel() * param.element_size()
            total += size_bytes
            print(f"  {name:<25}: {size_bytes / (1024 ** 2):.4f} MB")

        # Include custom components manually if they exist
        extra_components = {
            'classes_counter': getattr(model, 'classes_counter', None),
            'classes_hv': getattr(model, 'classes_hv', None),
        }

        for name, tensor in extra_components.items():
            if tensor is not None and isinstance(tensor, torch.Tensor):
                size_bytes = tensor.numel() * tensor.element_size()
                total += size_bytes
                print(f"  {name:<25}: {size_bytes / (1024 ** 2):.4f} MB")

        print(f"\nTotal model size: {total / (1024 ** 2):.4f} MB")


class OnlineHD(nn.Module):
    r"""Implements `Scalable edge-based hyperdimensional learning system with brain-like neural adaptation <https://dl.acm.org/doi/abs/10.1145/3458817.3480958>`_.

    Args:
        n_features (int): Size of each input sample.
        n_dimensions (int): The number of hidden dimensions to use.
        n_classes (int): The number of classes.
        regen_freq (int, optional): The frequency in epochs at which to regenerate hidden dimensions.
        regen_rate (int, optional): The fraction of hidden dimensions to regenerate.
        epochs (int, optional): The number of iteration over the training data.
        lr (float, optional): The learning rate.
        device (``torch.device``, optional):  the desired device of the weights. Default: if ``None``, uses the current device for the default tensor type (see ``torch.set_default_tensor_type()``). ``device`` will be the CPU for CPU tensor types and the current CUDA device for CUDA tensor types.
        dtype (``torch.dtype``, optional): the desired data type of the weights. Default: if ``None``, uses ``torch.get_default_dtype()``.

    """
    model: Centroid
    encoder: Sinusoid

    def __init__(
            self,
            n_features: int,
            n_dimensions: int,
            n_classes: int,
            *,
            regen_freq: int = 20,
            regen_rate: float = 0.04,
            epochs: int = 120,
            lr: float = 0.37,
            device: torch.device = None,
            dtype: torch.dtype = None
    ) -> None:
        super().__init__()

        self.n_features = n_features
        self.n_dimensions = n_dimensions
        self.n_classes = n_classes
        self.regen_freq = regen_freq
        self.regen_rate = regen_rate
        self.epochs = epochs
        self.lr = lr

        self.encoder = Sinusoid(n_features, n_dimensions, device=device, dtype=dtype)
        self.model = Centroid(n_dimensions, n_classes, device=device, dtype=dtype)

    def fit(self, input: Tensor, target: Tensor):
        encoded = self.encoder(input)
        n_regen_dims = math.ceil(self.regen_rate * self.n_dimensions)
        self.model.add(encoded, target)

        for epoch_idx in trange(1, self.epochs, desc='fit'):
            encoded = self.encoder(input)
            # self.model.add_adapt(encoded, target, lr=self.lr)
            self.model.add_online(encoded, target, lr=self.lr)

            # Regenerate feature dimensions
            if (epoch_idx % self.regen_freq) == (self.regen_freq - 1):
                weight = F.normalize(self.model.weight, dim=1)
                scores = torch.var(weight, dim=0)

                regen_dims = torch.topk(scores, n_regen_dims, largest=False).indices
                self.model.weight.data[:, regen_dims].zero_()
                self.encoder.weight.data[regen_dims, :].normal_()
                self.encoder.bias.data[:, regen_dims].uniform_(0, 2 * math.pi)

        return self

    def forward(self, samples: Tensor) -> Tensor:
        return self.model(self.encoder(samples))

    def predict(self, samples: Tensor) -> Tensor:
        return torch.argmax(self(samples), dim=-1)

    def print_model_size(self, model):
        total = 0
        print("Model component sizes:")

        # Print registered parameters (e.g., encoder weights)
        for name, param in model.named_parameters():
            size_bytes = param.numel() * param.element_size()
            total += size_bytes
            print(f"  {name:<25}: {size_bytes / (1024 ** 2):.4f} MB")

        # Include custom components manually if they exist
        extra_components = {
            'classes_counter': getattr(model, 'classes_counter', None),
            'classes_hv': getattr(model, 'classes_hv', None),
        }

        for name, tensor in extra_components.items():
            if tensor is not None and isinstance(tensor, torch.Tensor):
                size_bytes = tensor.numel() * tensor.element_size()
                total += size_bytes
                print(f"  {name:<25}: {size_bytes / (1024 ** 2):.4f} MB")

        print(f"\nTotal model size: {total / (1024 ** 2):.4f} MB")

class AdaptHD(nn.Module):
    r"""Implements `AdaptHD: Adaptive Efficient Training for Brain-Inspired Hyperdimensional Computing <https://ieeexplore.ieee.org/document/8918974>`_.

    Args:
        n_features (int): Size of each input sample.
        n_dimensions (int): The number of hidden dimensions to use.
        n_classes (int): The number of classes.
        n_levels (int, optional): The number of discretized levels for the level-hypervectors.
        min_level (int, optional): The lower-bound of the range represented by the level-hypervectors.
        max_level (int, optional): The upper-bound of the range represented by the level-hypervectors.
        epochs (int, optional): The number of iteration over the training data.
        lr (float, optional): The learning rate.
        device (``torch.device``, optional):  the desired device of the weights. Default: if ``None``, uses the current device for the default tensor type (see ``torch.set_default_tensor_type()``). ``device`` will be the CPU for CPU tensor types and the current CUDA device for CUDA tensor types.
        dtype (``torch.dtype``, optional): the desired data type of the weights. Default: if ``None``, uses ``torch.get_default_dtype()``.

    """

    model: Centroid

    def __init__(
        self,
        n_features: int,
        n_dimensions: int,
        n_classes: int,
        *,
        n_levels: int = 100,
        min_level: int = -1,
        max_level: int = 1,
        epochs: int = 120,
        lr: float = 0.035,
        device: torch.device = None,
        dtype: torch.dtype = None
    ) -> None:
        super().__init__()

        self.n_features = n_features
        self.n_dimensions = n_dimensions
        self.n_classes = n_classes
        self.n_levels = n_levels
        self.min_level = min_level
        self.max_level = max_level
        self.epochs = epochs
        self.lr = lr

        self.keys = Random(n_features, n_dimensions, device=device, dtype=dtype)
        self.levels = Level(
            n_levels,
            n_dimensions,
            low=min_level,
            high=max_level,
            device=device,
            dtype=dtype,
        )
        self.model = Centroid(n_dimensions, n_classes, device=device, dtype=dtype)

    def encoder(self, samples: Tensor) -> Tensor:
        return functional.hash_table(self.keys.weight, self.levels(samples)).sign()

    def fit(self, input: Tensor, target: Tensor):
        encoded = self.encoder(input)
        self.model.add(encoded, target)

        for _ in trange(1, self.epochs, desc="fit"):
            self.model.add_adapt(encoded, target, lr=self.lr)
        return self

    def forward(self, samples: Tensor) -> Tensor:
        return self.model(self.encoder(samples))

    def predict(self, samples: Tensor) -> Tensor:
        return torch.argmax(self(samples), dim=-1)

    def print_model_size(self, model):
        total = 0
        print("Model component sizes:")
        for name, param in model.named_parameters():
            size_bytes = param.numel() * param.element_size()
            total += size_bytes
            print(f"  {name:<25}: {size_bytes / (1024 ** 2):.4f} MB")
        print(f"\nTotal model size: {total / (1024 ** 2):.4f} MB")


class DistHD(nn.Module):
    r"""Implements `DistHD: A Learner-Aware Dynamic Encoding Method for Hyperdimensional Classification <https://ieeexplore.ieee.org/document/10247876>`_.

    Args:
        n_features (int): Size of each input sample.
        n_dimensions (int): The number of hidden dimensions to use.
        n_classes (int): The number of classes.
        regen_freq (int): The frequency in epochs at which to regenerate hidden dimensions.
        regen_rate (int): The fraction of hidden dimensions to regenerate.
        alpha (float): Parameter effecting the dimensions to regenerate, see paper for details.
        beta (float): Parameter effecting the dimensions to regenerate, see paper for details.
        theta (float): Parameter effecting the dimensions to regenerate, see paper for details.
        epochs (int): The number of iteration over the training data.
        lr (float): The learning rate.
        device (``torch.device``, optional):  the desired device of the weights. Default: if ``None``, uses the current device for the default tensor type (see ``torch.set_default_tensor_type()``). ``device`` will be the CPU for CPU tensor types and the current CUDA device for CUDA tensor types.
        dtype (``torch.dtype``, optional): the desired data type of the weights. Default: if ``None``, uses ``torch.get_default_dtype()``.

    """

    encoder: Projection
    model: Centroid

    def __init__(
        self,
        n_features: int,
        n_dimensions: int,
        n_classes: int,
        *,
        regen_freq: int = 20,
        regen_rate: float = 0.04,
        alpha: float = 0.5,
        beta: float = 1,
        theta: float = 0.25,
        epochs: int = 120,
        lr: float = 0.05,
        device: torch.device = None,
        dtype: torch.dtype = None
    ) -> None:
        super().__init__()

        self.n_features = n_features
        self.n_dimensions = n_dimensions
        self.n_classes = n_classes
        self.regen_freq = regen_freq
        self.regen_rate = regen_rate
        self.alpha = alpha
        self.beta = beta
        self.theta = theta
        self.epochs = epochs
        self.lr = lr

        self.encoder = Projection(n_features, n_dimensions, device=device, dtype=dtype)
        self.model = Centroid(n_dimensions, n_classes, device=device, dtype=dtype)

    def fit(self, input: Tensor, target: Tensor):
        n_regen_dims = math.ceil(self.regen_rate * self.n_dimensions)

        for epoch_idx in trange(1, self.epochs, desc="fit"):
            encoded = self.encoder(input)
            self.model.add_online(encoded, target, lr=self.lr)

            if (epoch_idx % self.regen_freq) == (self.regen_freq - 1):
                # Convert regen_score output to a tensor (if it's a scalar)
                raw_score = self.regen_score(input, target)
                if isinstance(raw_score, (int, float)):
                    scores = torch.zeros(self.n_dimensions, dtype=torch.float32)
                    scores.fill_(raw_score)  # Fill all dims with the same score
                else:
                    scores = torch.tensor(raw_score, dtype=torch.float32)

                # Ensure n_regen_dims is valid
                n_regen_dims = min(n_regen_dims, len(scores))
                regen_dims = torch.topk(scores, n_regen_dims, largest=False).indices

                self.model.weight.data[:, regen_dims].zero_()
                self.encoder.weight.data[regen_dims, :].normal_()
        return self
    # def fit(self, input: Tensor, target: Tensor):
    #
    #     n_regen_dims = math.ceil(self.regen_rate * self.n_dimensions)
    #
    #     for epoch_idx in trange(1, self.epochs, desc="fit"):
    #         encoded = self.encoder(input)
    #         self.model.add_online(encoded, target, lr=self.lr)
    #
    #         # Regenerate feature dimensions
    #         if (epoch_idx % self.regen_freq) == (self.regen_freq - 1):
    #             scores = 0
    #             scores += self.regen_score(input, target)
    #
    #             regen_dims = torch.topk(scores, n_regen_dims, largest=False).indices
    #             self.model.weight.data[:, regen_dims].zero_()
    #             self.encoder.weight.data[regen_dims, :].normal_()
    #     return self

    def regen_score(self, samples, labels):
        encoded = self.encoder(samples)
        scores = self.model(encoded)
        top2_preds = torch.topk(scores, k=2).indices
        pred1, pred2 = torch.unbind(top2_preds, dim=-1)
        is_wrong = pred1 != labels

        # cancel update if all predictions were correct
        if is_wrong.sum().item() == 0:
            return 0

        encoded = encoded[is_wrong]
        pred2 = pred2[is_wrong]
        labels = labels[is_wrong]
        pred1 = pred1[is_wrong]

        weight = F.normalize(self.model.weight, dim=1)

        # Partial correct
        partial = pred2 == labels

        dist2corr = torch.abs(weight[labels[partial]] - encoded[partial])
        dist2incorr = torch.abs(weight[pred1[partial]] - encoded[partial])
        partial_dist = torch.sum(
            (self.beta * dist2incorr - self.alpha * dist2corr), dim=0
        )

        # Completely incorrect
        complete = pred2 != labels
        dist2corr = torch.abs(weight[labels[complete]] - encoded[complete])
        dist2incorr1 = torch.abs(weight[pred1[complete]] - encoded[complete])
        dist2incorr2 = torch.abs(weight[pred2[complete]] - encoded[complete])
        complete_dist = torch.sum(
            (
                self.beta * dist2incorr1
                + self.theta * dist2incorr2
                - self.alpha * dist2corr
            ),
            dim=0,
        )
        return 0.5 * partial_dist + complete_dist

    def forward(self, samples: Tensor) -> Tensor:
        return self.model(self.encoder(samples))

    def predict(self, samples: Tensor) -> Tensor:
        return torch.argmax(self(samples), dim=-1)

    def print_model_size(self, model):
        total = 0
        print("Model component sizes:")

        # Print registered parameters (e.g., encoder weights)
        for name, param in model.named_parameters():
            size_bytes = param.numel() * param.element_size()
            total += size_bytes
            print(f"  {name:<25}: {size_bytes / (1024 ** 2):.4f} MB")

        # Include custom components manually if they exist
        extra_components = {
            'classes_counter': getattr(model, 'classes_counter', None),
            'classes_hv': getattr(model, 'classes_hv', None),
        }

        for name, tensor in extra_components.items():
            if tensor is not None and isinstance(tensor, torch.Tensor):
                size_bytes = tensor.numel() * tensor.element_size()
                total += size_bytes
                print(f"  {name:<25}: {size_bytes / (1024 ** 2):.4f} MB")

        print(f"\nTotal model size: {total / (1024 ** 2):.4f} MB")

class CompHD(nn.Module):
    r"""Implements `CompHD: Efficient Hyperdimensional Computing Using Model Compression <https://ieeexplore.ieee.org/document/8824908>`_.

    Args:
        n_features (int): Size of each input sample.
        n_dimensions (int): The number of hidden dimensions to use.
        n_classes (int): The number of classes.
        n_levels (int, optional): The number of discretized levels for the level-hypervectors.
        min_level (int, optional): The lower-bound of the range represented by the level-hypervectors.
        max_level (int, optional): The upper-bound of the range represented by the level-hypervectors.
        chunks (int, optional): The number of times the model is reduced in size.
        device (``torch.device``, optional):  the desired device of the weights. Default: if ``None``, uses the current device for the default tensor type (see ``torch.set_default_tensor_type()``). ``device`` will be the CPU for CPU tensor types and the current CUDA device for CUDA tensor types.
        dtype (``torch.dtype``, optional): the desired data type of the weights. Default: if ``None``, uses ``torch.get_default_dtype()``.

    """

    model: Centroid

    def __init__(
        self,
        n_features: int,
        n_dimensions: int,
        n_classes: int,
        *,
        n_levels: int = 100,
        min_level: int = -1,
        max_level: int = 1,
        chunks: int = 4,
        device: torch.device = None,
        dtype: torch.dtype = None
    ) -> None:
        # super().__init__(
        #     n_features, n_dimensions, n_classes, device=device, dtype=dtype
        # )
        super().__init__()
        self.n_features = n_features
        self.n_dimensions = n_dimensions
        self.n_classes = n_classes
        self.n_levels = n_levels
        self.chunks = chunks
        self.device = device
        self.dtype = dtype

        if n_dimensions % chunks != 0:
            raise ValueError("n_dimensions must be divisible by chunks.")

        self.chunks = chunks

        self.feat_keys = Random(n_features, n_dimensions, device=device, dtype=dtype)
        self.levels = Level(
            n_levels,
            n_dimensions,
            low=min_level,
            high=max_level,
            device=device,
            dtype=dtype,
        )

        self.model_count = Centroid(n_dimensions, n_classes, device=device, dtype=dtype)
        self.model = Centroid(
            n_dimensions // chunks, n_classes, device=device, dtype=dtype
        )

        n_chunk_keys = max(self.n_dimensions // self.chunks, self.chunks)
        chunk_keys = torch.from_numpy(scipy.linalg.hadamard(n_chunk_keys))
        chunk_keys = chunk_keys.to(self.device)
        self.chunk_keys = chunk_keys[: self.chunks, : self.n_dimensions // self.chunks]

    def encoder(self, samples: Tensor) -> Tensor:
        return functional.hash_table(self.feat_keys.weight, self.levels(samples)).sign()

    def forward(self, samples: Tensor) -> Tensor:
        return self.model(self.compress(self.encoder(samples)))

    def compress(self, input):
        shape = (input.size(0), self.chunks, self.n_dimensions // self.chunks)
        keys = self.chunk_keys[None, ...].expand(input.size(0), -1, -1)
        return functional.hash_table(keys, torch.reshape(input, shape))

    def fit(self, input: Tensor, target: Tensor):
        encoded = self.encoder(input)
        self.model_count.add(encoded, target)

        with torch.no_grad():
            self.model.weight.data = self.compress(self.model_count.weight)
        return self

    def predict(self, samples: Tensor) -> Tensor:
        return torch.argmax(self(samples), dim=-1)

    def print_model_size(self, model):
        total = 0
        print("Model component sizes:")

        # Print registered parameters (e.g., encoder weights)
        for name, param in model.named_parameters():
            size_bytes = param.numel() * param.element_size()
            total += size_bytes
            print(f"  {name:<25}: {size_bytes / (1024 ** 2):.4f} MB")

        # Include custom components manually if they exist
        extra_components = {
            'classes_counter': getattr(model, 'classes_counter', None),
            'classes_hv': getattr(model, 'classes_hv', None),
        }

        for name, tensor in extra_components.items():
            if tensor is not None and isinstance(tensor, torch.Tensor):
                size_bytes = tensor.numel() * tensor.element_size()
                total += size_bytes
                print(f"  {name:<25}: {size_bytes / (1024 ** 2):.4f} MB")

        print(f"\nTotal model size: {total / (1024 ** 2):.4f} MB")