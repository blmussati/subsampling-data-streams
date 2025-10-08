"""
N = number of examples
H = height
W = width
"""

from pathlib import Path
from typing import Any, Tuple, Union

import numpy as np
from torch import Tensor
from torchvision.datasets import MNIST as TorchVisionMNIST

from src.data.datasets.base import BaseDataset, BaseEmbeddingDataset
from src.data.input_preprocessing import preprocess_inputs_for_zero_mean_and_unit_variance
from src.typing import Array


class BaseSplitMNIST(BaseDataset):
    """
    If dataset = TorchVisionMNIST() then
    - dataset.data is a torch.Tensor with dtype torch.uint8, shape [N, H, W] and values in [0, 255]
    - dataset.targets is a torch.Tensor with dtype torch.int64
    - The class counts are [5923, 6742, 5958, 6131, 5842, 5421, 5918, 6265, 5851, 5949]
    """

    @staticmethod
    def preprocess_inputs_dtype_shape(inputs: Tensor) -> Tensor:
        inputs = inputs.numpy().astype(np.float32) / 255  # [N, 28, 28]
        return inputs[:, None, :, :]  # [N, 1, 28, 28]


class SplitMNIST(BaseSplitMNIST):
    def __init__(
        self,
        data_dir: Union[Path, str],
        classes_per_exp: np.ndarray = None,
        train: bool = True,
        input_preprocessing: str | None = "zero_mean_and_unit_variance",
        **kwargs: Any,
    ) -> None:
        data_dir = Path(data_dir) / "mnist"
        dataset = TorchVisionMNIST(root=data_dir, download=True, train=train, **kwargs)

        self.data = self.preprocess_inputs_dtype_shape(dataset.data)
        self.targets = dataset.targets.numpy()

        if input_preprocessing == "zero_mean_and_unit_variance":
            train_dataset = TorchVisionMNIST(data_dir)
            train_inputs = self.preprocess_inputs_dtype_shape(train_dataset.data)
            self = preprocess_inputs_for_zero_mean_and_unit_variance(
                self, train_inputs, axis=None, keepdims=False
            )
        
        if train:
            assert (classes_per_exp is not None)
            selected_inds = []
            for _class in classes_per_exp:
                class_inds = np.flatnonzero(self.targets == _class)

                selected_inds += [class_inds]
            
            # The inds are not shuffled
            selected_inds = np.concatenate(selected_inds)

            self.data = self.data[selected_inds]
            self.targets = self.targets[selected_inds]
            # The inds in the whole dataset of the points of the classes in the current task
            self.dataset_inds = selected_inds

    def __getitem__(self, index: int) -> Tuple[Array, Array]:
        return self.data[index], self.targets[index]


class EmbeddingSplitMNIST(BaseEmbeddingDataset):
    def __init__(self, data_dir: Union[Path, str], train, classes_per_exp, **kwargs: Any) -> None:
        data_dir = Path(data_dir) / "mnist"
        super().__init__(data_dir=data_dir, train=train, **kwargs)

        if train:
            assert (classes_per_exp is not None)
            selected_inds = []
            for _class in classes_per_exp:
                class_inds = np.flatnonzero(self.targets == _class)

                selected_inds += [class_inds]
            
            selected_inds = np.concatenate(selected_inds)

            self.data = self.data[selected_inds]
            self.targets = self.targets[selected_inds]
            self.dataset_inds = selected_inds

