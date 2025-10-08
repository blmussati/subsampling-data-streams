"""
N = number of examples
H = height
W = width
"""

from pathlib import Path
from typing import Any, Tuple, Union

import numpy as np
from torch import Tensor
from torchvision.datasets import CIFAR10 as TorchVisionCIFAR10

from src.data.datasets.base import BaseDataset, BaseEmbeddingDataset
from src.data.input_preprocessing import preprocess_inputs_for_zero_mean_and_unit_variance
from src.typing import Array


class BaseSplitCIFAR10(BaseDataset):

    @staticmethod
    def preprocess_inputs_dtype_shape(inputs: Tensor) -> Tensor:
        inputs = inputs.astype(np.float32) / 255  # [N, H, W, Ch]
        return inputs.transpose(0, 3, 1, 2)  # [N, Ch, H, W]


class SplitCIFAR10(BaseSplitCIFAR10):
    def __init__(
        self,
        data_dir: Union[Path, str],
        classes_per_exp: np.ndarray = None,
        train: bool = True,
        input_preprocessing: str | None = "zero_mean_and_unit_variance",
        **kwargs: Any,
    ) -> None:
        data_dir = Path(data_dir) / "cifar10"
        dataset = TorchVisionCIFAR10(root=data_dir, download=True, train=train, **kwargs)

        self.data = self.preprocess_inputs_dtype_shape(dataset.data)
        self.targets = np.array(dataset.targets)

        if input_preprocessing == "zero_mean_and_unit_variance":
            train_dataset = TorchVisionCIFAR10(data_dir)
            train_inputs = self.preprocess_inputs_dtype_shape(train_dataset.data)
            self = preprocess_inputs_for_zero_mean_and_unit_variance(
                self, train_inputs, axis=(0, 2, 3), keepdims=True
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


class EmbeddingSplitCIFAR10(BaseEmbeddingDataset):
    def __init__(self, data_dir: Union[Path, str], train, classes_per_exp, **kwargs: Any) -> None:
        data_dir = Path(data_dir) / "cifar10"
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

