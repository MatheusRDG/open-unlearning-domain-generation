import logging
import random

import torch
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)


class ForgetRetainDataset(Dataset):
    # https://github.com/OPTML-Group/SOUL/blob/main/src/dataset/Base.py
    def __init__(self, forget, retain, anchor="forget", seed=42):
        """Wraps the forget retain dataset into unlearning dataset.

        Args:
            forget (Dataset): Forget Dataset
            retain (Dataset): Retain Dataset
            anchor (str, optional): Specifies which dataset to anchor while randomly sampling from the other dataset. Defaults to 'forget'.
            seed (int, optional): Random seed for deterministic shuffling. Defaults to 42.
        """
        self.forget = forget
        self.retain = retain
        self.anchor = anchor
        self.seed = seed

        # Validate datasets are not empty
        if self.forget is not None and len(self.forget) == 0:
            raise ValueError(
                "Forget dataset is empty! No data to unlearn. "
                "Please check that your dataset generation produced QA pairs."
            )
        if self.retain is not None and len(self.retain) == 0:
            raise ValueError(
                "Retain dataset is empty! No data to retain. "
                "Please check that your dataset generation produced QA pairs."
            )

        # Log dataset sizes for debugging
        forget_size = len(self.forget) if self.forget is not None else 0
        retain_size = len(self.retain) if self.retain is not None else 0
        logger.info(
            "ForgetRetainDataset initialized: forget=%d samples, retain=%d samples, anchor='%s', seed=%d",
            forget_size,
            retain_size,
            anchor,
            seed,
        )
        
        # Initialize deterministic shuffled indices for sampling
        self._initialize_sampling_indices()

    def _initialize_sampling_indices(self):
        """Initialize shuffled indices for deterministic sampling."""
        rng = random.Random(self.seed)
        
        if self.anchor == "forget" and self.retain is not None:
            # Create shuffled indices for retain set
            self._retain_indices = list(range(len(self.retain)))
            rng.shuffle(self._retain_indices)
            self._retain_counter = 0
        elif self.anchor == "retain" and self.forget is not None:
            # Create shuffled indices for forget set
            self._forget_indices = list(range(len(self.forget)))
            rng.shuffle(self._forget_indices)
            self._forget_counter = 0

    def __len__(self):
        """Ensures the sampled dataset matches the anchor dataset's length."""
        if self.anchor == "forget":
            assert self.forget is not None, ValueError(
                "forget dataset can't be None when anchor=forget"
            )
            return len(self.forget)
        elif self.anchor == "retain":
            assert self.retain is not None, ValueError(
                "retain dataset can't be None when anchor=retain"
            )
            return len(self.retain)
        else:
            raise NotImplementedError(f"{self.anchor} can be only forget or retain")

    def __getitem__(self, idx):
        item = {}
        if self.anchor == "forget":
            item["forget"] = self.forget[idx]
            if self.retain:
                # Deterministic sampling with wrap-around
                retain_idx = self._retain_indices[self._retain_counter % len(self.retain)]
                self._retain_counter += 1
                item["retain"] = self.retain[retain_idx]
        elif self.anchor == "retain":
            item["retain"] = self.retain[idx]
            if self.forget:
                # Deterministic sampling with wrap-around
                forget_idx = self._forget_indices[self._forget_counter % len(self.forget)]
                self._forget_counter += 1
                item["forget"] = self.forget[forget_idx]
        return item
