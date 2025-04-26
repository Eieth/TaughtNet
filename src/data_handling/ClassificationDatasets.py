import logging
from typing import List, Union

from torch import nn
from torch.utils.data.dataset import Dataset

from .DataClasses import SequenceInputFeatures, InputFeatures

logger = logging.getLogger(__name__)

class TokenClassificationDataset(Dataset):
    """
    This will be superseded by a framework-agnostic approach
    soon.
    """

    features: List[InputFeatures]
    pad_token_label_id: int = nn.CrossEntropyLoss().ignore_index
    # Use cross entropy ignore_index as padding label id so that only
    # real label ids contribute to the loss later.

    def __init__(
        self,
        features,
    ):
        self.features = features

    def __len__(self):
        return len(self.features)

    def __getitem__(self, i) -> InputFeatures:
        return self.features[i]

class SequenceClassificationDataset(Dataset):
    """
    This will be superseded by a framework-agnostic approach
    soon.
    """

    features: List[SequenceInputFeatures]

    def __init__(
        self,
        features,
    ):
        self.features = features

    def __len__(self):
        return len(self.features)

    def __getitem__(self, i) -> Union[SequenceInputFeatures, list[SequenceInputFeatures]]:
        return self.features[i]