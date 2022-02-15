#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import List, Optional

import torch
from torch import nn
from torchrec.modules.embedding_modules import EmbeddingBagCollection
from torchrec.modules.mlp import MLP
from torchrec.sparse.jagged_tensor import (
    KeyedJaggedTensor,
    KeyedTensor,
)

# Sphinx Documentation Text (for user-facing classes only)

"""
.. fb:display_title::
    DLRM API
=====
Notations uses throughout:

F: number of sparseFeatures
D: embedding_dimension of sparse features
B: batch_size
num_features: number of dense features

"""


def choose(n: int, k: int) -> int:
    """
    Simple implementation of math.comb for python 3.7 compatibility
    """
    if 0 <= k <= n:
        ntok = 1
        ktok = 1
        for t in range(1, min(k, n - k) + 1):
            ntok *= n
            ktok *= t
            n -= 1
        return ntok // ktok
    else:
        return 0


class SparseArch(nn.Module):
    """
    Processes the Sparse Features of DLRM. Does Embedding Lookup for all
    EmbeddingBag and Embedding features of each collection.

    Constructor Args:
        embedding_bag_collection: EmbeddingBagCollection,

    Call Args:
        features: KeyedJaggedTensor,

    Returns:
        KeyedJaggedTensor - size F * D X B

    Example:
        >>> eb1_config = EmbeddingBagConfig(
            name="t1", embedding_dim=3, num_embeddings=10, feature_names=["f1"]
        )
        eb2_config = EmbeddingBagConfig(
            name="t2", embedding_dim=4, num_embeddings=10, feature_names=["f2"]
        )
        ebc_config = EmbeddingBagCollectionConfig(tables=[eb1_config, eb2_config])

        ebc = EmbeddingBagCollection(config=ebc_config)

        #     0       1        2  <-- batch
        # 0   [0,1] None    [2]
        # 1   [3]    [4]    [5,6,7]
        # ^
        # feature
        features = KeyedJaggedTensor.from_offsets_sync(
            keys=["f1", "f2"],
            values=torch.tensor([0, 1, 2, 3, 4, 5, 6, 7]),
            offsets=torch.tensor([0, 2, 2, 3, 4, 5, 8]),
        )

        sparse_arch(features)
    """

    def __init__(self, embedding_bag_collection: EmbeddingBagCollection) -> None:
        super().__init__()
        self.embedding_bag_collection: EmbeddingBagCollection = embedding_bag_collection

    def forward(
        self,
        features: KeyedJaggedTensor,
    ) -> KeyedTensor:
        return self.embedding_bag_collection(features)


class DenseArch(nn.Module):
    """
    Processes the dense features of DLRM model.

    Constructor Args:
        in_features: int - size of the input.
        layer_sizes: List[int] - list of layer sizes.
        device: (Optional[torch.device]).

    Call Args:
        features: torch.Tensor  - size B X num_features

    Returns:
        torch.Tensor  - size B X D

    Example:
        >>> B = 20
        D = 3
        dense_arch = DenseArch(10, layer_sizes=[15, D])
        dense_embedded = dense_arch(torch.rand((B, 10)))
    """

    def __init__(
        self,
        in_features: int,
        layer_sizes: List[int],
        device: Optional[torch.device] = None,
    ) -> None:
        super().__init__()
        self.model: nn.Module = MLP(
            in_features, layer_sizes, bias=True, activation="relu", device=device
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.model(features)


class InteractionArch(nn.Module):
    """
    Processes the output of both SparseArch (sparse_features) and DenseArch
    (dense_features). Returns the pairwise dot product of each sparse feature pair,
    the dot product of each sparse features with the output of the dense layer,
    and the dense layer itself (all concatenated).

    NOTE: The dimensionality of the dense_features (D) is expected to match the
    dimensionality of the sparse_features so that the dot products between them can be
    computed.

    Constructor Args:
        sparse_feature_names: List[str] - size F

    Call Args:
        dense_features: torch.Tensor  - size B X D
        sparse_features: KeyedJaggedTensor - size F * D X B

    Returns:
        torch.Tensor - B X (D + F + F choose 2)

    Example:
        >>> D = 3
        B = 10
        keys = ["f1", "f2"]
        F = len(keys)
        inter_arch = InteractionArch(sparse_feature_names=keys)

        dense_features = torch.rand((B, D))

        sparse_features = KeyedTensor(
            keys=keys,
            length_per_key=[D, D],
            values=torch.rand((B, D * F)),
        )

        #  B X (D + F + F choose 2)
        concat_dense = inter_arch(dense_features, sparse_features)
    """

    def __init__(self, sparse_feature_names: List[str]) -> None:
        super().__init__()
        self.F: int = len(sparse_feature_names)
        self.triu_indices: torch.Tensor = torch.triu_indices(
            self.F + 1, self.F + 1, offset=1
        )
        self.sparse_feature_names = sparse_feature_names

    def forward(
        self, dense_features: torch.Tensor, sparse_features: KeyedTensor
    ) -> torch.Tensor:
        if self.F <= 0:
            return dense_features
        (B, D) = dense_features.shape

        sparse = sparse_features.to_dict()
        sparse_values = []
        for name in self.sparse_feature_names:
            sparse_values.append(sparse[name])

        sparse_values = torch.cat(sparse_values, dim=1).reshape(B, self.F, D)
        combined_values = torch.cat((dense_features.unsqueeze(1), sparse_values), dim=1)

        # dense/sparse + sparse/sparse interaction
        # size B X (F + F choose 2)
        interactions = torch.bmm(
            combined_values, torch.transpose(combined_values, 1, 2)
        )
        interactions_flat = interactions[:, self.triu_indices[0], self.triu_indices[1]]

        return torch.cat((dense_features, interactions_flat), dim=1)


class OverArch(nn.Module):
    """
    Final Arch of DLRM - simple MLP over OverArch.

    Constructor Args:
        in_features: int
        layer_sizes: list[int]
        device: (Optional[torch.device]).

    Call Args:
        features: torch.Tensor

    Returns:
        torch.Tensor  - size B X layer_sizes[-1]

    Example:
        >>> B = 20
        D = 3
        over_arch = OverArch(10, [5, 1])
        logits = over_arch(torch.rand((B, 10)))
    """

    def __init__(
        self,
        in_features: int,
        layer_sizes: List[int],
        device: Optional[torch.device] = None,
    ) -> None:
        super().__init__()
        if len(layer_sizes) <= 1:
            raise ValueError("OverArch must have multiple layers.")
        self.model: nn.Module = nn.Sequential(
            MLP(
                in_features,
                layer_sizes[:-1],
                bias=True,
                activation="relu",
                device=device,
            ),
            nn.Linear(layer_sizes[-2], layer_sizes[-1], bias=True, device=device),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.model(features)


class DLRM(nn.Module):
    """
    Recsys model from "Deep Learning Recommendation Model for Personalization and
    Recommendation Systems" (https://arxiv.org/abs/1906.00091). Processes sparse
    features by learning pooled embeddings for each feature. Learns the relationship
    between dense features and sparse features by projecting dense features into the
    same embedding space. Also, learns the pairwise relationships between sparse
    features.

    The module assumes all sparse features have the same embedding dimension
    (i.e, each EmbeddingBagConfig uses the same embedding_dim)

    Constructor Args:
        embedding_bag_collection (EmbeddingBagCollection): collection of embedding bags
            used to define SparseArch.
        dense_in_features (int): the dimensionality of the dense input features.
        dense_arch_layer_sizes (list[int]): the layer sizes for the DenseArch.
        over_arch_layer_sizes (list[int]): the layer sizes for the OverArch. NOTE: The
            output dimension of the InteractionArch should not be manually specified
            here.
        dense_device: (Optional[torch.device]).

    Call Args:
        dense_features: torch.Tensor,
        sparse_features: KeyedJaggedTensor,

    Returns:
        torch.Tensor - logits with size B X 1

    Example:
        >>> B = 2
        D = 8

        eb1_config = EmbeddingBagConfig(
            name="t1", embedding_dim=D, num_embeddings=100, feature_names=["f1", "f3"]
        )
        eb2_config = EmbeddingBagConfig(
            name="t2",
            embedding_dim=D,
            num_embeddings=100,
            feature_names=["f2"],
        )
        ebc_config = EmbeddingBagCollectionConfig(tables=[eb1_config, eb2_config])

        ebc = EmbeddingBagCollection(config=ebc_config)
        model = DLRM(
            embedding_bag_collection=ebc,
            dense_in_features=100,
            dense_arch_layer_sizes=[20],
            over_arch_layer_sizes=[5, 1],
        )

        features = torch.rand((B, 100))

        #     0       1
        # 0   [1,2] [4,5]
        # 1   [4,3] [2,9]
        # ^
        # feature
        sparse_features = KeyedJaggedTensor.from_offsets_sync(
            keys=["f1", "f3"],
            values=torch.tensor([1, 2, 4, 5, 4, 3, 2, 9]),
            offsets=torch.tensor([0, 2, 4, 6, 8]),
        )

        logits = model(
            dense_features=features,
            sparse_features=sparse_features,
        )
    """

    def __init__(
        self,
        embedding_bag_collection: EmbeddingBagCollection,
        dense_in_features: int,
        dense_arch_layer_sizes: List[int],
        over_arch_layer_sizes: List[int],
        dense_device: Optional[torch.device] = None,
    ) -> None:
        super().__init__()
        assert (
            len(embedding_bag_collection.embedding_bag_configs) > 0
        ), "At least one embedding bag is required"
        for i in range(1, len(embedding_bag_collection.embedding_bag_configs)):
            conf_prev = embedding_bag_collection.embedding_bag_configs[i - 1]
            conf = embedding_bag_collection.embedding_bag_configs[i]
            assert (
                conf_prev.embedding_dim == conf.embedding_dim
            ), "All EmbeddingBagConfigs must have the same dimension"
        embedding_dim: int = embedding_bag_collection.embedding_bag_configs[
            0
        ].embedding_dim
        if dense_arch_layer_sizes[-1] != embedding_dim:
            raise ValueError(
                f"embedding_bag_collection dimension ({embedding_dim}) and final dense "
                "arch layer size ({dense_arch_layer_sizes[-1]}) must match."
            )

        feature_names = [
            name
            for conf in embedding_bag_collection.embedding_bag_configs
            for name in conf.feature_names
        ]
        num_feature_names = len(feature_names)

        over_in_features = (
            embedding_dim + choose(num_feature_names, 2) + num_feature_names
        )

        self.sparse_arch = SparseArch(embedding_bag_collection)
        self.dense_arch = DenseArch(
            in_features=dense_in_features,
            layer_sizes=dense_arch_layer_sizes,
            device=dense_device,
        )
        self.inter_arch = InteractionArch(sparse_feature_names=feature_names)
        self.over_arch = OverArch(
            in_features=over_in_features,
            layer_sizes=over_arch_layer_sizes,
            device=dense_device,
        )

    def forward(
        self,
        dense_features: torch.Tensor,
        sparse_features: KeyedJaggedTensor,
    ) -> torch.Tensor:
        embedded_dense = self.dense_arch(dense_features)
        embedded_sparse = self.sparse_arch(sparse_features)
        concatenated_dense = self.inter_arch(
            dense_features=embedded_dense, sparse_features=embedded_sparse
        )
        logits = self.over_arch(concatenated_dense)
        return logits
