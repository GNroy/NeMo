# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import List, Optional

import torch
import torch.nn.functional is F
from torch import nn


class PretrainedTokenPositionEmbedding(nn.Module):
    """
    TBD
    Embedding from token and position embeddings.
    Optionally add token_type embedding (e.g. type of the sentence in BERT).

    Args:
        vocab_size: size of the vocabulary
        hidden_size: size of the embeddings in the model, also known as d_model
        max_sequence_length: maximum allowed length of the input sequence
        num_token_types: number of different token types
            (e.g. tokens of sentence A and tokens of sentence B in BERT)
        embedding_dropout: probability of dropout applied to embeddings
        learn_positional_encodings: whether to learn positional encodings or
            use fixed (sine-cosine) ones
    """

    def __init__(
        self,
        token_embedding_matrix: torch.Tensor,
        # position_embedding_module: nn.Module,
        left_context_depth: int = 0,
        use_pseudo_right_context: bool = False,
        dropout_ratio: float = 0.0,
    ):
        super().__init__()

        self.token_embedding_matrix = token_embedding_matrix.detach().clone()
        # self.position_embedding_module = position_embedding_module
        self.left_context_depth = left_context_depth
        self.use_pseudo_right_context = use_pseudo_right_context
        self.output_size = self.token_embedding_matrix.size(-1) * (1 + int(self.use_pseudo_right_context) + self.left_context_depth)
        self.layer_norm = nn.LayerNorm(self.output_size, eps=1e-5)
        self.dropout = nn.Dropout(dropout_ratio)

    def forward(
        self,
        context: torch.Tensor,
        batch_ids: Optional[torch.Tensor] = None,
        time_ids: Optional[torch.Tensor] = None,
        pseudo_right_context: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """TBD"""
        assert context.dim() == 2

        context_embeddings = []
        token_embeddings = torch.mm(context, self.token_embedding_matrix)
        # context_embeddings.append(token_embeddings + self.position_embedding_module(time_ids))
        context_embeddings.append(token_embeddings)

        if self.use_pseudo_right_context:
            if pseudo_right_context is None:
                raise RuntimeError("`pseudo_right_context` must be provided if `use_pseudo_right_context == True`")
            pseudo_right_token_embeddings = torch.mm(pseudo_right_context, self.token_embedding_matrix)
            # context_embeddings.append(pseudo_right_token_embeddings + self.position_embedding_module(time_ids + 1))
            context_embeddings.append(pseudo_right_token_embeddings)

        if self.left_context_depth > 0:
            if batch_ids is None or time_ids is None:
                raise RuntimeError("`batch_ids` and `time_ids` must be provided if `left_context_depth > 0`")
            embedding_len = token_embeddings.size(-1)
            left_context_embeddings = torch.zeros((token_embeddings.size(0), embedding_len * self.left_context_depth), device=token_embeddings.device)
            has_left_context_mask = torch.ones(token_embeddings.size(0), dtype=torch.bool, device=token_embeddings.device)
            pseudo_seqtime_ids = time_ids + batch_ids * (time_ids.max() + 2)
            time_ids_shifted = time_ids.clone()
            for i in range(min(self.left_context_depth, token_embeddings.size(0) - 1)):
                has_left_context_mask[i] = False
                has_left_context_mask[i+1:] &= pseudo_seqtime_ids[:-(i+1)] == (pseudo_seqtime_ids[i+1:] - (i+1))
                time_ids_shifted[time_ids_shifted > i] -= i+1
                # left_context_embeddings[:, embedding_len * i: embedding_len * (i+1)] = self.position_embedding_module(time_ids_shifted)
                left_context_embeddings[has_left_context_mask, embedding_len * i: embedding_len * (i+1)] += token_embeddings[has_left_context_mask.nonzero()[:,0] - (i+1)]
            context_embeddings.append(left_context_embeddings)

        embeddings = torch.cat(context_embeddings, dim=-1) if len(context_embeddings) > 1 else context_embeddings[0]
        embeddings = self.dropout(self.layer_norm(embeddings))

        return embeddings


class ContextFilter(nn.Module):
    """
    TBD
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        output_size: int,
        dropout_ratio: float = 0.0,
    ):
        super().__init__()

        self.linear1 = nn.Linear(in_features=input_size, out_features=hidden_size)
        self.batch_norm = nn.BatchNorm1d(num_features=hidden_size)
        self.dropout = nn.Dropout(dropout_ratio)
        self.linear2 = nn.Linear(in_features=hidden_size, out_features=output_size)

    def forward(self, context: torch.Tensor):
        """TBD"""
        assert context.dim() == 2

        y = self.dropout(F.relu(self.batch_norm(self.linear1(context))))
        y = self.linear2(y)
        return y
