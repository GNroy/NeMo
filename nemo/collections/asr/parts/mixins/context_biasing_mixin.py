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

from abc import ABC
from random import randint, sample
from typing import List, Optional

import torch
from omegaconf import DictConfig, OmegaConf

from nemo.collections.common.parts.context_biasing.context_biasing_module import FsaContextBiasingModule


class FsaContextBiasingMixin(ABC):
    """Confidence Mixin class.

    It is responsible for confidence estimation method initialization and high-level confidence score calculation.
    """

    def init_context_biasing(self,
        vocab_size: int,
        biasing_cfg: DictConfig,
        training: bool = False,
        token_embedding_matrix: Optional[torch.Tensor] = None,
        context: Optional[List[List[int]]] = None,
        training_logit_dropout: Optional[float] = None,
    ):
        """Initialize confidence-related fields and confidence aggregation function from config.
        """
        self.training = training
        self.context_biasing_module = FsaContextBiasingModule(
            vocab_size=vocab_size,
            cfg=cfg,
            token_embedding_matrix=token_embedding_matrix,
            context=context if not self.training else None,
        )
        self.training_context = context if self.training else None
        self.logit_dropout = torch.nn.Dropout(training_logit_dropout) if training_logit_dropout is not None else None

    def _sample_context(self):
        return sample(self.training_context, k=randint(1, len(self.training_context) - 1))

    def forward_biasing(self, logits: torch.Tensor, input_lengths: Optional[torch.Tensor]):
        if self.training:
            self.context_biasing_module.init_context_fsa(self._sample_context())
            if self.logit_dropout is not None:
                logits = self.logit_dropout(logits)
        return self.context_biasing_module.forward_one_step(logits) if logits.size(1) == 1 else self.context_biasing_module(logits, torch.full(logits.size(0), logits.size(1), device=logits.device))
