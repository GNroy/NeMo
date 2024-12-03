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
from collections import OrderedDict
from typing import List, Optional

import torch

from nemo.collections.common.parts.context_biasing.utility_modules import ContextFilter, PretrainedTokenPositionEmbedding
from nemo.core.classes.common import typecheck
from nemo.core.classes.module import NeuralModule
from nemo.core.neural_types import LogitsType, LengthsType, NeuralType


class AbstractContextBiasingModule(NeuralModule, ABC):
    """TBD"""

    @property
    def input_types(self):
        """Returns definitions of module input ports.
        """
        return OrderedDict({
            "logits": NeuralType(("B", "T", "D"), LogitsType()),
            "input_lengths": NeuralType(tuple("B"), LengthsType()),
        })

    @property
    def output_types(self):
        return OrderedDict({"logits": NeuralType(('B', 'T', 'D'), LogitsType())})


class FsaContextBiasingModule(AbstractContextBiasingModule):
    """TBD"""
    def __init__(
        self,
        vocab_size: int,
        cfg: DictConfig,
        token_embedding_matrix: Optional[torch.Tensor] = None,
        context: Optional[List[List[int]]] = None,
    ):
        super().__init__()

        self.vocab_size = vocab_size
        self._cfg = cfg

        self.use_pseudo_right_context = self._cfg.use_pseudo_right_context or self._cfg.fsa_config.with_lookahead
        self.mode = ""
        self._detector = None
        self._frame_buffer = [None] * self._cfg.left_context_depth

        if context is not None:
            self.init_context_fsa(context)
        else:
            self._tff = None
            self.context_fsa = None
        if token_embedding_matrix is not None:
            self.init_neural_submodules(token_embedding_matrix)
        else:
            self.embedding = None
            self.filter = None
            self.final_layer = None

    def init_neural_submodules(self, token_embedding_matrix: torch.Tensor):
        self.embedding = PretrainedTokenPositionEmbedding(
            token_embedding_matrix=token_embedding_matrix,
            left_context_depth=self._cfg.left_context_depth,
            use_pseudo_right_context=self.use_pseudo_right_context,
            dropout_ratio=self._cfg.embedding_dropout,
        )
        self.filter = ContextFilter(
            input_size=self.embedding.output_size,
            hidden_size=self._cfg.filter_size,
            output_size=token_embedding_matrix.size(-1),
            dropout_ratio=self._cfg.filter_dropout,
        )
        self.final_layer = torch.nn.Linear(token_embedding_matrix.size(-1), self.vocab_size)
        self.final_layer.weight = torch.nn.Parameter(token_embedding_matrix)
        self.final_layer.requires_grad = False

    def init_context_fsa(self, context: List[List[int]]):
        if self._tff is None:
            from nemo.collections.common.parts.context_biasing.context_fsa import TrieFsaFactory

            self._tff = TrieFsaFactory(
                vocabulary_size=self.vocab_size,
                fail_score_multiplier=self._cfg.fsa_config.fail_score_multiplier,
                unit_cost_multiplier=self._cfg.fsa_config.unit_cost_multiplier,
                with_lookahead=self.use_pseudo_right_context,
                device=self.device,
            )
        self.context_fsa = self._tff.build(context)

        if self._detector is not None:
            self._detector.context_fsa = self.context_fsa

    def _forward_self_check(self):
        if self.embedding is None or self.filter is None:
            raise RuntimeError(
                """Context biasing submodules were not initialized.\n
                Run `init_neural_submodules` before calling `forward`."""
            )
        if self.context_fsa is None:
            raise RuntimeError(
                """Context biasing FSA is empty.\n
                Run `init_context_fsa` with the desired context before calling `forward`."""
            )

    @typecheck()
    def forward(self, logits, input_lengths):
        """TBD"""
        self._forward_self_check()

        if self._detector is None or self.mode != "offline":
            from nemo.collections.common.parts.context_biasing.fsa_context_detector import OfflineFsaContextDetector

            self._detector = OfflineFsaContextDetector(
                vocabulary_size=self.vocab_size,
                context_fsa=self.context_fsa,
                with_lookahead=self.use_pseudo_right_context,
            )
            self.mode = "offline"
        context, batch_ids, time_ids, pseudo_right_context = self._detector.detect(logits, input_lengths)

        context_embeddings_large = self.embedding(
            context=context,
            batch_ids=batch_ids,
            time_ids=time_ids,
            pseudo_right_context=pseudo_right_context,
        )
        context_embeddings = self.filter(context_embeddings_large)
        context_logits = self.final_layer(context_embeddings)

        new_logits = logits.clone()
        new_logits[batch_ids, time_ids] += context_logits
        return new_logits

    def forward_one_step(self, logits):
        """TBD"""
        self._forward_self_check()

        if self._detector is None or self.mode != "online":
            from nemo.collections.common.parts.context_biasing.fsa_context_detector import OnlineFsaContextDetector

            self._detector = OnlineFsaContextDetector(
                vocabulary_size=self.vocab_size,
                context_fsa=self.context_fsa,
                with_lookahead=self.use_pseudo_right_context,
            )
            self.mode = "online"
        context, batch_ids, _, pseudo_right_context = self._detector.detect(logits)

        if len(batch_ids) == 0:
            if self._cfg.left_context_depth > 0:
                self._frame_buffer = [None] * self._cfg.left_context_depth
            return logits

        if self._cfg.left_context_depth > 0:
            time_ids = torch.zeros_like(batch_ids) if self._frame_buffer[0] is None else self._frame_buffer[0][2] + 1
            context_list, batch_ids_list, time_ids_list = [context], [batch_ids], [time_ids]
            for buffer_item in self._frame_buffer:
                if self._frame_buffer[i] is not None:
                    context_list.append(buffer_item[0])
                    batch_ids_list.append(buffer_item[1])
                    time_ids_list.append(buffer_item[2])
                else:
                    break
            context_embeddings_large = self.embedding(
                context=torch.cat(context_list[::-1], dim=0),
                batch_ids=torch.cat(batch_ids_list[::-1]),
                time_ids=torch.cat(time_ids_list[::-1]),
                pseudo_right_context=pseudo_right_context
            )
            self._frame_buffer[1:] = self._frame_buffer[:-1]
            self._frame_buffer[0] = [context, batch_ids, time_ids]
        else:
            context_embeddings_large = self.embedding(context=context, pseudo_right_context=pseudo_right_context)
        context_embeddings = self.filter(context_embeddings_large)
        context_logits = self.final_layer(context_embeddings)

        new_logits = logits.clone()
        new_logits[batch_ids, 0] += context_logits
        return new_logits
