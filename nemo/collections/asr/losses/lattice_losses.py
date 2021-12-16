# ! /usr/bin/python
# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
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

import torch
from torch import nn

from nemo.collections.asr.parts.k2.grad_utils import PartialGrad
from nemo.core.classes import Loss, typecheck
from nemo.core.neural_types import (
    LabelsType,
    LengthsType,
    LogprobsType,
    LossType,
    NeuralType,
)
from nemo.utils import logging


class LatticeLoss(Loss):
    """TBD
    """

    @property
    def input_types(self):
        """Input types definitions for LatticeLoss.
        """
        return {
            "log_probs": NeuralType(("B", "T", "D"), LogprobsType()),
            "targets": NeuralType(("B", "T"), LabelsType()),
            "input_lengths": NeuralType(tuple("B"), LengthsType()),
            "target_lengths": NeuralType(tuple("B"), LengthsType()),
        }

    @property
    def output_types(self):
        """Output types definitions for LatticeLoss.
        loss:
            NeuralType(None)
        """
        return {"loss": NeuralType(elements_type=LossType())}

    def __init__(
        self,
        num_classes,
        reduction="mean_batch",
        backend="k2",
        criterion_type="ml",
        split_batch_size=0,
        **loss_kwargs,
    ):
        super().__init__()
        self._blank = num_classes
        self.split_batch_size = split_batch_size
        if reduction == "mean_batch":
            ctc_reduction = "none"
            self._apply_batch_mean = True
        elif reduction in ["sum", "mean", "none"]:
            ctc_reduction = reduction
            self._apply_batch_mean = False

        # we assume that self._blank + 1 == num_classes
        if backend == "k2":
            if criterion_type == "ml":
                from nemo.collections.asr.parts.k2.mlloss import MLLoss as K2Loss
            elif criterion_type == "map":
                from nemo.collections.asr.parts.k2.maploss import MAPLoss as K2Loss
            else:
                raise ValueError(
                    f"Invalid value of `criterion_type`: {criterion_type}."
                )

            self._loss = K2Loss(
                num_classes=self._blank + 1,
                blank=self._blank,
                reduction=ctc_reduction,
                **loss_kwargs,
            )
        elif backend == "gtn":
            raise NotImplementedError(f"Backend {backend} is not supported.")
        else:
            raise ValueError(f"Invalid value of `backend`: {backend}.")

        self.criterion_type = criterion_type

        if self.split_batch_size > 0:
            self._partial_loss = PartialGrad(self._loss)

    def update_graph(self, graph):
        if self.criterion_type != "ml":
            self._loss.update_graph(graph)

    @typecheck()
    def forward(self, log_probs, targets, input_lengths, target_lengths):
        # override forward implementation
        # custom logic, if necessary

        assert not (torch.isnan(log_probs).any() or torch.isinf(log_probs).any())

        log_probs = log_probs.float()
        input_lengths = input_lengths.long()
        target_lengths = target_lengths.long()
        targets = targets.long()
        batch_size = log_probs.shape[0]
        if self.split_batch_size > 0 and self.split_batch_size < batch_size:
            loss_list = []
            for batch_idx in range(0, batch_size, self.split_batch_size):
                begin = batch_idx
                end = min(begin + self.split_batch_size, batch_size)
                log_probs_part = log_probs[begin:end]
                targets_part = targets[begin:end]
                input_lengths_part = input_lengths[begin:end]
                target_lengths_part = target_lengths[begin:end]
                loss_part, _ = (
                    self._partial_loss(
                        log_probs_part,
                        targets_part,
                        input_lengths_part,
                        target_lengths_part,
                    )
                    if log_probs_part.requires_grad
                    else self._loss(
                        log_probs_part,
                        targets_part,
                        input_lengths_part,
                        target_lengths_part,
                    )
                )
                loss_list.append(loss_part)
            loss = torch.cat(loss_list, 0)
        else:
            loss, _ = self._loss(
                log_probs=log_probs,
                targets=targets,
                input_lengths=input_lengths,
                target_lengths=target_lengths,
            )
        if self._apply_batch_mean:
            # torch.mean gives nan if loss is empty
            loss = torch.mean(loss) if loss.nelement() > 0 else torch.sum(loss)
        return loss
