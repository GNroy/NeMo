# Copyright (c) 2022, NVIDIA CORPORATION.  All rights reserved.
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
from omegaconf import DictConfig

from nemo.core.classes import NeuralModule
from nemo.core.neural_types import LengthsType, LogprobsType, NeuralType, PredictionsType


class ViterbiDecoderWithGraph(NeuralModule):
    @property
    def input_types(self):
        """Returns definitions of module input ports.
        """
        return {
            "log_probs": NeuralType(("B", "T", "D"), LogprobsType()),
            "log_probs_length": NeuralType(tuple("B"), LengthsType()),
        }

    @property
    def output_types(self):
        """Returns definitions of module output ports.
        """
        return {"predictions": NeuralType(("B", "T"), PredictionsType())}

    def __init__(
        self,
        num_classes,
        backend: str = "k2",
        dec_type: str = "topo",
        return_type: str = "1best",
        return_ilabels: bool = True,
        output_aligned: bool = True,
        split_batch_size: int = 0,
        graph_module_cfg: Optional[DictConfig] = None,
    ):
        self._blank = num_classes
        self.return_ilabels = return_ilabels
        self.output_aligned = output_aligned
        self.split_batch_size = split_batch_size
        self.dec_type = dec_type

        if return_type == "1best":
            self.return_lattices = False
        elif return_type == "lattice":
            self.return_lattices = True
        elif return_type == "nbest":
            raise NotImplementedError(f"return_type {return_type} is not supported at the moment")
        else:
            raise ValueError(f"Unsupported return_type: {return_type}")

        # we assume that self._blank + 1 == num_classes
        if backend == "k2":
            # use k2 import guard
            from nemo.core.utils.k2_utils import k2_import_guard

            k2_import_guard()

            if self.dec_type == "topo":
                from nemo.collections.asr.parts.k2.graph_decoders import BaseDecoder as Decoder
            elif self.dec_type == "tokenlm":
                from nemo.collections.asr.parts.k2.graph_decoders import TokenLMDecoder as Decoder
            elif self.dec_type == "looseali":
                raise NotImplementedError()
            elif self.dec_type == "tlg":
                raise NotImplementedError(f"dec_type {dec_type} is not supported at the moment")
            else:
                raise ValueError(f"Unsupported dec_type: {dec_type}")

            self._decoder = Decoder(num_classes=self._blank + 1, blank=self._blank, cfg=graph_module_cfg)
        elif backend == "gtn":
            raise NotImplementedError("gtn-backed decoding is not implemented")

        super().__init__()

    def update_graph(self, graph):
        self._decoder.update_graph(graph)

    def _forward_impl(self, log_probs, log_probs_length, targets=None, target_length=None):
        if targets is None and target_length is not None or targets is not None and target_length is None:
            raise RuntimeError(
                f"Both targets and target_length have to be None or not None: {targets}, {target_length}"
            )
        # do not use self.return_lattices for now
        if targets is None:
            align = False
            decode_func = lambda a, b: self._decoder.decode(
                a, b, return_lattices=False, return_ilabels=self.return_ilabels, output_aligned=self.output_aligned
            )
        else:
            align = True
            decode_func = lambda a, b, c, d: self._decoder.align(
                a, b, c, d, return_lattices=False, return_ilabels=False, output_aligned=True
            )
        batch_size = log_probs.shape[0]
        if self.split_batch_size > 0 and self.split_batch_size < batch_size:
            predictions = []
            probs = []
            for batch_idx in range(0, batch_size, self.split_batch_size):
                begin = batch_idx
                end = min(begin + self.split_batch_size, batch_size)
                log_probs_part = log_probs[begin:end]
                log_probs_length_part = log_probs_length[begin:end]
                if align:
                    targets_part = targets[begin:end]
                    target_length_part = target_length[begin:end]
                    predictions_part, probs_part = decode_func(
                        log_probs_part, log_probs_length_part, targets_part, target_length_part
                    )
                    del targets_part, target_length_part
                else:
                    predictions_part, probs_part = decode_func(log_probs_part, log_probs_length_part)
                del log_probs_part, log_probs_length_part
                predictions += predictions_part
                probs += probs_part
        else:
            predictions, probs = (
                decode_func(log_probs, log_probs_length, targets, target_length)
                if align
                else decode_func(log_probs, log_probs_length)
            )
        assert len(predictions) == len(probs)
        return predictions, probs

    @torch.no_grad()
    def forward(self, log_probs, log_probs_length):
        if self.dec_type == "looseali":
            raise RuntimeError(f"Decoder with dec_type=`{self.dec_type}` is not intended for regular decoding.")
        predictions, probs = self._forward_impl(log_probs, log_probs_length)
        lengths = torch.tensor([len(pred) for pred in predictions], device=predictions[0].device)
        predictions_tensor = torch.full((len(predictions), lengths.max()), self._blank).to(
            device=predictions[0].device
        )
        probs_tensor = torch.full((len(probs), lengths.max()), 1.0).to(device=predictions[0].device)
        for i, (pred, prob) in enumerate(zip(predictions, probs)):
            predictions_tensor[i, : lengths[i]] = pred
            probs_tensor[i, : lengths[i]] = prob
        return predictions_tensor, lengths, probs_tensor

    @torch.no_grad()
    def align(self, log_probs, log_probs_length, targets, target_length):
        len_enough = (log_probs_length >= target_length) & (target_length > 0)
        if torch.all(len_enough) or self.dec_type == "looseali":
            results = self._forward_impl(log_probs, log_probs_length, targets, target_length)
        else:
            results = self._forward_impl(
                log_probs[len_enough], log_probs_length[len_enough], targets[len_enough], target_length[len_enough]
            )
            for i, computed in enumerate(len_enough):
                if not computed:
                    results[0].insert(i, torch.Tensor().to(dtype=torch.int32))
                    results[1].insert(i, torch.Tensor().to(dtype=torch.float))
        return results
