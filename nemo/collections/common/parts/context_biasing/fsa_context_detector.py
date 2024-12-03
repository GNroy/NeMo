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

from abc import ABC, abstractmethod
from typing import List, Optional, Tuple

import torch

from nemo.core.utils.k2_guard import k2  # import k2 from guard module
import k2.ragged as k2r


class AbstractFsaContextDetector(ABC):
    """TBD"""
    def __init__(
        self,
        vocabulary_size: int,
        context_fsa: 'k2.Fsa',
        decoding_beam: float = 4.0,
        with_lookahead: bool = False,
    ):
        self.vocabulary_size = vocabulary_size
        self.context_fsa = context_fsa
        self.with_lookahead = with_lookahead
        self.decoding_beam = decoding_beam
        self._min_active_states = 1
        self._max_active_states = 10000

    @abstractmethod
    def detect(self, logits: torch.Tensor, logit_lengths: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        pass


class OfflineFsaContextDetector(AbstractFsaContextDetector):
    """TBD"""
    def __init__(
        self,
        vocabulary_size: int,
        context_fsa: 'k2.Fsa',
        decoding_beam: float = 4.0,
        with_lookahead: bool = False,
    ):
        super.__init__(vocabulary_size, context_fsa, decoding_beam, with_lookahead)

    def _get_lattice(self, logits: torch.Tensor, logit_lengths: torch.Tensor) -> "k2.Fsa":
        logprob = logits.log_softmax(dim=-1)
        eps_logprob = torch.zeros((logits.size(0), logits.size(1), 1), dtype=logits.dtype, device=logits.device)
        logprob_eps = torch.cat([logprob, eps_logprob], dim=-1)
        supervision_segment = (
            torch.stack(
                [
                    torch.arange(logprob_eps.size(0)),
                    torch.zeros(logprob_eps.size(0)),
                    logit_lengths.cpu(),
                ],
            )
            .t()
            .to(torch.int32)
        )
        dense_fsa_vec = k2.DenseFsaVec(logprob_eps, supervision_segment)

        lattice = k2.intersect_dense_pruned(
            a_fsas=self.context_fsa,
            b_fsas=dense_fsa_vec,
            search_beam=self.decoding_beam,
            output_beam=self.decoding_beam,
            min_active_states=self._min_active_states,
            max_active_states=self._max_active_states,
            seqframe_idx_name="seqframe_id",
            allow_partial=True,
        )
        # k2.RaggedShape is a non-tensor class so its objects are not propagated by binary FSA operations
        # i.e. it can be safely added
        setattr(lattice, "dense_fsa_shape", dense_fsa_vec.dense_fsa_vec.shape())

        return lattice

    def _lattice_to_sparse_context_tensors(self, lattice: "k2.Fsa") -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        kept_indices = (lattice.labels >= 0) & (lattice.labels < self.vocabulary_size)

        # k2.index_select is a bit faster than torch.index_select if both tensors are int32
        batch_ids = k2.index_select(lattice.dense_fsa_shape.row_ids(1), lattice.seqframe_id)[kept_indices]
        seqframe_ids = lattice.seqframe_id[kept_indices]
        frame_ids = seqframe_ids - k2.index_select(lattice.dense_fsa_shape.row_splits(1), batch_ids)

        _, sparse_indices_dim0, seqframe_ids_counts = torch.unique_consecutive(seqframe_ids, return_inverse=True, return_counts=True)
        sparse_indices_dim1 = lattice.labels[kept_indices].to(dtype=torch.int64)

        seqframe_ids_unique_indices = torch.zeros_like(seqframe_ids_counts)
        seqframe_ids_unique_indices[1:] = seqframe_ids_counts[:-1]
        seqframe_ids_unique_indices = seqframe_ids_unique_indices.cumsum(0).to(dtype=frame_ids.dtype)
        batch_ids_mapped = k2.index_select(batch_ids, seqframe_ids_unique_indices)
        frame_ids_mapped = k2.index_select(frame_ids, seqframe_ids_unique_indices)

        sparse_indices_size = (sparse_indices_dim0.max().item() + 1, self.vocabulary_size)
        sparse_indices = torch.stack([sparse_indices_dim0, sparse_indices_dim1])
        values = lattice.scores[kept_indices]

        # values with the same indices are summed using .coalesce()
        lattice_tokens_encoded = torch.sparse_coo_tensor(
            indices=sparse_indices,
            values=values.exp(),
            size=sparse_indices_size,
            device=lattice.device,
            requires_grad=lattice.scores.requires_grad,
        ).coalesce()

        if self.with_lookahead:
            lookahead = lattice.lookahead_tokens[(kept_indices.nonzero()[:,0]).to(dtype=torch.int32)]
            lookahead_sparse_indices_dim0 = sparse_indices_dim0[lookahead.shape.row_ids(1)]
            lookahead_sparse_indices_dim1 = lookahead.values.to(dtype=torch.int64)
            lookahead_sparse_indices = torch.stack([lookahead_sparse_indices_dim0, lookahead_sparse_indices_dim1])
            lookahead_values = (1 / k2r.RaggedTensor(lookahead.shape, torch.full((lookahead.numel(),), 1, dtype=torch.int32, device=lattice.device), device=lattice.device).sum())[lookahead.shape.row_ids(1)]

            lattice_lookahead_encoded = torch.sparse_coo_tensor(
                indices=lookahead_sparse_indices,
                values=lookahead_values,
                size=sparse_indices_size,
                device=lattice.device,
                requires_grad=lattice.scores.requires_grad,
            ).coalesce()
        else:
            lattice_lookahead_encoded = None

        return lattice_tokens_encoded, batch_ids_mapped, frame_ids_mapped, lattice_lookahead_encoded

    def detect(self, logits: torch.Tensor, logit_lengths: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        """TBD"""
        assert self.context_fsa.device == logits.device
        assert logits.size(-1) == self.vocabulary_size

        if logit_lengths is None:
            logit_lengths = torch.full(logits.size(1), device=logits.device)
        lattice = self._get_lattice(logits, logit_lengths)
        return self._lattice_to_sparse_context_tensors(lattice)


class OnlineFsaContextDetector(AbstractFsaContextDetector):
    """TBD"""
    def __init__(
        self,
        vocabulary_size: int,
        context_fsa: 'k2.Fsa',
        decoding_beam: float = 4.0,
        with_lookahead: bool = False,
    ):
        super.__init__(vocabulary_size, context_fsa, decoding_beam, with_lookahead)

        self.current_batch_size = 0
        self.online_intersecter = None
        self.decode_states = None
        self.previous_final_states = None
        self.fill_tensor = None

    def _init_decoder(self, batch_size: int):
        self.online_intersecter = k2.OnlineDenseIntersecter(
            decoding_graph=self.context_fsa,
            num_streams=batch_size,
            search_beam=self.decoding_beam,
            output_beam=self.decoding_beam,
            min_active_states=self._min_active_states,
            max_active_states=self._max_active_states,
            allow_partial=True,
        )
        self.decode_states = [k2.DecodeStateInfo()] * batch_size
        self.previous_final_states = [0] * batch_size
        self.current_batch_size = batch_size
        self.fill_tensor = torch.ones((batch_size, 1, self.vocab_size + 1), device=self.context_fsa.device).log_softmax(-1)

    def _get_lattice(self, logits: torch.Tensor) -> Tuple["k2.Fsa", List[int]]:
        """TBD"""
        logprob = logits.log_softmax(dim=-1)
        eps_logprob = torch.zeros((logits.size(0), logits.size(1), 1), dtype=logits.dtype, device=logits.device)
        logprob_eps = torch.cat([logprob, eps_logprob], dim=-1)

        # TODO: explain
        supervision_segment_helper = (
            torch.stack(
                [
                    torch.arange(logprob_eps.size(0)),
                    torch.zeros(logprob_eps.size(0)),
                    torch.ones(logprob_eps.size(0)),
                ],
            )
            .t()
            .to(torch.int32)
        )
        supervision_segment = (
            torch.stack(
                [
                    torch.arange(logprob_eps.size(0)),
                    torch.zeros(logprob_eps.size(0)),
                    torch.full(logprob_eps.size(0), 2),
                ],
            )
            .t()
            .to(torch.int32)
        )
        dense_fsa_vec_helper = k2.DenseFsaVec(logprob_eps, supervision_segment_helper)
        dense_fsa_vec = k2.DenseFsaVec(torch.cat([logprob_eps, self.fill_tensor], dim=1), supervision_segment)

        lattice, _ = intersector.decode(
            dense_fsa_vec, self.decode_states
        )
        lattice_helper, decode_states = intersector.decode(
            dense_fsa_vec_true, self.decode_states
        )
        self.decode_states = decode_states
        current_final_states = [lattice_helper[i].shape[0] - 1 for i in range(lattice_helper.shape[0])]

        return lattice, current_final_states

    def _lattice_to_sparse_context_tensors(self, lattice: "k2.Fsa", current_final_states: List[int]) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        assert lattice.shape[0] == len(current_final_states)

        kept_indices_list = []
        sparse_indices_list = []
        values_list = []
        batch_ids_mapped_list = []
        num_non_empty_frames = 0
        for j in range(lattice.shape[0]):
            jfsa = lattice[j]
            current_final_state = current_final_states[j]
            receiving_states = jfsa.arcs.values()[:, 1]
            kept_indices = (receiving_states >= self.previous_final_states[j]) & (receiving_states < current_final_state) & (jfsa.labels < self.vocabulary_size)
            kept_indices_list.append(kept_indices)
            if torch.any(kept_indices):
                sparse_indices_dim1 = jfsa.labels[kept_indices].to(dtype=torch.int64)
                sparse_indices_dim0 = torch.full_like(sparse_indices_dim1, num_non_empty_frames)
                num_non_empty_frames += 1
                batch_ids_mapped_list.append(j)
                sparse_indices_list.append(torch.stack([sparse_indices_dim0, sparse_indices_dim1]))
                values_list.append(jfsa.scores[kept_indices])
                self.previous_final_states[j] = current_final_state
            else:
                self.previous_final_states[j] = 0
                self.decode_states[j] = k2.DecodeStateInfo()
        batch_ids_mapped = torch.tensor(batch_ids_mapped_list, dtype=torch.int32)

        if num_non_empty_frames > 0:
            sparse_indices = torch.cat(sparse_indices_list, dim=1)
            values = torch.cat(values_list, dim=0).exp()
            sparse_indices_size = (sparse_indices[0].max().item() + 1, self.vocabulary_size)

            # values with the same indices are summed using .coalesce()
            lattice_tokens_encoded = torch.sparse_coo_tensor(
                indices=sparse_indices,
                values=values,
                size=sparse_indices_size,
                device=lattice.device,
                requires_grad=lattice.scores.requires_grad,
            ).coalesce()

            if self.with_lookahead:
                kept_indices = torch.cat(kept_indices_list, dim=0)
                lookahead = lattice.lookahead_tokens[(kept_indices.nonzero()[:,0]).to(dtype=torch.int32)]
                lookahead_sparse_indices_dim0 = sparse_indices_dim0[lookahead.shape.row_ids(1)]
                lookahead_sparse_indices_dim1 = lookahead.values.to(dtype=torch.int64)
                lookahead_sparse_indices = torch.stack([lookahead_sparse_indices_dim0, lookahead_sparse_indices_dim1])
                lookahead_values = (1 / k2r.RaggedTensor(lookahead.shape, torch.full((lookahead.numel(),), 1, dtype=torch.int32, device=lattice.device), device=lattice.device).sum())[lookahead.shape.row_ids(1)]

                lattice_lookahead_encoded = torch.sparse_coo_tensor(
                    indices=lookahead_sparse_indices,
                    values=lookahead_values,
                    size=sparse_indices_size,
                    device=lattice.device,
                    requires_grad=lattice.scores.requires_grad,
                ).coalesce()
            else:
                lattice_lookahead_encoded = None
        else:
            sparse_indices = (0, 0)
            sparse_indices = torch.LongTensor([]).reshape(2, 0) 
            values = torch.FloatTensor([])
            lattice_tokens_encoded = torch.sparse_coo_tensor(
                indices=sparse_indices,
                values=values,
                size=sparse_indices_size,
                device=lattice.device,
                requires_grad=lattice.scores.requires_grad,
            )
            lattice_lookahead_encoded = lattice_tokens_encoded.clone() if self.with_lookahead else None

        return lattice_tokens_encoded, batch_ids_mapped, lattice_lookahead_encoded

    def detect(self, logits: torch.Tensor, logit_lengths: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        """TBD"""
        assert self.context_fsa.device == logits.device
        assert logits.size(-1) == self.vocabulary_size
        assert logits.size(1) == 1

        if logits.size(0) != self.current_batch_size:
            self._init_decoder(logits.size(0))
        lattice, current_final_states = self._get_lattice(logits)
        lattice_tokens_encoded, batch_ids_mapped, lattice_lookahead_encoded = self._lattice_to_sparse_context_tensors(lattice, current_final_states)
        return lattice_tokens_encoded, batch_ids_mapped, None, lattice_lookahead_encoded
