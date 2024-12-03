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

import math
import struct
from dataclasses import dataclass
from typing import Dict, List, Optional

import torch
from torch.nn.utils.rnn import pad_sequence

from nemo.core.utils.k2_guard import k2  # import k2 from guard module
import k2.ragged as k2r


@dataclass
class TrieFsaFactoryConfig:
    fail_score_multiplier: float = 1.0
    unit_cost_multiplier: float = 0.0
    with_lookahead: bool = False


class TrieFsaFactory:
    """TBD"""
    def __init__(
        self,
        vocabulary_size: int,
        fail_score_multiplier: float = 1.0,
        unit_cost_multiplier: float = 0.0,
        with_lookahead: bool = False,
        device: str = "cpu",
        symbol_table: Optional[Dict[int, str]] = None,
    ):
        self.vocabulary_size = vocabulary_size + 1
        self.fail_score_multiplier = fail_score_multiplier
        self.unit_cost_multiplier = unit_cost_multiplier
        self.with_lookahead = with_lookahead
        self.device = device
        self.symbol_table = symbol_table
        self.star_id = vocabulary_size
        self._pad_token = -1
        self._fail_score = math.log(self.fail_score_multiplier / vocabulary_size)

    def _float_to_int(self, f: float) -> int:
        f = struct.pack('f', f)
        return int.from_bytes(f, 'little', signed=True)

    def _float_to_int_tensor(self, f: torch.Tensor) -> torch.Tensor:
        f_list = f.tolist()
        i_list = struct.unpack(f'{len(f_list)}i', struct.pack(f'{len(f_list)}f', *f_list))
        return torch.tensor(i_list, dtype=f.dtype, device=f.device)

    def build(self, token_ids: List[List[int]]) -> 'k2.Fsa':
        """TBD"""
        padded_sorted_token_ids = pad_sequence([torch.tensor(item, dtype=torch.int32, device=self.device) for item in sorted(token_ids)], batch_first=True, padding_value=-1).T

        assert self.star_id > padded_sorted_token_ids.max()

        arc_chunk_list = []
        arc_chunk_list.append(torch.tensor([0, 0, self.star_id, self._float_to_int(self._fail_score)], dtype=torch.int32, device=self.device).unsqueeze(0))

        no_pad_mask = padded_sorted_token_ids != self._pad_token
        nonfinal_token_indicator = torch.logical_xor(no_pad_mask, ~torch.cat([no_pad_mask[1:], torch.full((1, no_pad_mask.size(1)), False, dtype=torch.bool, device=no_pad_mask.device)])).to(dtype=torch.int32)
        branch_indices = torch.zeros(padded_sorted_token_ids.size(1), dtype=torch.int32, device=self.device)

        start_node = 1
        for i, token_slice in enumerate(padded_sorted_token_ids):
            current_mask = no_pad_mask[i]
            current_nonfinal_indicator = nonfinal_token_indicator[i]

            unique_tokens_container, inverse_indices = torch.unique_consecutive(torch.stack([token_slice[current_mask], branch_indices[current_mask], current_nonfinal_indicator[current_mask]], dim=1).T, dim=1, return_inverse=True)
            container_size = unique_tokens_container.size(1)

            new_nodes_number = unique_tokens_container[2].sum()
            new_nodes_to = torch.arange(start_node, start_node + new_nodes_number, dtype=torch.int32, device=self.device)
            nodes_to = torch.zeros(container_size, dtype=torch.int32, device=self.device)
            nodes_to[unique_tokens_container[2].to(dtype=torch.bool)] = new_nodes_to
            nodes_from = torch.zeros_like(nodes_to) if i == 0 else unique_tokens_container[1]

            arc_chunk = torch.zeros((container_size + new_nodes_number, 4), dtype=nodes_to.dtype, device=nodes_to.device)
            arc_chunk[:container_size,:3] = torch.stack([nodes_from, nodes_to, unique_tokens_container[0]], dim=1)
            if i == 0:
                unit_cost = self._float_to_int(math.log(self.unit_cost_multiplier / container_size)) if self.unit_cost_multiplier > 0 else 0.0
                arc_chunk[:container_size,3] = torch.full_like(nodes_from, unit_cost)
            arc_chunk[container_size:] = torch.stack([
                new_nodes_to,
                torch.zeros_like(new_nodes_to),
                torch.full_like(new_nodes_to, self.star_id),
                torch.full_like(new_nodes_to, self._float_to_int(self._fail_score * (i + 2)))
            ], dim=1)
            if self.with_lookahead:
                # alaptev: I don't know how to implement it without python loops
                if i == 0:
                    lookahead_list = [[]]
                    num_non_state_arcs = 0
                else:
                    j = 0
                    unique, counts = torch.unique_consecutive(unique_tokens_container[1], return_counts=True)
                    unique += previous_nonfinal_indicator.nonzero()[:,0] - torch.arange(unique.size(0), device=unique.device) + num_non_state_arcs
                    for u, c in zip(unique, counts):
                        lookahead_list[u] += unique_tokens_container[0, j:j+c].tolist()
                        j += c
                    num_non_state_arcs += previous_container_size
                lookahead_list += [[] for _ in range(arc_chunk.size(0))]
                previous_container_size = container_size
                previous_nonfinal_indicator = unique_tokens_container[2]
            arc_chunk_list.append(arc_chunk)

            branch_indices[current_mask] = nodes_to[inverse_indices]
            start_node += new_nodes_number
        final_arc = torch.tensor([0, start_node, -1, 0], dtype=torch.int32, device=self.device).unsqueeze(0)
        arc_chunk_list.append(final_arc)
        arcs = torch.cat(arc_chunk_list)
        sorted_indices = arcs[:, 0].sort()[1]
        arcs_sorted = arcs[sorted_indices]
        fsa = k2.Fsa(arcs_sorted)
        if self.with_lookahead:
            lookahead_list.append([])
            lookahead_list = [lookahead_list[si] for si in sorted_indices]
            setattr(fsa, "lookahead_tokens", k2r.RaggedTensor(lookahead_list, device=self.device))
        fsa = k2.arc_sort(fsa)

        if self.symbol_table is not None:
            fsa.labels_sym = k2.SymbolTable.from_str("\n".join([f"{t} {i}" for i, t in self.symbol_table.items()]))
        return fsa
