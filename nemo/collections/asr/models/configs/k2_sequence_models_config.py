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

from dataclasses import dataclass

from nemo.collections.asr.models.configs.asr_models_config import EncDecCTCConfig
from nemo.collections.asr.parts.k2.classes import GraphModuleConfig as BackgroundConfig
from nemo.core.config.modelPT import NemoConfig


@dataclass
class GraphModuleConfig:
    criterion_type: str = "ml"
    split_batch_size: int = 0
    dec_type: str = "topo"
    transcribe_training: bool = True
    background_cfg: BackgroundConfig = BackgroundConfig()

@dataclass
class EncDecK2SeqConfig(EncDecCTCConfig):
    token_lm_overwrite: bool = False
    graph_module_cfg: GraphModuleConfig = GraphModuleConfig()

@dataclass
class EncDecK2SeqModelConfig(NemoConfig):
    model: EncDecK2SeqConfig = EncDecK2SeqConfig()
