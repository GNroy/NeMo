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

from typing import Dict, List, Optional, Union

import numpy as np
import torch
from omegaconf import DictConfig

from nemo.collections.asr.data.audio_to_ctm_dataset import FrameCtmUnit
from nemo.collections.asr.data.audio_to_text_dali import DALIOutputs
from nemo.collections.asr.models.asr_model import ASRModel
from nemo.collections.asr.models.ctc_models import EncDecCTCModel
from nemo.collections.asr.models.rnnt_models import EncDecRNNTModel
from nemo.collections.asr.modules.graph_decoder import ViterbiDecoderWithGraph


class AlignerWrapperModel(ASRModel):
    """ASR model wrapper to perform alignment building.
    Functionality is limited to the components needed to build an alignment."""

    def __init__(self, model: ASRModel, cfg: DictConfig):
        model_cfg = model.cfg
        for ds in ("train_ds", "validation_ds", "test_ds"):
            if ds in model_cfg:
                model_cfg[ds] = None
        super().__init__(cfg=model_cfg, trainer=model.trainer)
        self._model = model
        self.alignment_type = cfg.get("alignment_type", "forced")
        self.word_output = cfg.get("word_output", True)
        self.cpu_decoding = cfg.get("cpu_decoding", False)
        self.blank_id = self._model.decoder.num_classes_with_blank - 1
        decode_batch_size = cfg.get("decode_batch_size", 0)
        prob_suppress_index = cfg.get("prob_suppress_index", -1)
        prob_suppress_value = cfg.get("prob_suppress_value", 1.0)
        if prob_suppress_value > 1 or prob_suppress_value <= 0:
            raise ValueError(f"Suppression value has to be in (0,1]: {prob_suppress_value}")
        if prob_suppress_index < -(self.blank_id + 1) or prob_suppress_index > self.blank_id:
            raise ValueError(
                f"Suppression index for the provided model has to be in [{-self.blank_id+1},{self.blank_id}]: {prob_suppress_index}"
            )
        self.prob_suppress_index = (
            self._model.decoder.num_classes_with_blank + prob_suppress_index
            if prob_suppress_index < 0
            else prob_suppress_index
        )
        self.prob_suppress_value = prob_suppress_value

        # list possible alignment types here for future work
        if self.alignment_type == "forced":
            pass
        elif self.alignment_type == "argmax":
            pass
        elif self.alignment_type == "loose":
            raise NotImplementedError("alignment_type=`{self.alignment_type}` is not supported at the moment.")
        elif self.alignment_type == "rnnt_decoding_aux":
            raise NotImplementedError("alignment_type=`{self.alignment_type}` is not supported at the moment.")
        else:
            raise RuntimeError(f"Unsupported alignment type: {self.alignment_type}")

        if isinstance(self._model, EncDecCTCModel):
            self.model_type = "ctc"
            if self.alignment_type == "forced":
                if hasattr(self._model, "use_graph_lm"):
                    if self._model.use_graph_lm:
                        self.graph_decoder = self._model.transcribe_decoder
                        self._model.use_graph_lm = False
                    else:
                        self.graph_decoder = ViterbiDecoderWithGraph(
                            num_classes=self.blank_id, backend="k2", dec_type="topo", return_type="1best"
                        )
                    # override split_batch_size
                    self.graph_decoder.split_batch_size = decode_batch_size
                else:
                    self.graph_decoder = ViterbiDecoderWithGraph(
                        num_classes=self.blank_id, split_batch_size=decode_batch_size,
                    )
            elif self.alignment_type == "argmax":
                if hasattr(self._model, "use_graph_lm"):
                    if not self._model.use_graph_lm:
                        self._model.transcribe_decoder = ViterbiDecoderWithGraph(
                            num_classes=self.blank_id, backend="k2", dec_type="topo", return_type="1best"
                        )
                    # override decoder args
                    self._model.transcribe_decoder.return_ilabels = False
                    self._model.transcribe_decoder.output_aligned = True
                    self._model.transcribe_decoder.split_batch_size = decode_batch_size
                    self._model.use_graph_lm = False
        elif isinstance(self._model, EncDecRNNTModel):
            self.model_type = "rnnt"
            raise NotImplementedError("Only CTC models are supported at the moment.")
        else:
            raise RuntimeError(f"Unsupported model type: {type(self._model)}")

    def _apply_prob_suppress(self, log_probs):
        exp_probs = (log_probs).exp()
        x = exp_probs[:, :, self.prob_suppress_index]
        # we cannot do y=1-x because exp_probs can be not stochastic due to numerical limitations
        y = torch.cat(
            [exp_probs[:, :, : self.prob_suppress_index], exp_probs[:, :, self.prob_suppress_index + 1 :]], 2
        ).sum(-1)
        b1 = torch.full((exp_probs.shape[0], exp_probs.shape[1], 1), self.prob_suppress_value, device=log_probs.device)
        b2 = ((1 - self.prob_suppress_value * x) / y).unsqueeze(2).repeat(1, 1, exp_probs.shape[-1] - 1)
        return (
            exp_probs * torch.cat([b2[:, :, : self.prob_suppress_index], b1, b2[:, :, self.prob_suppress_index :]], 2)
        ).log()

    def _prepare_argmax_predictions(self, log_probs, encoded_len):
        if hasattr(self._model, "transcribe_decoder"):
            predictions, _, probs = self.transcribe_decoder.forward(log_probs=log_probs, log_probs_length=encoded_len)
        else:
            greedy_predictions = log_probs.argmax(dim=-1, keepdim=False)
            probs_tensor, _ = log_probs.exp().max(dim=-1, keepdim=False)
            predictions, probs = [], []
            for i in range(log_probs.shape[0]):
                utt_len = encoded_len[i]
                probs.append(probs_tensor[i, :utt_len])
                pred_candidate = greedy_predictions[i, :utt_len].cpu()
                # replace consecutive tokens with <blank>
                previous = self.blank_id
                for j in range(utt_len):
                    p = pred_candidate[j]
                    if p == previous and previous != self.blank_id:
                        pred_candidate[j] = self.blank_id
                    previous = p
                predictions.append(pred_candidate.to(device=greedy_predictions.device))
        return predictions, probs

    def _results_to_ctmUnits(self, s_id, pred, prob):
        if len(pred) == 0:
            return (s_id, [])
        else:
            non_blank_idx = (pred != self.blank_id).nonzero(as_tuple=True)[0].tolist()
            pred_ids = pred[non_blank_idx].tolist()
            tokens = self._model._wer.decode_ids_to_tokens(pred_ids)
            prob_list = prob.tolist()
            token_begin = non_blank_idx
            token_len, token_prob = [], []
            for i, j in zip(token_begin, token_begin[1:] + [len(pred)]):
                t_l = j - i
                token_len.append(t_l)
                token_prob.append(sum(prob_list[i:j]) / (t_l))
            if self.word_output:
                words = [w for w in self._model._wer.decode_tokens_to_str(pred_ids).split(" ") if w != ""]
                if hasattr(self._model, "tokenizer"):
                    # suppose that there are no whitespaces
                    assert len(self._model.tokenizer.text_to_tokens(words[0])) == len(
                        self._model.tokenizer.text_to_tokens(words[0] + " ")
                    )
                    word_begin, word_len, word_prob = [], [], []
                    i = 0
                    for word in words:
                        j = i + len(self._model.tokenizer.text_to_tokens(word))
                        word_begin.append(token_begin[i])
                        word_len.append(sum(token_len[i:j]))
                        word_prob.append(sum(token_prob[k] * token_len[k] for k in range(i, j)) / word_len[-1])
                        i = j
                else:
                    # suppose that there are no whitespaces anywhere except between words
                    space_idx = (np.array(tokens) == " ").nonzero()[0].tolist()
                    assert len(words) == len(space_idx) + 1
                    if len(space_idx) == 0:
                        word_begin = [token_begin[0]]
                        word_len = [sum(token_len)]
                        word_prob = [sum(t_p * t_l for t_p, t_l in zip(token_prob, token_len)) / word_len[0]]
                    else:
                        space_word = "[SEP]"
                        word_begin = [token_begin[0]]
                        word_len = [sum(token_len[: space_idx[0]])]
                        word_prob = [sum(token_prob[k] * token_len[k] for k in range(space_idx[0])) / word_len[-1]]
                        words_with_space = [words[0]]
                        for word, i, j in zip(words[1:], space_idx, space_idx[1:] + [len(tokens)]):
                            # append space
                            word_begin.append(token_begin[i])
                            word_len.append(token_len[i])
                            word_prob.append(token_prob[i])
                            words_with_space.append(space_word)
                            # append next word
                            word_begin.append(token_begin[i + 1])
                            word_len.append(sum(token_len[i + 1 : j]))
                            word_prob.append(sum(token_prob[k] * token_len[k] for k in range(i + 1, j)) / word_len[-1])
                            words_with_space.append(word)
                        words = words_with_space
                return (s_id, [FrameCtmUnit(t, b, l, p) for t, b, l, p in zip(words, word_begin, word_len, word_prob)])
            else:
                return (
                    s_id,
                    [FrameCtmUnit(t, b, l, p) for t, b, l, p in zip(tokens, token_begin, token_len, token_prob)],
                )

    @torch.no_grad()
    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        signal, signal_len, transcript, transcript_len, sample_id = batch

        if self.model_type == "ctc":
            if isinstance(batch, DALIOutputs) and batch.has_processed_signal:
                log_probs, encoded_len, _ = self._model.forward(
                    processed_signal=signal, processed_signal_length=signal_len
                )
            else:
                log_probs, encoded_len, _ = self._model.forward(input_signal=signal, input_signal_length=signal_len)
        elif self.model_type == "rnnt":
            raise NotImplementedError("Only CTC models are supported at the moment.")
        else:
            raise RuntimeError(f"Unsupported model type: {type(self._model)}")

        if self.prob_suppress_value != 1.0:
            log_probs = self._apply_prob_suppress(log_probs)

        if self.alignment_type == "argmax":
            predictions, probs = self._prepare_argmax_predictions(log_probs, encoded_len)
        elif self.alignment_type == "forced":
            if self.cpu_decoding:
                log_probs, encoded_len, transcript, transcript_len = (
                    log_probs.cpu(),
                    encoded_len.cpu(),
                    transcript.cpu(),
                    transcript_len.cpu(),
                )
            predictions, probs = self.graph_decoder.align(log_probs, encoded_len, transcript, transcript_len)
        else:
            raise NotImplementedError()

        return [
            self._results_to_ctmUnits(s_id, pred, prob)
            for s_id, pred, prob in zip(sample_id.tolist(), predictions, probs)
        ]

    @torch.no_grad()
    def transcribe(
        self,
        paths2audio_files: List[str],
        batch_size: int = 4,
        logprobs: bool = False,
        return_hypotheses: bool = False,
        num_workers: int = None,
    ) -> List[str]:
        raise NotImplementedError()

    def setup_training_data(self, train_data_config: Optional[Union[DictConfig, Dict]]):
        raise RuntimeError("This module cannot be used in training.")

    def setup_validation_data(self, val_data_config: Optional[Union[DictConfig, Dict]]):
        raise RuntimeError("This module cannot be used in validation.")

    def setup_test_data(self, val_data_config: Optional[Union[DictConfig, Dict]]):
        raise RuntimeError("This module cannot be used in testing.")
