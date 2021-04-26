# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
# Copyright 2015 and onwards Google, Inc.
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

from nemo_text_processing.text_normalization.data_loader_utils import get_abs_path
from nemo_text_processing.text_normalization.graph_utils import NEMO_NOT_QUOTE, NEMO_SPACE, GraphFst, delete_space

try:
    import pynini
    from pynini.lib import pynutil

    PYNINI_AVAILABLE = True
except (ModuleNotFoundError, ImportError):
    PYNINI_AVAILABLE = False


class ElectronicFst(GraphFst):
    """
    Finite state transducer for verbalizing electronic
        e.g. tokens { email { username: "a.bc" servername: "g m a i l " domain: "com" } } -> a.bc et g m a i l dot com

    """

    def __init__(self):
        super().__init__(name="electronic", kind="verbalize")
        graph_digit = pynini.invert(pynini.string_file(get_abs_path("data/numbers/digit.tsv"))).optimize()
        user_name = (
            pynutil.delete("username:")
            + delete_space
            + pynutil.delete("\"")
            + (
                pynini.closure(
                    pynutil.add_weight(graph_digit + pynutil.insert(' '), 1.09)
                    | pynutil.add_weight(pynini.closure(pynini.cross(".", "dot ")), 1.09)
                    | pynutil.add_weight(NEMO_NOT_QUOTE + pynutil.insert(' '), 1.1)
                )
            )
            + pynutil.delete("\"")
        )

        domain_default = (
            pynini.closure(NEMO_NOT_QUOTE + pynutil.insert(' '))
            + pynini.cross(".", "dot ")
            + NEMO_NOT_QUOTE
            + pynini.closure(pynutil.insert(' ') + NEMO_NOT_QUOTE)
        )

        server_default = pynini.closure(NEMO_NOT_QUOTE + pynutil.insert(' '))
        server_common = pynini.string_file(get_abs_path("data/electronic/server_name.tsv")) + pynutil.insert(' ')

        domain_common = pynini.cross(".", "dot ") + pynini.string_file(get_abs_path("data/electronic/domain.tsv"))

        domain = (
            pynutil.delete("domain:")
            + delete_space
            + pynutil.delete("\"")
            + (pynutil.add_weight(server_common, 1.09) | pynutil.add_weight(server_default, 1.1))
            + (pynutil.add_weight(domain_common, 1.09) | pynutil.add_weight(domain_default, 1.1))
            + delete_space
            + pynutil.delete("\"")
        )

        graph = user_name + delete_space + pynutil.insert("at ") + delete_space + domain + delete_space

        delete_tokens = self.delete_tokens(graph)
        self.fst = delete_tokens.optimize()
