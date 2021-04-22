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

from nemo_text_processing.text_normalization.graph_utils import NEMO_NOT_QUOTE, GraphFst, delete_space

try:
    import pynini
    from pynini.lib import pynutil

    PYNINI_AVAILABLE = True
except (ModuleNotFoundError, ImportError):
    PYNINI_AVAILABLE = False


class EmailFst(GraphFst):
    """
    Finite state transducer for verbalizing date
        e.g. tokens { email { username: "a.bc" servername: "g m a i l " domain: "com" } } -> a.bc et g m a i l dot com

    """

    def __init__(self):
        super().__init__(name="email", kind="verbalize")
        user_name = (
            pynutil.delete("username:")
            + delete_space
            + pynutil.delete("\"")
            + pynini.closure(NEMO_NOT_QUOTE, 1)
            + pynutil.delete("\"")
        )
        server_name = (
            pynutil.delete("servername:")
            + delete_space
            + pynutil.delete("\"")
            + pynini.closure(NEMO_NOT_QUOTE, 1)
            + delete_space
            + pynutil.delete("\"")
        )
        domain = (
            pynutil.delete("domain:")
            + delete_space
            + pynutil.delete("\"")
            + pynini.closure(NEMO_NOT_QUOTE, 1)
            + delete_space
            + pynutil.delete("\"")
        )

        graph = (
            user_name
            + delete_space
            + pynutil.insert(" et ")
            + server_name
            + delete_space
            + pynutil.insert("dot ")
            + domain
        )

        delete_tokens = self.delete_tokens(graph)
        self.fst = delete_tokens.optimize()
