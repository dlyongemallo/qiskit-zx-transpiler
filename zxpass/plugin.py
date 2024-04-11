# ZX transpiler pass for Qiskit
# Copyright (C) 2023 David Yonge-Mallo
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#         http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""A transpiler stage plugin for Qiskit which uses ZX-Calculus for circuit optimization, implemented using PyZX."""

from typing import Optional

from qiskit.transpiler.preset_passmanagers.plugin import PassManagerStagePlugin
from qiskit.transpiler import PassManagerConfig, PassManager

from .zxpass import ZXPass


class ZXPlugin(PassManagerStagePlugin):  # pylint: disable=too-few-public-methods
    """Plugin class for optimization stage with :class:`~.ZXPass`."""

    def pass_manager(
        self, pass_manager_config: PassManagerConfig, optimization_level: Optional[int] = None
    ) -> PassManager:
        return PassManager([ZXPass()])
