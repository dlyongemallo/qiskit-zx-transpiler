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

"""Shared helpers for the ZX transpiler test modules."""

from typing import Callable, Optional

import pyzx as zx
from qiskit.circuit import QuantumCircuit
from qiskit.quantum_info import Statevector
from qiskit.transpiler import PassManager

from zxpass import ZXPass


def run_zxpass(
    qc: QuantumCircuit, optimize: Optional[Callable[[zx.Circuit], zx.Circuit]] = None
) -> bool:
    """Run ZXPass on ``qc`` and return whether the result is equivalent to the original."""
    zxpass = ZXPass(optimize)
    pass_manager = PassManager(zxpass)
    zx_qc = pass_manager.run(qc)

    return Statevector.from_instruction(qc).equiv(Statevector.from_instruction(zx_qc))


def assert_equiv(original: QuantumCircuit, optimized: QuantumCircuit) -> None:
    """Assert that two circuits have the same statevector."""
    assert Statevector.from_instruction(original).equiv(
        Statevector.from_instruction(optimized)
    )
