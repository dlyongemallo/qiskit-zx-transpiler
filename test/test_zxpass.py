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

from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector
from qiskit.transpiler import PassManager
from zxpass import ZXPass
import numpy as np


def _run_zxpass(qc: QuantumCircuit) -> bool:
    zxpass = ZXPass()
    pass_manager = PassManager(zxpass)
    zx_qc = pass_manager.run(qc)

    return Statevector.from_instruction(qc).equiv(Statevector.from_instruction(zx_qc))


def test_basic_circuit() -> None:
    """Test a basic circuit.

    Taken from https://github.com/Quantomatic/pyzx/blob/master/circuits/Fast/mod5_4_before
    """
    qc = QuantumCircuit(5)
    qc.x(4)
    qc.h(4)
    qc.ccz(0, 3, 4)
    qc.ccz(2, 3, 4)
    qc.h(4)
    qc.cx(3, 4)
    qc.h(4)
    qc.ccz(1, 2, 4)
    qc.h(4)
    qc.cx(2, 4)
    qc.h(4)
    qc.ccz(0, 1, 4)
    qc.h(4)
    qc.cx(1, 4)
    qc.cx(0, 4)

    assert _run_zxpass(qc)


def test_pyzx_issue_102() -> None:
    """Regression test for pyzx issue #102.
    """
    qc = QuantumCircuit(4)
    qc.ccx(2, 1, 0)
    qc.ccz(0, 1, 2)
    qc.h(1)
    qc.ccx(1, 2, 3)
    qc.t(1)
    qc.ccz(0, 1, 2)
    qc.h(1)
    qc.t(0)
    qc.ccz(2, 1, 0)
    qc.s(1)
    qc.ccx(2, 1, 0)
    qc.crz(0.2*np.pi, 0, 1)
    qc.rz(0.8*np.pi, 1)
    qc.cry(0.4*np.pi, 2, 1)
    qc.crx(0.02*np.pi, 2, 0)

    assert _run_zxpass(qc)
