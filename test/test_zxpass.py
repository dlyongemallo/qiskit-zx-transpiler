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

import pytest
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.quantum_info import Statevector
from qiskit.transpiler import PassManager
import qiskit.converters
from typing import Callable
from zxpass import ZXPass
import pyzx as zx
import numpy as np


def _run_zxpass(qc: QuantumCircuit, optimize: Callable[[zx.Circuit], zx.Circuit] = None) -> bool:
    zxpass = ZXPass(optimize)
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


def test_custom_optimize() -> None:
    """Test custom optimize method.
    """
    qc = QuantumCircuit(4)
    qc.h(0)
    qc.h(1)
    qc.h(2)
    qc.h(3)
    qc.cx(0, 1)
    qc.cx(1, 2)
    qc.cx(2, 3)
    qc.cx(3, 0)

    def optimize(circ: zx.Circuit) -> zx.Circuit:
        # Any function that takes a zx.Circuit and returns a zx.Circuit will do.
        return circ.to_basic_gates()

    assert _run_zxpass(qc, optimize)


def test_measurement() -> None:
    """Test a circuit with a measurement.
    """
    q = QuantumRegister(1, 'q')
    c = ClassicalRegister(1, 'c')
    qc = QuantumCircuit(q, c)
    qc.h(q[0])
    qc.measure(q[0], c[0])
    qc.h(q[0])

    dag = qiskit.converters.circuit_to_dag(qc)
    zxpass = ZXPass()
    circuits_and_nodes = zxpass._dag_to_circuits_and_nodes(dag)
    assert len(circuits_and_nodes) == 3
    assert circuits_and_nodes[1] == dag.op_nodes()[1]


def test_conditional_gate() -> None:
    """Test a circuit with a conditional gate.
    """
    q = QuantumRegister(1, 'q')
    c = ClassicalRegister(1, 'c')
    qc = QuantumCircuit(q, c)
    qc.h(q[0]).c_if(c, 0)

    assert _run_zxpass(qc)


def test_measurement() -> None:
    """Test a circuit with measurement.

    From: https://qiskit.org/documentation/tutorials/circuits_advanced/04_transpiler_passes_and_passmanager.html
    """
    q = QuantumRegister(3, 'q')
    c = ClassicalRegister(3, 'c')
    qc = QuantumCircuit(q, c)
    qc.h(q[0])
    qc.cx(q[0], q[1])
    qc.measure(q[0], c[0])
    # qc.rz(0.5, q[1]).c_if(c, 2)

    assert _run_zxpass(qc)


def test_pyzx_issue_102() -> None:
    """Regression test for PyZX issue #102.

    This tests for a bug which prevented an earlier attempt at a Qiskit ZX transpiler pass from working.
    See: https://github.com/Quantomatic/pyzx/issues/102
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
