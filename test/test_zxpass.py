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

"""End-to-end tests for the ZX transpiler pass.

This module covers the core pass API (custom ``optimize`` callback, previously
known regressions, random-circuit equivalence).  Focused suites for conversion,
hybrid segmentation, permutation utilities, and ``_optimize_unitary`` live in
sibling ``test_*.py`` files.
"""

import pyzx as zx
import numpy as np

import qiskit.converters
from qiskit.circuit import QuantumCircuit
from qiskit.circuit.random import random_circuit

from zxpass import ZXPass

from ._helpers import run_zxpass


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

    assert run_zxpass(qc)


def test_custom_optimize() -> None:
    """Test custom optimize method."""
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

    assert run_zxpass(qc, optimize)


def test_unitary() -> None:
    """Test a circuit with a unitary gate."""
    matrix = [[0, 0, 0, 1], [0, 0, 1, 0], [1, 0, 0, 0], [0, 1, 0, 0]]
    qc = QuantumCircuit(2)
    qc.unitary(matrix, [0, 1])

    assert run_zxpass(qc)


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
    qc.crz(0.2 * np.pi, 0, 1)
    qc.rz(0.8 * np.pi, 1)
    qc.cry(0.4 * np.pi, 2, 1)
    qc.crx(0.02 * np.pi, 2, 0)

    assert run_zxpass(qc)


def test_random_circuits() -> None:
    """Test random circuits."""
    for _ in range(20):
        num_qubits = np.random.randint(4, 9)
        depth = np.random.randint(10, 21)
        qc = random_circuit(num_qubits, depth)
        assert run_zxpass(qc)


def test_no_regression() -> None:
    """``ZXPass.run`` should not return a DAG with a larger operation count."""
    for _ in range(20):
        num_qubits = np.random.randint(2, 7)
        depth = np.random.randint(5, 21)
        qc = random_circuit(num_qubits, depth)
        dag = qiskit.converters.circuit_to_dag(qc)
        original_count = dag.size(recurse=True)

        zxpass = ZXPass()
        optimized_dag = zxpass.run(dag)
        optimized_count = optimized_dag.size(recurse=True)

        assert optimized_count <= original_count, (
            f"ZXPass increased operation count from {original_count} to {optimized_count}"
        )


def test_custom_optimize_result_preserved_when_inflating() -> None:
    """A custom ``optimize`` callback's result must not be discarded when it inflates the circuit.

    Custom optimisers may target metrics other than total operation count
    (e.g. depth, T-count), so the default-optimiser regression guard in
    ``ZXPass.run`` should not apply to them.
    """
    qc = QuantumCircuit(2)
    qc.h(0)
    qc.cx(0, 1)

    dag = qiskit.converters.circuit_to_dag(qc)
    original_count = dag.size(recurse=True)

    def optimize(circ: zx.Circuit) -> zx.Circuit:
        inflated = zx.Circuit(circ.qubits, bit_amount=circ.bits or None)
        for gate in circ.gates:
            inflated.add_gate(gate)
        # Append a cancelling Hadamard pair to inflate the circuit.
        inflated.add_gate("HAD", 0)
        inflated.add_gate("HAD", 0)
        return inflated

    zxpass = ZXPass(optimize=optimize)
    optimized_dag = zxpass.run(dag)
    optimized_count = optimized_dag.size(recurse=True)

    assert optimized_count > original_count, (
        f"Custom optimize result should not be discarded by the gate-count guard; "
        f"expected size > {original_count}, got {optimized_count}"
    )
