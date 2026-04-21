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

"""Tests for hybrid circuits combining unitary segments with non-unitary operations."""

# pylint: disable=duplicate-code

from qiskit.circuit import QuantumCircuit, QuantumRegister, ClassicalRegister, IfElseOp
from qiskit.transpiler import PassManager
import qiskit.converters

from zxpass import ZXPass


def test_hybrid_circuit_optimization() -> None:
    """Test the issue #18 pattern: [A, measure, B, conditional, reset, C].

    Verifies that unitary segments A, B, and C are each optimised while
    measurement, conditional, and reset gates are preserved.
    """
    q = QuantumRegister(2, "q")
    c = ClassicalRegister(1, "c")
    qc = QuantumCircuit(q, c)

    # Segment A: some gates that can be optimised (H H = I).
    qc.h(q[0])
    qc.h(q[0])
    qc.cx(q[0], q[1])

    # Measurement.
    qc.measure(q[0], c[0])

    # Conditional correction.
    with qc.if_test((c, 1)):  # pylint: disable=not-context-manager
        qc.z(q[1])

    # Reset and segment C.
    qc.reset(q[0])
    qc.h(q[0])
    qc.h(q[0])

    zxpass = ZXPass()
    pass_manager = PassManager(zxpass)
    result = pass_manager.run(qc)

    dag = qiskit.converters.circuit_to_dag(result)
    op_names = [node.op.name for node in dag.topological_op_nodes()]

    # Non-unitary operations should be preserved.
    assert "measure" in op_names
    assert "reset" in op_names
    cond_ops = [node for node in dag.topological_op_nodes() if isinstance(node.op, IfElseOp)]
    assert len(cond_ops) == 1

    # The H-H pairs in segments A and C should have been optimised away.
    assert result.size() < qc.size(), (
        f"Expected optimised circuit to be smaller: got {result.size()} >= {qc.size()}"
    )


def test_measure_reset_same_qubit() -> None:
    """Test measure followed by reset on the same qubit."""
    q = QuantumRegister(1, "q")
    c = ClassicalRegister(1, "c")
    qc = QuantumCircuit(q, c)
    qc.h(q[0])
    qc.measure(q[0], c[0])
    qc.reset(q[0])
    qc.h(q[0])

    zxpass = ZXPass()
    pass_manager = PassManager(zxpass)
    result = pass_manager.run(qc)

    dag = qiskit.converters.circuit_to_dag(result)
    op_names = [node.op.name for node in dag.topological_op_nodes()]
    assert "measure" in op_names
    assert "reset" in op_names


def test_mid_circuit_measure_optimization() -> None:
    """Test that unitary segments around mid-circuit measurements are optimised.

    Each segment contains redundant CX pairs that ZX-calculus cancels, while
    the mid-circuit measurement is preserved between the two segments.
    """
    q = QuantumRegister(3, "q")
    c = ClassicalRegister(1, "c")
    qc = QuantumCircuit(q, c)

    # Segment A: CX pairs cancel, leaving one CX worth of entanglement.
    qc.cx(q[0], q[1])
    qc.cx(q[1], q[2])
    qc.cx(q[1], q[2])
    qc.cx(q[0], q[1])
    qc.cx(q[0], q[2])

    # Mid-circuit measurement (not terminal).
    qc.measure(q[0], c[0])

    # Segment B: same reducible pattern.
    qc.cx(q[0], q[2])
    qc.cx(q[0], q[1])
    qc.cx(q[1], q[2])
    qc.cx(q[1], q[2])
    qc.cx(q[0], q[1])

    zxpass = ZXPass()
    result = PassManager(zxpass).run(qc)
    dag = qiskit.converters.circuit_to_dag(result)
    op_names = [node.op.name for node in dag.topological_op_nodes()]

    assert op_names.count("measure") == 1
    # The measurement must appear before at least one post-measurement gate.
    measure_idx = op_names.index("measure")
    assert measure_idx < len(op_names) - 1, "measure should not be the last op"
    assert result.size() < qc.size(), (
        f"Expected optimisation: {result.size()} >= {qc.size()}"
    )


def test_repeated_measure_reset_pattern() -> None:
    """Test the measure-reset-compute pattern used in QEC.

    Multiple rounds of [compute, measure, reset] should each have
    their unitary segments optimised independently.
    """
    q = QuantumRegister(3, "q")
    c = ClassicalRegister(1, "c")
    qc = QuantumCircuit(q, c)

    for _ in range(3):
        # Compute: CX pairs cancel, leaving one CX worth of entanglement.
        qc.cx(q[0], q[1])
        qc.cx(q[1], q[2])
        qc.cx(q[1], q[2])
        qc.cx(q[0], q[1])
        qc.cx(q[0], q[2])
        # Measure and reset.
        qc.measure(q[0], c[0])
        qc.reset(q[0])

    zxpass = ZXPass()
    result = PassManager(zxpass).run(qc)
    dag = qiskit.converters.circuit_to_dag(result)
    op_names = [node.op.name for node in dag.topological_op_nodes()]

    assert op_names.count("measure") == 3
    assert op_names.count("reset") == 3
    assert result.size() < qc.size(), (
        f"Expected optimisation in repeated measure-reset pattern: "
        f"{result.size()} >= {qc.size()}"
    )


def test_benchmark_teleportation() -> None:
    """Benchmark: quantum teleportation circuit with mid-circuit measurement.

    Teleportation uses a Bell pair, mid-circuit measurements, and conditional
    corrections. Verifies the full hybrid pipeline handles this pattern.
    """
    q = QuantumRegister(3, "q")
    c = ClassicalRegister(2, "c")
    qc = QuantumCircuit(q, c)

    # Prepare state to teleport.
    qc.rx(0.7, q[0])

    # Create Bell pair between q[1] and q[2].
    qc.h(q[1])
    qc.cx(q[1], q[2])

    # Bell measurement on q[0], q[1].
    qc.cx(q[0], q[1])
    qc.h(q[0])
    qc.measure(q[0], c[0])
    qc.measure(q[1], c[1])

    # Conditional corrections on q[2].
    with qc.if_test((c, 1)):  # pylint: disable=not-context-manager
        qc.x(q[2])
    with qc.if_test((c, 2)):  # pylint: disable=not-context-manager
        qc.z(q[2])
    with qc.if_test((c, 3)):  # pylint: disable=not-context-manager
        qc.x(q[2])
        qc.z(q[2])

    zxpass = ZXPass()
    result = PassManager(zxpass).run(qc)

    dag = qiskit.converters.circuit_to_dag(result)
    op_names = [node.op.name for node in dag.topological_op_nodes()]
    assert op_names.count("measure") == 2
    cond_ops = [node for node in dag.topological_op_nodes() if isinstance(node.op, IfElseOp)]
    assert len(cond_ops) == 3


def test_benchmark_qec_syndrome_extraction() -> None:
    """Benchmark: QEC-like syndrome extraction with mid-circuit measurement.

    A simplified 3-qubit bit-flip code: data qubits are entangled with
    ancilla qubits, ancillae are measured mid-circuit, and corrections
    are applied conditionally.  Multiple rounds exercise the repeated
    measure-reset-compute pattern.
    """
    data = QuantumRegister(3, "data")
    ancilla = QuantumRegister(2, "anc")
    syn = ClassicalRegister(2, "syn")
    qc = QuantumCircuit(data, ancilla, syn)

    for _ in range(2):
        # Syndrome extraction: CNOT from data to ancilla.
        qc.cx(data[0], ancilla[0])
        qc.cx(data[1], ancilla[0])
        qc.cx(data[1], ancilla[1])
        qc.cx(data[2], ancilla[1])

        # Measure ancillae.
        qc.measure(ancilla[0], syn[0])
        qc.measure(ancilla[1], syn[1])

        # Conditional correction based on syndrome.
        with qc.if_test((syn, 1)):  # pylint: disable=not-context-manager
            qc.x(data[0])

        # Reset ancillae for next round.
        qc.reset(ancilla[0])
        qc.reset(ancilla[1])

    zxpass = ZXPass()
    result = PassManager(zxpass).run(qc)

    dag = qiskit.converters.circuit_to_dag(result)
    op_names = [node.op.name for node in dag.topological_op_nodes()]
    assert op_names.count("measure") == 4
    assert op_names.count("reset") == 4


def test_benchmark_repeated_conditional_corrections() -> None:
    """Benchmark: repeated measure-correct-compute cycles.

    Each cycle measures a qubit, applies a conditional correction, resets,
    and continues with unitary computation that has redundant gates for
    ZX-calculus to simplify.
    """
    q = QuantumRegister(3, "q")
    c = ClassicalRegister(1, "c")
    qc = QuantumCircuit(q, c)

    for _ in range(3):
        # Unitary segment with redundant CX pairs.
        qc.cx(q[0], q[1])
        qc.cx(q[1], q[2])
        qc.cx(q[1], q[2])
        qc.cx(q[0], q[1])
        qc.cx(q[0], q[2])

        # Mid-circuit measurement.
        qc.measure(q[0], c[0])

        # Conditional correction.
        with qc.if_test((c, 1)):  # pylint: disable=not-context-manager
            qc.z(q[1])

        # Reset and continue.
        qc.reset(q[0])

    zxpass = ZXPass()
    result = PassManager(zxpass).run(qc)

    dag = qiskit.converters.circuit_to_dag(result)
    op_names = [node.op.name for node in dag.topological_op_nodes()]
    assert op_names.count("measure") == 3
    assert op_names.count("reset") == 3
    cond_ops = [node for node in dag.topological_op_nodes() if isinstance(node.op, IfElseOp)]
    assert len(cond_ops) == 3
    # The redundant CX pairs in each unitary segment should be optimised.
    assert result.size() < qc.size(), (
        f"Expected optimisation in repeated cycles: {result.size()} >= {qc.size()}"
    )


def test_benchmark_multi_qubit_mid_circuit_measure() -> None:
    """Benchmark: multi-qubit circuit with measurements on different qubits.

    Tests that the hybrid splitter correctly handles measurements on
    different qubits within the same circuit, with unitary computation
    interleaved between measurements.
    """
    q = QuantumRegister(4, "q")
    c = ClassicalRegister(2, "c")
    qc = QuantumCircuit(q, c)

    # Entangle all qubits.
    qc.h(q[0])
    qc.cx(q[0], q[1])
    qc.cx(q[1], q[2])
    qc.cx(q[2], q[3])

    # Measure qubit 0.
    qc.measure(q[0], c[0])

    # More unitary computation on remaining qubits.
    qc.h(q[1])
    qc.cx(q[1], q[2])
    qc.cx(q[2], q[3])
    qc.h(q[3])

    # Measure qubit 1.
    qc.measure(q[1], c[1])

    # Conditional correction and final unitary.
    with qc.if_test((c, 1)):  # pylint: disable=not-context-manager
        qc.z(q[2])

    qc.h(q[2])
    qc.cx(q[2], q[3])

    zxpass = ZXPass()
    result = PassManager(zxpass).run(qc)

    dag = qiskit.converters.circuit_to_dag(result)
    op_names = [node.op.name for node in dag.topological_op_nodes()]
    assert op_names.count("measure") == 2
    cond_ops = [node for node in dag.topological_op_nodes() if isinstance(node.op, IfElseOp)]
    assert len(cond_ops) == 1
