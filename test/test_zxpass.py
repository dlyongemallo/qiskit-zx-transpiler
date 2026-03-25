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

"""Tests for Qiskit ZX transpiler."""

from typing import Callable, Optional
import pyzx as zx
from pyzx.circuit.gates import Measurement as PyzxMeasurement, Reset as PyzxReset
from pyzx.circuit.gates import ConditionalGate
import numpy as np

from qiskit.circuit import QuantumCircuit, QuantumRegister, ClassicalRegister, IfElseOp
from qiskit.circuit import Measure, Reset
from qiskit.dagcircuit import DAGOpNode
from qiskit.quantum_info import Statevector
from qiskit.transpiler import PassManager
from qiskit.circuit.random import random_circuit
import qiskit.converters

from zxpass import ZXPass


def _run_zxpass(
    qc: QuantumCircuit, optimize: Optional[Callable[[zx.Circuit], zx.Circuit]] = None
) -> bool:
    zxpass = ZXPass(optimize)
    pass_manager = PassManager(zxpass)
    zx_qc = pass_manager.run(qc)

    return Statevector.from_instruction(qc).equiv(Statevector.from_instruction(zx_qc))


def _assert_equiv(original: QuantumCircuit, optimized: QuantumCircuit) -> None:
    """Assert that two circuits have the same statevector."""
    assert Statevector.from_instruction(original).equiv(
        Statevector.from_instruction(optimized)
    )


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

    assert _run_zxpass(qc, optimize)


def test_measurement() -> None:
    """Test that measurements are included in the PyZX circuit and round-trip correctly."""
    q = QuantumRegister(1, "q")
    c = ClassicalRegister(1, "c")
    qc = QuantumCircuit(q, c)
    qc.h(q[0])
    qc.measure(q[0], c[0])
    qc.h(q[0])

    dag = qiskit.converters.circuit_to_dag(qc)
    zxpass = ZXPass()
    circuits_and_nodes = zxpass._dag_to_circuits_and_nodes(  # pylint: disable=protected-access
        dag
    )
    assert len(circuits_and_nodes) == 1
    assert isinstance(circuits_and_nodes[0], zx.Circuit)
    gates = circuits_and_nodes[0].gates
    assert len(gates) == 3
    assert isinstance(gates[1], PyzxMeasurement)
    assert gates[1].target == 0
    assert gates[1].result_bit == 0

    # Round-trip: convert back to DAG and verify measure is preserved.
    identity = ZXPass(optimize=lambda circ: circ)
    result_dag = identity.run(dag)
    op_names = [node.op.name for node in result_dag.topological_op_nodes()]
    assert op_names == ["h", "measure", "h"]
    measure_node = [n for n in result_dag.topological_op_nodes() if n.op.name == "measure"][0]
    assert isinstance(measure_node.op, Measure)
    assert measure_node.qargs == (q[0],)
    assert measure_node.cargs == (c[0],)


def test_reset() -> None:
    """Test that resets are included in the PyZX circuit and round-trip correctly."""
    q = QuantumRegister(1, "q")
    qc = QuantumCircuit(q)
    qc.h(q[0])
    qc.reset(q[0])
    qc.h(q[0])

    dag = qiskit.converters.circuit_to_dag(qc)
    zxpass = ZXPass()
    circuits_and_nodes = zxpass._dag_to_circuits_and_nodes(  # pylint: disable=protected-access
        dag
    )
    assert len(circuits_and_nodes) == 1
    assert isinstance(circuits_and_nodes[0], zx.Circuit)
    gates = circuits_and_nodes[0].gates
    assert len(gates) == 3
    assert isinstance(gates[1], PyzxReset)
    assert gates[1].target == 0

    # Round-trip: convert back to DAG and verify reset is preserved.
    identity = ZXPass(optimize=lambda circ: circ)
    result_dag = identity.run(dag)
    op_names = [node.op.name for node in result_dag.topological_op_nodes()]
    assert op_names == ["h", "reset", "h"]
    reset_node = [n for n in result_dag.topological_op_nodes() if n.op.name == "reset"][0]
    assert isinstance(reset_node.op, Reset)
    assert reset_node.qargs == (q[0],)


def test_measurement_round_trip() -> None:
    """Test that a circuit with measurement survives the full pass."""
    q = QuantumRegister(2, "q")
    c = ClassicalRegister(2, "c")
    qc = QuantumCircuit(q, c)
    qc.h(q[0])
    qc.cx(q[0], q[1])
    qc.measure(q[0], c[0])
    qc.measure(q[1], c[1])

    zxpass = ZXPass()
    pass_manager = PassManager(zxpass)
    result = pass_manager.run(qc)

    op_names = [
        node.op.name
        for node in qiskit.converters.circuit_to_dag(result).topological_op_nodes()
    ]
    assert op_names.count("measure") == 2


def test_reset_round_trip() -> None:
    """Test that a circuit with reset survives the full pass."""
    q = QuantumRegister(1, "q")
    qc = QuantumCircuit(q)
    qc.h(q[0])
    qc.reset(q[0])
    qc.x(q[0])

    zxpass = ZXPass()
    pass_manager = PassManager(zxpass)
    result = pass_manager.run(qc)

    op_names = [
        node.op.name
        for node in qiskit.converters.circuit_to_dag(result).topological_op_nodes()
    ]
    assert "reset" in op_names


def test_conditional_round_trip() -> None:
    """Test that a circuit with measurement and conditional gate survives the pass."""
    q = QuantumRegister(1, "q")
    c = ClassicalRegister(1, "c")
    qc = QuantumCircuit(q, c)
    qc.h(q[0])
    qc.measure(q[0], c[0])
    with qc.if_test((c, 1)):  # pylint: disable=not-context-manager
        qc.z(q[0])
    qc.h(q[0])

    zxpass = ZXPass()
    pass_manager = PassManager(zxpass)
    result = pass_manager.run(qc)

    op_names = [
        node.op.name
        for node in qiskit.converters.circuit_to_dag(result).topological_op_nodes()
    ]
    assert "measure" in op_names
    # The conditional Z should be present as an IfElseOp.
    cond_ops = [
        node
        for node in qiskit.converters.circuit_to_dag(result).topological_op_nodes()
        if isinstance(node.op, IfElseOp)
    ]
    assert len(cond_ops) == 1


def test_conditional_z_gate() -> None:
    """Test that a conditional Z gate is converted to a PyZX ConditionalGate and round-trips."""
    q = QuantumRegister(1, "q")
    c = ClassicalRegister(1, "c")
    qc = QuantumCircuit(q, c)
    qc.measure(q[0], c[0])
    with qc.if_test((c, 1)):  # pylint: disable=not-context-manager
        qc.z(q[0])

    dag = qiskit.converters.circuit_to_dag(qc)
    zxpass = ZXPass()
    circuits_and_nodes = zxpass._dag_to_circuits_and_nodes(  # pylint: disable=protected-access
        dag
    )
    # Both measure and conditional Z should be in the same PyZX circuit.
    assert len(circuits_and_nodes) == 1
    assert isinstance(circuits_and_nodes[0], zx.Circuit)
    cond_gates = [g for g in circuits_and_nodes[0].gates if isinstance(g, ConditionalGate)]
    assert len(cond_gates) == 1
    assert cond_gates[0].condition_register == "c"
    assert cond_gates[0].condition_value == 1

    # Round-trip: convert back to DAG and verify the conditional gate is preserved.
    identity = ZXPass(optimize=lambda circ: circ)
    result_dag = identity.run(dag)
    cond_ops = [n for n in result_dag.topological_op_nodes() if isinstance(n.op, IfElseOp)]
    assert len(cond_ops) == 1
    assert cond_ops[0].op.condition == (c, 1)
    assert cond_ops[0].op.blocks[0].data[0].operation.name == "z"


def test_conditional_x_gate() -> None:
    """Test that a conditional X gate is converted to a PyZX ConditionalGate and round-trips."""
    q = QuantumRegister(1, "q")
    c = ClassicalRegister(1, "c")
    qc = QuantumCircuit(q, c)
    qc.measure(q[0], c[0])
    with qc.if_test((c, 1)):  # pylint: disable=not-context-manager
        qc.x(q[0])

    dag = qiskit.converters.circuit_to_dag(qc)
    zxpass = ZXPass()
    circuits_and_nodes = zxpass._dag_to_circuits_and_nodes(  # pylint: disable=protected-access
        dag
    )
    assert len(circuits_and_nodes) == 1
    assert isinstance(circuits_and_nodes[0], zx.Circuit)
    cond_gates = [g for g in circuits_and_nodes[0].gates if isinstance(g, ConditionalGate)]
    assert len(cond_gates) == 1

    # Round-trip.
    identity = ZXPass(optimize=lambda circ: circ)
    result_dag = identity.run(dag)
    cond_ops = [n for n in result_dag.topological_op_nodes() if isinstance(n.op, IfElseOp)]
    assert len(cond_ops) == 1
    assert cond_ops[0].op.condition == (c, 1)
    assert cond_ops[0].op.blocks[0].data[0].operation.name == "x"


def test_conditional_s_gate() -> None:
    """Test that a conditional S gate is converted to a PyZX ConditionalGate and round-trips."""
    q = QuantumRegister(1, "q")
    c = ClassicalRegister(2, "c")
    qc = QuantumCircuit(q, c)
    qc.measure(q[0], c[0])
    with qc.if_test((c, 2)):  # pylint: disable=not-context-manager
        qc.s(q[0])

    dag = qiskit.converters.circuit_to_dag(qc)
    zxpass = ZXPass()
    circuits_and_nodes = zxpass._dag_to_circuits_and_nodes(  # pylint: disable=protected-access
        dag
    )
    assert len(circuits_and_nodes) == 1
    cond_gates = [g for g in circuits_and_nodes[0].gates if isinstance(g, ConditionalGate)]
    assert len(cond_gates) == 1
    assert cond_gates[0].condition_register == "c"
    assert cond_gates[0].condition_value == 2
    assert cond_gates[0].register_size == 2

    # Round-trip.
    identity = ZXPass(optimize=lambda circ: circ)
    result_dag = identity.run(dag)
    cond_ops = [n for n in result_dag.topological_op_nodes() if isinstance(n.op, IfElseOp)]
    assert len(cond_ops) == 1
    assert cond_ops[0].op.condition == (c, 2)
    assert cond_ops[0].op.blocks[0].data[0].operation.name == "s"


def test_conditional_unsupported_gate() -> None:
    """Test that an unsupported conditional gate (e.g. H) is stored as a DAGOpNode."""
    q = QuantumRegister(1, "q")
    c = ClassicalRegister(1, "c")
    qc = QuantumCircuit(q, c)
    with qc.if_test((c, 0)):  # pylint: disable=not-context-manager
        qc.h(q[0])

    dag = qiskit.converters.circuit_to_dag(qc)
    zxpass = ZXPass()
    circuits_and_nodes = zxpass._dag_to_circuits_and_nodes(  # pylint: disable=protected-access
        dag
    )
    # Conditional H is not supported by PyZX ConditionalGate.
    assert len(circuits_and_nodes) == 1
    assert isinstance(circuits_and_nodes[0], DAGOpNode)


def test_conditional_gate_passthrough() -> None:
    """Test a circuit with an unsupported conditional gate passes through."""
    q = QuantumRegister(1, "q")
    c = ClassicalRegister(1, "c")
    qc = QuantumCircuit(q, c)
    with qc.if_test((c, 0)):  # pylint: disable=not-context-manager
        qc.h(q[0])

    zxpass = ZXPass()
    pass_manager = PassManager(zxpass)
    result = pass_manager.run(qc)

    dag = qiskit.converters.circuit_to_dag(result)
    ops = list(dag.topological_op_nodes())
    assert len(ops) == 1
    assert isinstance(ops[0].op, IfElseOp)


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


def test_unitary() -> None:
    """Test a circuit with a unitary gate."""
    matrix = [[0, 0, 0, 1], [0, 0, 1, 0], [1, 0, 0, 0], [0, 1, 0, 0]]
    qc = QuantumCircuit(2)
    qc.unitary(matrix, [0, 1])

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
    qc.crz(0.2 * np.pi, 0, 1)
    qc.rz(0.8 * np.pi, 1)
    qc.cry(0.4 * np.pi, 2, 1)
    qc.crx(0.02 * np.pi, 2, 0)

    assert _run_zxpass(qc)


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


def test_post_extraction_cleanup() -> None:
    """Test that ``_optimize_unitary`` applies ``basic_optimization`` after extraction.

    Extraction produces CZ + HAD form with redundant single-qubit gates;
    ``basic_optimization`` cancels those and folds HAD-CZ-HAD into CNOT.
    Compares against naive extraction to verify the post-pass actually ran:
    if the ``basic_optimization`` step were removed, ``_optimize_unitary``
    would return a circuit no smaller than ``extract_circuit`` alone and
    this assertion would fail.
    """
    # pylint: disable=import-outside-toplevel
    from fractions import Fraction
    from zxpass.zxpass import _optimize_unitary

    # Build a circuit whose extraction leaves enough single-qubit redundancy
    # for basic_optimization to produce a measurable reduction, while being
    # small enough that the gate-count fallback does not trigger.
    c = zx.Circuit(4)
    for j in range(4):
        c.add_gate("HAD", j)
    for _ in range(3):
        for i in range(3):
            c.add_gate("CNOT", i, i + 1)
        for j in range(4):
            c.add_gate("ZPhase", j, phase=Fraction(1, 2 + j))
        for i in range(3):
            c.add_gate("CNOT", i, i + 1)

    optimized = _optimize_unitary(c)

    # Size of the extraction output without the post-pass, i.e., what
    # ``_optimize_unitary`` would have produced if the ``basic_optimization``
    # call were removed.
    g = c.to_graph()
    zx.simplify.full_reduce(g)
    extraction_only = zx.extract.extract_circuit(g)

    assert len(optimized.gates) < len(extraction_only.gates), (
        f"Expected _optimize_unitary to yield fewer gates than extract_circuit alone; "
        f"got {len(optimized.gates)} vs {len(extraction_only.gates)}"
    )


def test_post_extraction_cleanup_equivalence() -> None:
    """Test that the post-extraction cleanup preserves circuit equivalence.

    Runs multiple circuits through the full ZXPass pipeline and verifies
    statevector equivalence after the basic_optimization post-pass.
    """
    circuits = []

    # Bell state preparation with extra gates.
    qc1 = QuantumCircuit(3)
    qc1.h(0)
    qc1.cx(0, 1)
    qc1.cx(1, 2)
    qc1.h(2)
    qc1.cx(2, 0)
    circuits.append(qc1)

    # Multi-CX circuit.
    qc2 = QuantumCircuit(4)
    for i in range(3):
        qc2.cx(i, i + 1)
    qc2.h(0)
    qc2.cx(3, 0)
    qc2.h(1)
    circuits.append(qc2)

    for qc in circuits:
        assert _run_zxpass(qc), f"Equivalence failed for circuit:\n{qc}"


def test_gate_count_fallback_toffoli() -> None:
    """Test that the gate-count fallback prevents regression on a Toffoli circuit.

    A Toffoli decomposed into a basic gate basis expands to many basic gates;
    running full_reduce + extract_circuit produces even more. The fallback
    should return a circuit no larger than the decomposed original.
    """
    qc = QuantumCircuit(3)
    qc.ccx(0, 1, 2)
    qc = qc.decompose()

    zxpass = ZXPass()
    result = PassManager(zxpass).run(qc)

    assert result.size() <= qc.size(), (
        f"Gate-count fallback failed: {result.size()} > {qc.size()}"
    )
    _assert_equiv(qc, result)


def test_gate_count_fallback_fredkin() -> None:
    """Test that the gate-count fallback prevents regression on a Fredkin circuit."""
    qc = QuantumCircuit(3)
    qc.cswap(0, 1, 2)
    qc = qc.decompose()

    zxpass = ZXPass()
    result = PassManager(zxpass).run(qc)

    assert result.size() <= qc.size(), (
        f"Gate-count fallback failed: {result.size()} > {qc.size()}"
    )
    _assert_equiv(qc, result)


def test_gate_count_fallback_toffoli_compact() -> None:
    """Test the fallback preserves a compact Toffoli (single CCX instruction).

    A lone ``ccx`` is a single PyZX ``Tofolli`` gate; ``extract_circuit``
    would expand it into many basic gates, so the fallback should keep the
    original compact form.
    """
    qc = QuantumCircuit(3)
    qc.ccx(0, 1, 2)

    zxpass = ZXPass()
    result = PassManager(zxpass).run(qc)

    assert result.size() <= qc.size(), (
        f"Gate-count fallback failed: {result.size()} > {qc.size()}"
    )
    _assert_equiv(qc, result)


def test_gate_count_fallback_fredkin_compact() -> None:
    """Test the fallback preserves a compact Fredkin (single CSWAP instruction)."""
    qc = QuantumCircuit(3)
    qc.cswap(0, 1, 2)

    zxpass = ZXPass()
    result = PassManager(zxpass).run(qc)

    assert result.size() <= qc.size(), (
        f"Gate-count fallback failed: {result.size()} > {qc.size()}"
    )
    _assert_equiv(qc, result)


def test_gate_count_fallback_still_optimizes() -> None:
    """Test that the fallback does not prevent genuine optimisation.

    Redundant CX pairs cancel under ZX-calculus; the extracted circuit
    should be strictly smaller than the original.
    """
    qc = QuantumCircuit(3)
    qc.cx(0, 1)
    qc.cx(0, 1)
    qc.cx(1, 2)
    qc.cx(1, 2)
    qc.h(2)

    zxpass = ZXPass()
    result = PassManager(zxpass).run(qc)

    assert result.size() < qc.size(), (
        f"Expected optimisation: {result.size()} >= {qc.size()}"
    )
    _assert_equiv(qc, result)


def test_gate_count_fallback_hybrid_segments() -> None:
    """Test the gate-count fallback applies per-segment in hybrid circuits.

    Segment A (a decomposed Toffoli) should trigger the fallback while the
    measurement and subsequent gates are preserved.
    """
    q = QuantumRegister(3, "q")
    c = ClassicalRegister(1, "c")
    qc = QuantumCircuit(q, c)
    qc.ccx(q[0], q[1], q[2])
    qc.measure(q[0], c[0])
    qc.h(q[1])
    qc = qc.decompose()

    zxpass = ZXPass()
    result = PassManager(zxpass).run(qc)

    dag = qiskit.converters.circuit_to_dag(result)
    op_names = [node.op.name for node in dag.topological_op_nodes()]
    assert "measure" in op_names
    assert result.size() <= qc.size(), (
        f"Gate-count fallback failed in hybrid circuit: "
        f"{result.size()} > {qc.size()}"
    )


def test_random_circuits() -> None:
    """Test random circuits."""
    for _ in range(20):
        num_qubits = np.random.randint(4, 9)
        depth = np.random.randint(10, 21)
        qc = random_circuit(num_qubits, depth)
        assert _run_zxpass(qc)


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
