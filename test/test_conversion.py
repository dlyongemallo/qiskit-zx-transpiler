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

"""Tests for measurement, reset, and conditional-gate conversion and round-trips."""

# pylint: disable=duplicate-code

import pyzx as zx
from pyzx.circuit.gates import Measurement as PyzxMeasurement, Reset as PyzxReset
from pyzx.circuit.gates import ConditionalGate

from qiskit.circuit import QuantumCircuit, QuantumRegister, ClassicalRegister, IfElseOp
from qiskit.circuit import Measure, Reset
from qiskit.dagcircuit import DAGOpNode
from qiskit.transpiler import PassManager
import qiskit.converters

from zxpass import ZXPass


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
