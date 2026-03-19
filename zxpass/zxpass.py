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

"""A transpiler pass for Qiskit which uses ZX-Calculus for circuit optimization, implemented using PyZX."""

from typing import Dict, List, Tuple, Callable, Optional, Type, Union
from fractions import Fraction
import numpy as np

from qiskit.transpiler.basepasses import TransformationPass
from qiskit.dagcircuit import DAGCircuit, DAGOpNode
from qiskit.circuit import Qubit, Instruction, Measure, Reset, ClassicalRegister

from qiskit.circuit.library import XGate, YGate, ZGate, HGate, SGate, TGate, SXGate
from qiskit.circuit.library import SdgGate, TdgGate, SXdgGate
from qiskit.circuit.library import (
    RXGate,
    RYGate,
    RZGate,
    PhaseGate,
    U1Gate,
    U2Gate,
    U3Gate,
)
from qiskit.circuit.library import SwapGate, CXGate, CYGate, CZGate, CHGate, CSXGate
from qiskit.circuit.library import (
    CRXGate,
    CRYGate,
    CRZGate,
    CPhaseGate,
    CU1Gate,
    RXXGate,
    RZZGate,
    CU3Gate,
    CUGate,
)
from qiskit.circuit.library import CSwapGate, CCXGate, CCZGate

import pyzx as zx
from pyzx.circuit.gates import Gate
from pyzx.circuit.gates import NOT, Y, Z, S, T, HAD, SX
from pyzx.circuit.gates import XPhase, YPhase, ZPhase, U2, U3
from pyzx.circuit.gates import SWAP, CNOT, CY, CZ, CHAD, CSX
from pyzx.circuit.gates import CRX, CRY, CRZ, CPhase, RXX, RZZ, CU3, CU
from pyzx.circuit.gates import CSWAP, Tofolli, CCZ
from pyzx.circuit.gates import Measurement as PyzxMeasurement, Reset as PyzxReset
from pyzx.circuit.gates import ConditionalGate

qiskit_gate_table: Dict[str, Tuple[Type[Gate], Type[Instruction], int, int]] = {
    # OpenQASM gate name: (PyZX gate type, Qiskit gate type, number of qubits, number of parameters, adjoint)
    "x": (NOT, XGate, 1, 0),
    "y": (Y, YGate, 1, 0),
    "z": (Z, ZGate, 1, 0),
    "h": (HAD, HGate, 1, 0),
    "s": (S, SGate, 1, 0),
    "t": (T, TGate, 1, 0),
    "sx": (SX, SXGate, 1, 0),
    "sdg": (S, SdgGate, 1, 0, True),  # type: ignore
    "tdg": (T, TdgGate, 1, 0, True),  # type: ignore
    "sxdg": (SX, SXdgGate, 1, 0, True),  # type: ignore
    "rx": (XPhase, RXGate, 1, 1),
    "ry": (YPhase, RYGate, 1, 1),
    "rz": (ZPhase, RZGate, 1, 1),
    "p": (ZPhase, PhaseGate, 1, 1),
    "u1": (ZPhase, U1Gate, 1, 1),
    "u2": (U2, U2Gate, 1, 2),
    "u3": (U3, U3Gate, 1, 3),
    "swap": (SWAP, SwapGate, 2, 0),
    "cx": (CNOT, CXGate, 2, 0),
    "cy": (CY, CYGate, 2, 0),
    "cz": (CZ, CZGate, 2, 0),
    "ch": (CHAD, CHGate, 2, 0),
    "csx": (CSX, CSXGate, 2, 0),
    "crx": (CRX, CRXGate, 2, 1),
    "cry": (CRY, CRYGate, 2, 1),
    "crz": (CRZ, CRZGate, 2, 1),
    "cp": (CPhase, CPhaseGate, 2, 1),
    "cphase": (CPhase, CPhaseGate, 2, 1),
    "cu1": (CPhase, CU1Gate, 2, 1),
    "rxx": (RXX, RXXGate, 2, 1),
    "rzz": (RZZ, RZZGate, 2, 1),
    "cu3": (CU3, CU3Gate, 2, 3),
    "cu": (CU, CUGate, 2, 4),
    "cswap": (CSWAP, CSwapGate, 3, 0),
    "ccx": (Tofolli, CCXGate, 3, 0),
    "ccz": (CCZ, CCZGate, 3, 0),
}

# Gates that can appear as inner gates in a ConditionalGate. These must be
# single-qubit Z or X rotations (subclasses of ZPhase or XPhase in PyZX).
_conditional_inner_gates = {
    "x", "z", "s", "t", "sx",
    "sdg", "tdg", "sxdg",
    "rx", "rz", "p", "u1",
}


def _optimize(c: zx.Circuit) -> zx.Circuit:
    g = c.to_graph()
    zx.simplify.full_reduce(g)
    return zx.extract.extract_circuit(g)


class ZXPass(TransformationPass):
    """This is a ZX transpiler pass using PyZX for circuit optimization.

    :param optimize: The function to use for optimizing a PyZX Circuit. If not specified, uses
        :py:meth:`~pyzx.simplify.full_reduce` by default.
    :type optimize: Callable[[pyzx.Circuit], pyzx.Circuit], optional
    """

    def __init__(self, optimize: Optional[Callable[[zx.Circuit], zx.Circuit]] = None):
        super().__init__()
        self.optimize: Callable[[zx.Circuit], zx.Circuit] = optimize or _optimize

    @staticmethod
    def _try_convert_conditional(
        node: DAGOpNode,
        qubit_to_index: Dict[Qubit, int],
    ) -> Optional[ConditionalGate]:
        """Try to convert a conditional DAGOpNode to a PyZX ConditionalGate.

        Returns ``None`` if the gate is not a supported conditional conversion
        (e.g. unsupported gate type, Clbit condition, or multi-qubit gate).
        """
        gate = node.op
        cond_reg, cond_val = gate.condition
        if not isinstance(cond_reg, ClassicalRegister):
            return None
        if gate.name not in _conditional_inner_gates or gate.name not in qiskit_gate_table:
            return None
        if len(node.qargs) != 1:
            return None
        gate_type, _, _, num_params, *adjoint = qiskit_gate_table[gate.name]  # type: ignore
        if len(node.op.params) != num_params:
            raise ValueError(
                f"Expected {num_params} parameters for conditional gate "
                f"{gate.name}, got {len(node.op.params)}: {node.op.params}."
            )
        inner_gate = gate_type(  # type: ignore[call-arg]
            qubit_to_index[node.qargs[0]],
            *[Fraction(param / np.pi) for param in node.op.params],
            **{"adjoint": adjoint[0]} if adjoint else {},
        )
        return ConditionalGate(
            cond_reg.name, int(cond_val), inner_gate, cond_reg.size
        )

    def _dag_to_circuits_and_nodes(
        self, dag: DAGCircuit
    ) -> List[Union[zx.Circuit, DAGOpNode]]:
        """Convert a DAG to a list of PyZX Circuits and DAGOpNodes. As much of the DAG is converted to PyZX Circuits as
        possible, but some gates are not supported by PyZX and are left as DAGOpNodes.

        :param dag: The DAG to convert.
        :return: A list of PyZX Circuits and DAGOpNodes corresponding to the DAG.
        """

        circuits_and_nodes: List[Union[zx.Circuit, DAGOpNode]] = []
        qubit_to_index = {qubit: index for index, qubit in enumerate(dag.qubits)}
        clbit_to_index = {clbit: index for index, clbit in enumerate(dag.clbits)}

        current_circuit: Optional[zx.Circuit] = None

        def _ensure_circuit() -> zx.Circuit:
            nonlocal current_circuit
            if current_circuit is None:
                current_circuit = zx.Circuit(
                    len(dag.qubits),
                    bit_amount=len(dag.clbits) if dag.clbits else None,
                )
            return current_circuit

        for node in dag.topological_op_nodes():
            gate = node.op

            if gate.name == "measure" and gate.condition is None:
                _ensure_circuit().add_gate(
                    PyzxMeasurement(
                        qubit_to_index[node.qargs[0]],
                        result_bit=clbit_to_index[node.cargs[0]],
                    )
                )
                continue

            if gate.name == "reset" and gate.condition is None:
                _ensure_circuit().add_gate(
                    PyzxReset(qubit_to_index[node.qargs[0]])
                )
                continue

            # Handle conditional gates: convert supported single-qubit Z/X
            # rotations to PyZX ConditionalGate; otherwise store as DAGOpNode.
            if gate.condition is not None:
                converted = self._try_convert_conditional(node, qubit_to_index)
                if converted is not None:
                    _ensure_circuit().add_gate(converted)
                    continue

            if gate.name not in qiskit_gate_table or gate.condition is not None:
                # Encountered an operation not supported by PyZX (or an unsupported
                # conditional gate), so just store the DAGOpNode.
                # Flush the current PyZX Circuit first if there is one.
                # TODO: It might be possible to do something more clever here by "snipping out"
                # the unsupported operations, optimizing the rest of the circuit, and then
                # reinserting the unsupported operations, but this is very tricky as the unsupported
                # operations may have side effects on the rest of the circuit.
                # See https://github.com/dlyongemallo/qiskit-zx-transpiler/issues/18.
                if current_circuit is not None:
                    circuits_and_nodes.append(current_circuit)
                    current_circuit = None
                circuits_and_nodes.append(node)
                continue

            gate_type, _, num_qubits, num_params, *adjoint = qiskit_gate_table[gate.name]  # type: ignore
            if len(node.qargs) != num_qubits:
                raise ValueError(
                    f"Expected {num_qubits} qubits for gate {gate.name}, got {len(node.qargs)}: "
                    f"{node.qargs}."
                )
            if len(node.op.params) != num_params:
                raise ValueError(
                    f"Expected {num_params} parameters for gate {gate.name}, got {len(node.op.params)}: "
                    f"{node.op.params}."
                )
            kwargs = {"adjoint": adjoint[0]} if adjoint else {}
            _ensure_circuit().add_gate(
                gate_type(
                    *[qubit_to_index[qarg] for qarg in node.qargs],
                    *[Fraction(param / np.pi) for param in node.op.params],
                    **kwargs,
                )
            )

        # Flush any remaining PyZX Circuit.
        if current_circuit is not None:
            circuits_and_nodes.append(current_circuit)

        return circuits_and_nodes

    @staticmethod
    def _recover_conditional_gate(
        gate: ConditionalGate, original_dag: DAGCircuit, dag: DAGCircuit
    ) -> None:
        """Recover a ConditionalGate from a PyZX circuit into a Qiskit DAG."""
        inner = gate.inner_gate
        inner_name = (
            inner.qasm_name
            if not (hasattr(inner, "adjoint") and inner.adjoint)
            else inner.qasm_name_adjoint
        )
        if inner_name not in qiskit_gate_table:
            raise ValueError(
                f"Unsupported inner gate in ConditionalGate: {inner_name}."
            )
        qubit = original_dag.qubits[inner.target]  # type: ignore[attr-defined]
        params: List[float] = []
        num_params = qiskit_gate_table[inner_name][3]
        if num_params > 0 and hasattr(inner, "phase"):
            params = [float(inner.phase) * np.pi]
        _, inner_gate_type, _, _, *_ = qiskit_gate_table[inner_name]
        qiskit_gate = inner_gate_type(*params)
        # Find the classical register by name.
        creg = original_dag.cregs[gate.condition_register]
        qiskit_gate = qiskit_gate.c_if(creg, gate.condition_value)
        dag.apply_operation_back(qiskit_gate, (qubit,))

    def _recover_dag(
        self,
        circuits_and_nodes: List[Union[zx.Circuit, DAGOpNode]],
        original_dag: DAGCircuit,
    ) -> DAGCircuit:
        """Recover a DAG from a list of a pyzx Circuits and DAGOpNodes.

        :param circuits_and_nodes: The list of (optimized) PyZX Circuits and DAGOpNodes from which to recover the DAG.
        :param original_dag: The original input DAG to ZXPass.
        :return: An optimized version of the original input DAG to ZXPass.
        """

        dag = DAGCircuit()
        dag.cregs = original_dag.cregs
        dag.add_clbits(original_dag.clbits)
        dag.qregs = original_dag.qregs
        dag.add_qubits(original_dag.qubits)
        for circuit_or_node in circuits_and_nodes:
            if isinstance(circuit_or_node, DAGOpNode):
                dag.apply_operation_back(
                    circuit_or_node.op, circuit_or_node.qargs, circuit_or_node.cargs
                )
                continue
            for gate in circuit_or_node.gates:
                if isinstance(gate, PyzxMeasurement):
                    qubit = original_dag.qubits[gate.target]
                    clbit = original_dag.clbits[gate.result_bit]
                    dag.apply_operation_back(Measure(), (qubit,), (clbit,))
                    continue

                if isinstance(gate, PyzxReset):
                    qubit = original_dag.qubits[gate.target]
                    dag.apply_operation_back(Reset(), (qubit,))
                    continue

                if isinstance(gate, ConditionalGate):
                    self._recover_conditional_gate(gate, original_dag, dag)
                    continue

                gate_name = (
                    gate.qasm_name
                    if not (hasattr(gate, "adjoint") and gate.adjoint)
                    else gate.qasm_name_adjoint
                )
                if gate_name not in qiskit_gate_table:
                    raise ValueError(f"Unsupported gate: {gate_name}.")
                qargs: List[Qubit] = []
                for attr in ["ctrl1", "ctrl2", "control", "target"]:
                    if hasattr(gate, attr):
                        qargs.append(original_dag.qubits[getattr(gate, attr)])
                params: List[float] = []
                if hasattr(gate, "phase"):
                    params = [float(gate.phase) * np.pi]
                elif hasattr(gate, "phases"):
                    params = [float(phase) * np.pi for phase in gate.phases]
                _, gate_type, _, _, *_ = qiskit_gate_table[gate_name]
                dag.apply_operation_back(gate_type(*params), tuple(qargs))

        return dag

    def run(self, dag: DAGCircuit) -> DAGCircuit:
        """Run the ZX transpiler pass on the given DAG.

        :param dag: The directed acyclic graph to optimize using pyzx.
        :return: The transformed DAG.
        """

        circuits_and_nodes = self._dag_to_circuits_and_nodes(dag)
        if not circuits_and_nodes:
            return dag

        circuits_and_nodes = [
            self.optimize(circuit) if isinstance(circuit, zx.Circuit) else circuit
            for circuit in circuits_and_nodes
        ]

        return self._recover_dag(circuits_and_nodes, dag)

    def name(self) -> str:
        return "ZXPass"
