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

from qiskit.transpiler.basepasses import TransformationPass
from qiskit.dagcircuit import DAGCircuit
from qiskit.converters import circuit_to_dag
from qiskit import QuantumCircuit
import pyzx as zx
from pyzx.circuit.gates import Gate
from pyzx.circuit.gates import NOT, Y, Z, S, T, HAD, SX
from pyzx.circuit.gates import XPhase, YPhase, ZPhase, U2, U3
from pyzx.circuit.gates import SWAP, CNOT, CY, CZ, CHAD, CSX
from pyzx.circuit.gates import CRX, CRY, CRZ, CPhase, RXX, RZZ, CU3, CU
from pyzx.circuit.gates import CSWAP, Tofolli, CCZ
import numpy as np
from typing import Dict, List, Tuple, Type, Callable

qiskit_gate_table: Dict[str, Tuple[Type[Gate], int, int, bool]] = {
    'x': (NOT, 1, 0),
    'y': (Y, 1, 0),
    'z': (Z, 1, 0),
    'h': (HAD, 1, 0),
    's': (S, 1, 0),
    't': (T, 1, 0),
    'sx': (SX, 1, 0),

    'sdg': (S, 1, 0, True),
    'tdg': (T, 1, 0, True),
    'sxdg': (SX, 1, 0, True),

    'rx': (XPhase, 1, 1),
    'ry': (YPhase, 1, 1),
    'rz': (ZPhase, 1, 1),
    'p': (ZPhase, 1, 1),
    'u1': (ZPhase, 1, 1),
    'u2': (U2, 1, 2),
    'u3': (U3, 1, 3),

    'swap': (SWAP, 2, 0),
    'cx': (CNOT, 2, 0),
    'cy': (CY, 2, 0),
    'cz': (CZ, 2, 0),
    'ch': (CHAD, 2, 0),
    'csx': (CSX, 2, 0),

    'crx': (CRX, 2, 1),
    'cry': (CRY, 2, 1),
    'crz': (CRZ, 2, 1),
    'cp': (CPhase, 2, 1),
    'cphase': (CPhase, 2, 1),
    'cu1': (CPhase, 2, 1),
    'rxx': (RXX, 2, 1),
    'rzz': (RZZ, 2, 1),
    'cu3': (CU3, 2, 3),
    'cu': (CU, 2, 4),

    'cswap': (CSWAP, 3, 0),
    'ccx': (Tofolli, 3, 0),
    'ccz': (CCZ, 3, 0),
}


def _dag_to_circuit(dag: DAGCircuit) -> zx.Circuit:
    """Convert a DAG to a pyzx Circuit.

    :param dag: The DAG to convert.
    :return: The pyzx Circuit corresponding to the DAG.
    """

    gates: List[Gate] = []
    for node in dag.topological_op_nodes():
        gate = node.op
        if gate.name not in qiskit_gate_table:
            raise ValueError(f"Unsupported gate: {gate.name}.")
        gate_type, num_qubits, num_params, *adjoint = qiskit_gate_table[gate.name]
        assert len(node.qargs) == num_qubits
        assert len(node.op.params) == num_params
        kwargs = {'adjoint': adjoint[0]} if adjoint else {}
        gates.append(gate_type(*[qarg.index for qarg in node.qargs], *[param / np.pi for param in node.op.params], **kwargs))  # type: ignore
    # TODO: Properly handle number of qubits. The following assumes there is only one quantum register.
    num_qubits = len(dag.qubits)
    circ = zx.Circuit(num_qubits)
    circ.gates = gates

    return circ


def _circuit_to_dag(circ: zx.Circuit) -> DAGCircuit:
    """Convert a pyzx Circuit to a DAG.

    :param circ: The pyzx Circuit to convert.
    :return: The DAG corresponding to the pyzx Circuit.
    """

    # TODO: skip the QASM intermediary step.
    qc = QuantumCircuit.from_qasm_str(circ.to_qasm())
    dag = circuit_to_dag(qc)
    return dag


class ZXPass(TransformationPass):
    """This is a ZX transpiler pass using pyzx for circuit optimization.

    :param optimize: The function to use for optimizing the pyzx Circuit. If not specified, uses
        :py:meth:`~pyzx.simplify.full_reduce` by default.
    :type optimize: Callable[[pyzx.Circuit], pyzx.Circuit], optional
    """

    def __init__(self, optimize: Callable[[zx.Circuit], zx.Circuit] = None):
        super().__init__()

        self.optimize: Callable[[zx.Circuit], zx.Circuit] = optimize or self._optimize

    def _optimize(self, c: zx.Circuit) -> zx.Circuit:
        g = c.to_graph()
        zx.simplify.full_reduce(g)
        return zx.extract.extract_circuit(g)

    def run(self, dag: DAGCircuit) -> DAGCircuit:
        """
        Run the ZX transpiler pass on the given DAG.

        :param dag: The directed acyclic graph to optimize using pyzx.
        :return: The transformed DAG.
        """

        return _circuit_to_dag(self.optimize(_dag_to_circuit(dag)))

    def name(self) -> str:
        return "ZXPass"
