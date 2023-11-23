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
from pyzx.circuit.gates import CRX, CRY, CRZ, CPhase, RXX, RZZ
from pyzx.circuit.gates import CSWAP, Tofolli, CCZ
import numpy as np
from typing import Dict, Type

qiskit_gate_table: Dict[str, Type[Gate]] = {
    'x': NOT,
    'y': Y,
    'z': Z,
    'h': HAD,
    's': S,
    't': T,
    'sx': SX,

    'sdg': S,
    'tdg': T,
    'sxdg': SX,

    'rx': XPhase,
    'ry': YPhase,
    'rz': ZPhase,
    'p': ZPhase,
    'u1': ZPhase,
    'u2': U2,
    'u3': U3,

    'swap': SWAP,
    'cx': CNOT,
    'cy': CY,
    'cz': CZ,
    'ch': CHAD,
    'csx': CSX,

    'crx': CRX,
    'cry': CRY,
    'crz': CRZ,
    'cp': CPhase,
    'cphase': CPhase,
    'cu1': CPhase,
    'rxx': RXX,
    'rzz': RZZ,

    'cswap': CSWAP,
    'ccx': Tofolli,
    'ccz': CCZ,
}


def _dag_to_circuit(dag: DAGCircuit) -> zx.Circuit:
    """Convert a DAG to a pyzx Circuit.

    :param dag: The DAG to convert.
    :return: The pyzx Circuit corresponding to the DAG.
    """

    gates: list[Gate] = []
    for node in dag.topological_op_nodes():
        gate = node.op
        print(gate)
        if gate.name in ('x', 'y', 'z', 'h', 's', 't', 'sx'):
            assert len(node.qargs) == 1
            assert len(node.op.params) == 0
            gates.append(qiskit_gate_table[gate.name](node.qargs[0].index))  # type: ignore
        elif gate.name in ('sdg', 'tdg', 'sxdg'):
            assert len(node.qargs) == 1
            assert len(node.op.params) == 0
            gates.append(qiskit_gate_table[gate.name](node.qargs[0].index), adjoint=True)  # type: ignore
        elif gate.name in ('rx', 'ry', 'rz', 'p', 'u1'):
            assert len(node.qargs) == 1
            assert len(node.op.params) == 1
            gates.append(qiskit_gate_table[gate.name](node.qargs[0].index, node.op.params[0] / np.pi))  # type: ignore
        elif gate.name == 'u2':
            assert len(node.qargs) == 1
            assert len(node.op.params) == 2
            gates.append(qiskit_gate_table[gate.name](node.qargs[0].index, node.op.params[0] / np.pi, node.op.params[1] / np.pi))  # type: ignore
        elif gate.name == 'u3':
            assert len(node.qargs) == 1
            assert len(node.op.params) == 3
            gates.append(qiskit_gate_table[gate.name](node.qargs[0].index, node.op.params[0] / np.pi, node.op.params[1] / np.pi, node.op.params[2] / np.pi))  # type: ignore
        elif gate.name in ('swap', 'cx', 'cy', 'cz', 'ch', 'csx'):
            assert len(node.qargs) == 2
            assert len(node.op.params) == 0
            gates.append(qiskit_gate_table[gate.name](node.qargs[0].index, node.qargs[1].index))  # type: ignore
        elif gate.name in ('crx', 'cry', 'crz', 'cp', 'cphase', 'cu1', 'rxx', 'rzz'):
            assert len(node.qargs) == 2
            assert len(node.op.params) == 1
            gates.append(qiskit_gate_table[gate.name](node.qargs[0].index, node.qargs[1].index, node.op.params[0] / np.pi))  # type: ignore
        elif gate.name in ('cswap', 'ccx', 'ccz'):
            assert len(node.qargs) == 3
            assert len(node.op.params) == 0
            gates.append(qiskit_gate_table[gate.name](node.qargs[0].index, node.qargs[1].index, node.qargs[2].index))  # type: ignore
        elif gate.name == 'cu3':
            assert len(node.qargs) == 2
            assert len(node.op.params) == 3
            gates.append(qiskit_gate_table[gate.name](node.qargs[0].index, node.qargs[1].index, node.op.params[0] / np.pi, node.op.params[1] / np.pi, node.op.params[2] / np.pi))  # type: ignore
        elif gate.name == 'cu':
            assert len(node.qargs) == 2
            assert len(node.op.params) == 4
            gates.append(qiskit_gate_table[gate.name](node.qargs[0].index, node.qargs[1].index, node.op.params[0] / np.pi, node.op.params[1] / np.pi, node.op.params[2] / np.pi, node.op.params[3] / np.pi))  # type: ignore
        else:
            raise ValueError(f"Unsupported gate: {gate.name}.")
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
    """

    def __init__(self):
        super().__init__()

    def run(self, dag: DAGCircuit) -> DAGCircuit:
        """
        Run the ZX transpiler pass on the given DAG.

        :param dag: The directed acyclic graph to optimize using pyzx.
        :return: The transformed DAG.
        """

        zx_circ = _dag_to_circuit(dag)
        g = zx_circ.to_graph()
        zx.simplify.full_reduce(g)
        out_dag = _circuit_to_dag(zx.extract.extract_circuit(g))
        return out_dag

    def name(self) -> str:
        return "ZXPass"
