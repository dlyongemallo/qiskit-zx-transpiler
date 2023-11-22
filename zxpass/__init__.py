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
from qiskit.converters import dag_to_circuit, circuit_to_dag
from qiskit import QuantumCircuit
import pyzx as zx


def _dag_to_circuit(dag: DAGCircuit) -> zx.Circuit:
    """Convert a DAG to a pyzx Circuit.

    :param dag: The DAG to convert.
    :return: The pyzx Circuit corresponding to the DAG.
    """

    # For now, convert it through QASM.
    # TODO: read the DAG's op nodes directly.
    qasm = dag_to_circuit(dag).qasm()
    circ = zx.Circuit.from_qasm(qasm)
    print("Original T-count: ", zx.tcount(circ))
    return circ


def _circuit_to_dag(circ: zx.Circuit) -> DAGCircuit:
    """Convert a pyzx Circuit to a DAG.

    :param circ: The pyzx Circuit to convert.
    :return: The DAG corresponding to the pyzx Circuit.
    """

    # TODO: skip the QASM intermediary step.
    print("Optimized T-count: ", zx.tcount(circ))
    # print the list of gates in circ
    for gate in circ.gates:
        print(gate)
    qasm = circ.to_qasm()
    print(qasm)
    qc = QuantumCircuit.from_qasm_str(qasm)
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

        # print("Before:")
        # zx.draw(g)

        zx.simplify.full_reduce(g)

        # print("After:")
        # zx.draw(g)

        out_dag = _circuit_to_dag(zx.extract.extract_circuit(g))
        return out_dag

    def name(self) -> str:
        return "ZXPass"
