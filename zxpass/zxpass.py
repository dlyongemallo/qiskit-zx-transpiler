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
from qiskit.dagcircuit import DAGCircuit, DAGOpNode
from qiskit.circuit import Qubit, Clbit, Instruction

from qiskit.circuit.library import XGate, YGate, ZGate, HGate, SGate, TGate, SXGate
from qiskit.circuit.library import SdgGate, TdgGate, SXdgGate
from qiskit.circuit.library import RXGate, RYGate, RZGate, PhaseGate, U1Gate, U2Gate, U3Gate
from qiskit.circuit.library import SwapGate, CXGate, CYGate, CZGate, CHGate, CSXGate
from qiskit.circuit.library import CRXGate, CRYGate, CRZGate, CPhaseGate, CU1Gate, RXXGate, RZZGate, CU3Gate, CUGate
from qiskit.circuit.library import CSwapGate, CCXGate, CCZGate

import pyzx as zx
from pyzx.circuit.gates import Gate
from pyzx.circuit.gates import NOT, Y, Z, S, T, HAD, SX
from pyzx.circuit.gates import XPhase, YPhase, ZPhase, U2, U3
from pyzx.circuit.gates import SWAP, CNOT, CY, CZ, CHAD, CSX
from pyzx.circuit.gates import CRX, CRY, CRZ, CPhase, RXX, RZZ, CU3, CU
from pyzx.circuit.gates import CSWAP, Tofolli, CCZ

from collections import OrderedDict
import numpy as np
from typing import Dict, List, Tuple, Callable, Optional, Type, Union

qiskit_gate_table: Dict[str, Tuple[Type[Gate], Type[Instruction], int, int]] = {
    # OpenQASM gate name: (PyZX gate type, Qiskit gate type, number of qubits, number of parameters, adjoint)

    'x': (NOT, XGate, 1, 0),
    'y': (Y, YGate, 1, 0),
    'z': (Z, ZGate, 1, 0),
    'h': (HAD, HGate, 1, 0),
    's': (S, SGate, 1, 0),
    't': (T, TGate, 1, 0),
    'sx': (SX, SXGate, 1, 0),

    'sdg': (S, SdgGate, 1, 0, True),     # type: ignore
    'tdg': (T, TdgGate, 1, 0, True),     # type: ignore
    'sxdg': (SX, SXdgGate, 1, 0, True),  # type: ignore

    'rx': (XPhase, RXGate, 1, 1),
    'ry': (YPhase, RYGate, 1, 1),
    'rz': (ZPhase, RZGate, 1, 1),
    'p': (ZPhase, PhaseGate, 1, 1),
    'u1': (ZPhase, U1Gate, 1, 1),
    'u2': (U2, U2Gate, 1, 2),
    'u3': (U3, U3Gate, 1, 3),

    'swap': (SWAP, SwapGate, 2, 0),
    'cx': (CNOT, CXGate, 2, 0),
    'cy': (CY, CYGate, 2, 0),
    'cz': (CZ, CZGate, 2, 0),
    'ch': (CHAD, CHGate, 2, 0),
    'csx': (CSX, CSXGate, 2, 0),

    'crx': (CRX, CRXGate, 2, 1),
    'cry': (CRY, CRYGate, 2, 1),
    'crz': (CRZ, CRZGate, 2, 1),
    'cp': (CPhase, CPhaseGate, 2, 1),
    'cphase': (CPhase, CPhaseGate, 2, 1),
    'cu1': (CPhase, CU1Gate, 2, 1),
    'rxx': (RXX, RXXGate, 2, 1),
    'rzz': (RZZ, RZZGate, 2, 1),
    'cu3': (CU3, CU3Gate, 2, 3),
    'cu': (CU, CUGate, 2, 4),

    'cswap': (CSWAP, CSwapGate, 3, 0),
    'ccx': (Tofolli, CCXGate, 3, 0),
    'ccz': (CCZ, CCZGate, 3, 0),
}


class ZXPass(TransformationPass):
    """This is a ZX transpiler pass using PyZX for circuit optimization.

    :param optimize: The function to use for optimizing a PyZX Circuit. If not specified, uses
        :py:meth:`~pyzx.simplify.full_reduce` by default.
    :type optimize: Callable[[pyzx.Circuit], pyzx.Circuit], optional
    """

    def __init__(self, optimize: Optional[Callable[[zx.Circuit], zx.Circuit]] = None):
        super().__init__()

        self.cregs: OrderedDict[str, Clbit] = OrderedDict()
        self.clbits: List[Clbit] = []
        self.qregs: OrderedDict[str, Qubit] = OrderedDict()
        self.qubits: List[Qubit] = []
        self.qubit_to_index: Dict[Qubit, int] = {}
        self.optimize: Callable[[zx.Circuit], zx.Circuit] = optimize or self._optimize

    def _optimize(self, c: zx.Circuit) -> zx.Circuit:
        g = c.to_graph()
        zx.simplify.full_reduce(g)
        return zx.extract.extract_circuit(g)

    def _dag_to_circuits_and_nodes(self, dag: DAGCircuit) -> List[Union[zx.Circuit, DAGOpNode]]:
        """Convert a DAG to a list of PyZX Circuits and DAGOpNodes. As much of the DAG is converted to PyZX Circuits as
        possible, but some gates are not supported by PyZX and are left as DAGOpNodes.

        :param dag: The DAG to convert.
        :return: A list of PyZX Circuits and DAGOpNodes corresponding to the DAG.
        """

        circuits_and_nodes: List[Union[zx.Circuit, DAGOpNode]] = []
        self.cregs = dag.cregs
        self.clbits = dag.clbits
        self.qregs = dag.qregs
        self.qubits = dag.qubits
        self.qubit_to_index = {qubit: index for index, qubit in enumerate(dag.qubits)}

        current_circuit: Optional[zx.Circuit] = None
        for node in dag.topological_op_nodes():
            gate = node.op
            # TODO: It might be possible to do something more clever here by "snipping out" the unsupported operations,
            #       optimizing the rest of the circuit, and then reinserting the unsupported operations, but this is
            #       very tricky as the unsupported operations may have side effects on the rest of the circuit.
            if gate.name not in qiskit_gate_table or gate.condition is not None:
                # Encountered an operation not supported by PyZX, so just store the DAGOpNode.
                # Flush the current PyZX Circuit first if there is one.
                if current_circuit is not None:
                    circuits_and_nodes.append(current_circuit)
                    current_circuit = None
                circuits_and_nodes.append(node)
                continue
            gate_type, _, num_qubits, num_params, *adjoint = qiskit_gate_table[gate.name]  # type: ignore
            if len(node.qargs) != num_qubits:
                raise ValueError(f"Expected {num_qubits} qubits for gate {gate.name}, got {len(node.qargs)}: "
                                 f"{node.qargs}.")
            if len(node.op.params) != num_params:
                raise ValueError(f"Expected {num_params} parameters for gate {gate.name}, got {len(node.op.params)}: "
                                 f"{node.op.params}.")
            kwargs = {'adjoint': adjoint[0]} if adjoint else {}
            if current_circuit is None:
                current_circuit = zx.Circuit(len(dag.qubits))
            current_circuit.add_gate(gate_type(*[self.qubit_to_index[qarg] for qarg in node.qargs],  # type: ignore
                                               *[param / np.pi for param in node.op.params],         # type: ignore
                                               **kwargs))                                            # type: ignore

        # Flush any remaining PyZX Circuit.
        if current_circuit is not None:
            circuits_and_nodes.append(current_circuit)

        return circuits_and_nodes

    def _recover_dag(self, circuits_and_nodes: List[Union[zx.Circuit, DAGOpNode]]) -> DAGCircuit:
        """Recovers a DAG from a list of a pyzx Circuits and DAGOpNodes

        :param circuits_and_nodes: The list of (optimized) PyZX Circuits and DAGOpNodes from which to recover the DAG.
        :return: An optimized version of the original input DAG to ZXPass.
        """

        dag = DAGCircuit()
        dag.cregs = self.cregs
        dag.add_clbits(self.clbits)
        dag.qregs = self.qregs
        dag.add_qubits(self.qubits)
        for circuit in circuits_and_nodes:
            if isinstance(circuit, DAGOpNode):
                dag.apply_operation_back(circuit.op, circuit.qargs, circuit.cargs)
                continue
            for gate in circuit.gates:
                gate_name = gate.qasm_name if not (hasattr(gate, 'adjoint') and gate.adjoint) \
                    else gate.qasm_name_adjoint
                if gate_name not in qiskit_gate_table:
                    raise ValueError(f"Unsupported gate: {gate_name}.")
                qargs: List[Qubit] = []
                for attr in ['ctrl1', 'ctrl2', 'control', 'target']:
                    if hasattr(gate, attr):
                        qargs.append(self.qubits[getattr(gate, attr)])
                params: List[float] = []
                if hasattr(gate, 'phase'):
                    params = [float(gate.phase) * np.pi]
                elif hasattr(gate, 'phases'):
                    params = [float(phase) * np.pi for phase in gate.phases]
                _, gate_type, num_qubits, num_params, *adjoint = qiskit_gate_table[gate_name]  # type: ignore
                dag.apply_operation_back(gate_type(*params), tuple(qargs))

        return dag

    def run(self, dag: DAGCircuit) -> DAGCircuit:
        """
        Run the ZX transpiler pass on the given DAG.

        :param dag: The directed acyclic graph to optimize using pyzx.
        :return: The transformed DAG.
        """

        circuits_and_nodes = self._dag_to_circuits_and_nodes(dag)
        if not circuits_and_nodes:
            return dag

        circuits_and_nodes = [self.optimize(circuit) if isinstance(circuit, zx.Circuit) else circuit
                              for circuit in circuits_and_nodes]

        return self._recover_dag(circuits_and_nodes)

    def name(self) -> str:
        return "ZXPass"
