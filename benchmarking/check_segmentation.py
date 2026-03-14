"""Check the segmentation impact of ZXPass on DNN circuits.
Shows how many segments are created and what causes the splits.

This script analyses both the DAG-level segmentation (PyZX circuits vs.
unsupported DAGOpNodes) and the non-unitary boundary splitting that
_optimize() performs within each PyZX circuit segment."""

import sys
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from qiskit.circuit import QuantumCircuit
from qiskit.converters import circuit_to_dag
from qiskit.transpiler import PassManager

import pyzx as zx

from zxpass import ZXPass
from zxpass.zxpass import _is_unitary_gate


def _gate_breakdown(gates):
    """Return a dict of gate type name to count."""
    counts = {}
    for gate in gates:
        name = type(gate).__name__
        counts[name] = counts.get(name, 0) + 1
    return counts


def analyze_segmentation(subdir, circuit_name):
    print(f"\n{'='*80}")
    print(f"Segmentation analysis: {circuit_name}")
    print(f"{'='*80}")

    qc = QuantumCircuit.from_qasm_file(
        f"QASMBench/{subdir}/{circuit_name}/{circuit_name}.qasm"
    )
    print(f"Original - depth: {qc.depth()}, size: {qc.size()}, "
          f"non-local gates: {qc.num_nonlocal_gates()}")
    print(f"Gates: {dict(qc.count_ops())}")

    # DAG-level segmentation (PyZX circuits vs. unsupported DAGOpNodes).
    dag = circuit_to_dag(qc)
    zxpass = ZXPass()
    circuits_and_nodes = zxpass._dag_to_circuits_and_nodes(dag)

    num_zx_circuits = sum(1 for x in circuits_and_nodes if isinstance(x, zx.Circuit))
    num_dag_nodes = sum(1 for x in circuits_and_nodes if not isinstance(x, zx.Circuit))

    print(f"\nDAG-level segments: {len(circuits_and_nodes)} total")
    print(f"  PyZX circuits: {num_zx_circuits}")
    print(f"  Unsupported DAGOpNodes: {num_dag_nodes}")

    # Analyse each segment, including non-unitary boundary splitting.
    for i, item in enumerate(circuits_and_nodes):
        if not isinstance(item, zx.Circuit):
            print(f"\n  Segment {i}: DAGOpNode - {item.op.name} on qubits {item.qargs}")
            continue

        print(f"\n  Segment {i}: PyZX Circuit")
        print(f"    Gates: {len(item.gates)}, Qubits: {item.qubits}")
        print(f"    Gate breakdown: {_gate_breakdown(item.gates)}")

        # Show how _optimize() splits this segment at non-unitary boundaries.
        unitary_segments = []
        non_unitary_gates = []
        current = []
        for gate in item.gates:
            if _is_unitary_gate(gate):
                current.append(gate)
            else:
                if current:
                    unitary_segments.append(current)
                    current = []
                non_unitary_gates.append(gate)
        if current:
            unitary_segments.append(current)

        if non_unitary_gates:
            print(f"    Non-unitary boundaries: {len(non_unitary_gates)} "
                  f"({_gate_breakdown(non_unitary_gates)})")
            print(f"    Unitary sub-segments: {len(unitary_segments)}")

        # Optimise each unitary sub-segment.
        for j, seg_gates in enumerate(unitary_segments):
            seg = zx.Circuit(item.qubits)
            for gate in seg_gates:
                seg.add_gate(gate)
            graph = seg.to_graph()
            zx.simplify.full_reduce(graph)
            try:
                opt = zx.extract.extract_circuit(graph)
                label = f"sub-segment {j}" if len(unitary_segments) > 1 else "unitary part"
                print(f"    Optimised {label}: {len(seg_gates)} -> {len(opt.gates)} gates")
                print(f"      Output breakdown: {_gate_breakdown(opt.gates)}")
            except Exception:  # pylint: disable=broad-except
                print(f"    Optimise sub-segment {j} failed:")
                traceback.print_exc()

    # Run ZXPass end-to-end and show the result.
    print(f"\n--- ZXPass result ---")
    pm = PassManager(ZXPass())
    zx_qc = pm.run(qc)
    original_depth = qc.depth()
    optimised_depth = zx_qc.depth()
    print(f"  ZX-optimised - depth: {optimised_depth}, size: {zx_qc.size()}, "
          f"non-local gates: {zx_qc.num_nonlocal_gates()}")
    print(f"  Gates: {dict(zx_qc.count_ops())}")
    if optimised_depth == 0:
        print("  Depth compression ratio: undefined (optimised circuit depth is 0)")
    else:
        print(f"  Depth compression ratio: {original_depth / optimised_depth:.2f}")


if __name__ == "__main__":
    analyze_segmentation("small", "dnn_n2")
    analyze_segmentation("small", "dnn_n8")
