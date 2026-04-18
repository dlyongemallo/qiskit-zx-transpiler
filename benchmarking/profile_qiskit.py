"""Profile Qiskit's optimisation passes on DNN circuits to identify which passes
contribute most to depth reduction."""

from qiskit.circuit import QuantumCircuit
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager


def profile_circuit(subdir, circuit_name):
    print(f"\n{'='*80}")
    print(f"Profiling: {circuit_name}")
    print(f"{'='*80}")

    qc = QuantumCircuit.from_qasm_file(
        f"QASMBench/{subdir}/{circuit_name}/{circuit_name}.qasm"
    )
    print(f"Original - depth: {qc.depth()}, size: {qc.size()}, "
          f"non-local gates: {qc.num_nonlocal_gates()}")
    print(f"Gates: {dict(qc.count_ops())}")

    pm = generate_preset_pass_manager(
        optimization_level=3, basis_gates=["u3", "cx"]
    )

    # Use the callback to record metrics after each pass.
    log = []

    def callback(**kwargs):
        pass_obj = kwargs.get("pass_")
        pass_name = pass_obj.__class__.__name__ if pass_obj is not None else "unknown"
        dag = kwargs.get("dag")
        if dag is None:
            return
        # Read metrics directly from the DAG to avoid the overhead of
        # converting to a QuantumCircuit on every pass.
        depth = dag.depth()
        size = dag.size()
        nonlocal_gates = len(dag.two_qubit_ops()) + len(dag.multi_qubit_ops())
        log.append((pass_name, depth, size, nonlocal_gates))

    pm.run(qc, callback=callback)

    # Print the log, showing only passes that changed any tracked metric.
    prev_depth, prev_size, prev_nonlocal = qc.depth(), qc.size(), qc.num_nonlocal_gates()
    for pass_name, depth, size, nonlocal_gates in log:
        if depth != prev_depth or size != prev_size or nonlocal_gates != prev_nonlocal:
            print(f"\n  {pass_name}:")
            print(f"    Depth: {prev_depth} -> {depth} (delta: {prev_depth - depth:+d})")
            print(f"    Size:  {prev_size} -> {size} (delta: {prev_size - size:+d})")
            print(f"    Non-local: {prev_nonlocal} -> {nonlocal_gates}")
        prev_depth, prev_size, prev_nonlocal = depth, size, nonlocal_gates

    print(f"\n  Final - depth: {prev_depth}, size: {prev_size}, "
          f"non-local gates: {prev_nonlocal}")


if __name__ == "__main__":
    profile_circuit("small", "dnn_n2")
    profile_circuit("small", "dnn_n8")
