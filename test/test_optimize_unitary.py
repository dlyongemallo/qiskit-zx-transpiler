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

"""Tests for ``_optimize_unitary``: post-extraction cleanup, gate-count fallback, SWAP prefix."""

# pylint: disable=duplicate-code

from fractions import Fraction

import pyzx as zx
from pyzx.circuit.gates import SWAP

from qiskit.circuit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.transpiler import PassManager
import qiskit.converters

from zxpass import ZXPass
from zxpass.zxpass import (
    _optimize_unitary,
    _permutation_to_swaps,
    compute_output_permutation,
)

from ._helpers import assert_equiv, run_zxpass


def test_post_extraction_cleanup() -> None:
    """Test that ``_optimize_unitary`` applies ``basic_optimization`` after extraction.

    Extraction produces CZ + HAD form with redundant single-qubit gates;
    ``basic_optimization`` cancels those and folds HAD-CZ-HAD into CNOT.
    Compares against naive extraction to verify the post-pass actually ran:
    if the ``basic_optimization`` step were removed, ``_optimize_unitary``
    would return a circuit no smaller than ``extract_circuit`` alone and
    this assertion would fail.
    """
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
    """Test that post-extraction cleanup and permutation integration preserve equivalence."""
    qc1 = QuantumCircuit(3)
    qc1.h(0)
    qc1.cx(0, 1)
    qc1.cx(1, 2)
    qc1.h(2)
    qc1.cx(2, 0)

    qc2 = QuantumCircuit(5)
    for i in range(5):
        qc2.h(i)
    for i in range(4):
        qc2.cx(i, i + 1)
    qc2.cx(4, 0)
    for i in range(5):
        qc2.rz(0.3 * (i + 1), i)

    qc3 = QuantumCircuit(4)
    qc3.h(0)
    qc3.cx(0, 1)
    qc3.h(1)
    qc3.cx(1, 2)
    qc3.h(2)
    qc3.cx(2, 3)
    qc3.cx(3, 0)
    qc3.cx(2, 1)
    qc3.cx(1, 0)

    for qc in [qc1, qc2, qc3]:
        assert run_zxpass(qc), f"Equivalence failed for circuit:\n{qc}"


def test_permutation_swaps_in_pipeline() -> None:
    """Test that _optimize_unitary prepends SWAP gates for the output permutation.

    After extraction with up_to_perm=True, the output permutation should be
    prepended as SWAP gate objects.  Each SWAP counts as one gate instead of
    three CNOTs, improving the gate-count comparison.
    """
    # Build a circuit that produces a non-trivial output permutation and is
    # large enough that the gate-count fallback does not trigger.
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

    # Replay the same extraction path to compute the SWAP prefix that
    # ``_optimize_unitary`` should prepend.
    g = c.to_graph()
    zx.simplify.full_reduce(g)
    zx.extract.extract_circuit(g, up_to_perm=True)
    expected_swaps = _permutation_to_swaps(compute_output_permutation(g))
    # Sanity-check that this circuit exercises the SWAP-prefix path, so the
    # test does not silently degenerate into a trivial no-op assertion.
    assert expected_swaps, (
        "Test circuit no longer produces a non-trivial output permutation. "
        "This is likely a PyZX extraction-heuristic change (pyzx is unpinned) "
        "rather than a regression in _optimize_unitary. Pick a different "
        "circuit that yields a non-identity permutation under the current "
        "PyZX version, or pin pyzx to a version where this circuit does."
    )

    optimized = _optimize_unitary(c)

    # Confirm the fallback did not trigger for this circuit; otherwise the
    # SWAP-prefix assertions below would be vacuously skipped.
    assert list(optimized.gates) != list(c.gates), (
        "Test circuit unexpectedly hit the gate-count fallback. This is "
        "likely a PyZX optimisation-heuristic change (pyzx is unpinned) "
        "rather than a regression in _optimize_unitary. Pick a different "
        "circuit that reliably reduces in gate count under the current "
        "PyZX version, or pin pyzx to a version where this circuit does."
    )

    # The optimized circuit should begin with exactly the expected SWAP prefix.
    assert len(optimized.gates) >= len(expected_swaps)
    for k, (i, j) in enumerate(expected_swaps):
        gate = optimized.gates[k]
        assert isinstance(gate, SWAP), (
            f"Expected SWAP at position {k}, got {type(gate).__name__}."
        )
        assert {gate.control, gate.target} == {i, j}, (
            f"SWAP at position {k} acts on {{{gate.control}, {gate.target}}}, "
            f"expected {{{i}, {j}}}."
        )
    # No SWAPs should appear after the prefix.
    for gate in optimized.gates[len(expected_swaps):]:
        assert not isinstance(gate, SWAP), (
            "SWAP gates should only appear at the beginning of the circuit."
        )


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
    assert_equiv(qc, result)


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
    assert_equiv(qc, result)


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
    assert_equiv(qc, result)


def test_gate_count_fallback_fredkin_compact() -> None:
    """Test the fallback preserves a compact Fredkin (single CSWAP instruction)."""
    qc = QuantumCircuit(3)
    qc.cswap(0, 1, 2)

    zxpass = ZXPass()
    result = PassManager(zxpass).run(qc)

    assert result.size() <= qc.size(), (
        f"Gate-count fallback failed: {result.size()} > {qc.size()}"
    )
    assert_equiv(qc, result)


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
    assert_equiv(qc, result)


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
