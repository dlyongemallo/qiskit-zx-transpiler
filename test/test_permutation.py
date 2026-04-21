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

"""Tests for the output-permutation utility and SWAP decomposition helper."""

from typing import Any

import pyzx as zx
import pytest

from zxpass.zxpass import compute_output_permutation, _permutation_to_swaps


def test_compute_output_permutation() -> None:
    """Test compute_output_permutation on a graph after extraction."""
    c = zx.Circuit(3)
    c.add_gate("CNOT", 0, 1)
    c.add_gate("CNOT", 1, 2)
    c.add_gate("HAD", 0)
    c.add_gate("CNOT", 2, 0)
    c.add_gate("CNOT", 0, 1)
    c.add_gate("HAD", 2)
    c.add_gate("CNOT", 1, 0)

    g = c.to_graph()
    zx.simplify.full_reduce(g)
    zx.extract.extract_circuit(g, up_to_perm=True)
    perm = compute_output_permutation(g)

    # The permutation should be a valid bijection on {0, 1, 2}.
    assert set(perm.keys()) == set(range(3))
    assert set(perm.values()) == set(range(3))
    assert len(perm) == 3


def test_compute_output_permutation_identity() -> None:
    """Test compute_output_permutation when the permutation is identity."""
    # A single CNOT should not require any output permutation.
    c = zx.Circuit(2)
    c.add_gate("CNOT", 0, 1)

    g = c.to_graph()
    zx.simplify.full_reduce(g)
    zx.extract.extract_circuit(g, up_to_perm=True)
    perm = compute_output_permutation(g)

    assert perm == {0: 0, 1: 1}


def test_compute_output_permutation_matches_swaps() -> None:
    """Test that the permutation matches the SWAPs that extract_circuit adds.

    Extracts the same graph with and without ``up_to_perm``.  Prepending SWAPs
    corresponding to ``compute_output_permutation`` onto the ``up_to_perm=True``
    circuit should produce a circuit equivalent (up to global phase) to the
    original, matching what ``up_to_perm=False`` extraction already yields.
    """
    c = zx.Circuit(3)
    c.add_gate("CNOT", 0, 1)
    c.add_gate("CNOT", 1, 2)
    c.add_gate("HAD", 0)
    c.add_gate("CNOT", 2, 0)
    c.add_gate("CNOT", 0, 1)
    c.add_gate("HAD", 2)
    c.add_gate("CNOT", 1, 0)

    # Extract with up_to_perm=True and compute the permutation.
    g1 = c.to_graph()
    zx.simplify.full_reduce(g1)
    c_no_perm = zx.extract.extract_circuit(g1, up_to_perm=True)
    perm = compute_output_permutation(g1)

    # Extract with up_to_perm=False (adds SWAPs).
    g2 = c.to_graph()
    zx.simplify.full_reduce(g2)
    c_with_swaps = zx.extract.extract_circuit(g2, up_to_perm=False)

    # The version with SWAPs should have at least as many gates.
    assert len(c_with_swaps.gates) >= len(c_no_perm.gates)

    # up_to_perm=False is equivalent to the original (phase-insensitive).
    assert c_with_swaps.verify_equality(c), "up_to_perm=False should match original"

    # Decompose perm into SWAPs via cycle decomposition. Applied left-to-right,
    # the sequence permutes wire labels so that wire j ends up holding input
    # perm[j], which is exactly what c_no_perm expects at its inputs.
    n = c.qubits
    current = [perm[i] for i in range(n)]
    swap_seq = []
    for i in range(n):
        while current[i] != i:
            j = current[i]
            swap_seq.append((i, j))
            current[i], current[j] = current[j], current[i]
    swap_seq.reverse()

    # Prepending those SWAPs to c_no_perm should reproduce the original circuit.
    combined = zx.Circuit(n)
    for i, j in swap_seq:
        combined.add_gate("SWAP", i, j)
    for gate in c_no_perm.gates:
        combined.add_gate(gate)
    assert combined.verify_equality(c), (
        "Prepending SWAPs derived from perm onto c_no_perm should match the original."
    )


def _extracted_graph(num_qubits: int) -> Any:
    """Build a small post-extraction graph for error-branch tests."""
    c = zx.Circuit(num_qubits)
    for i in range(num_qubits - 1):
        c.add_gate("CNOT", i, i + 1)
    g = c.to_graph()
    zx.simplify.full_reduce(g)
    zx.extract.extract_circuit(g, up_to_perm=True)
    return g


def test_compute_output_permutation_mismatched_counts() -> None:
    """Raises ValueError when input and output counts differ."""
    g = _extracted_graph(2)
    # Drop one output so the counts no longer match.
    g.set_outputs(tuple(list(g.outputs())[:1]))
    with pytest.raises(ValueError, match="equal numbers of inputs and outputs"):
        compute_output_permutation(g)


def test_compute_output_permutation_wrong_degree() -> None:
    """Raises ValueError when an output vertex has more than one neighbour."""
    g = _extracted_graph(2)
    outputs = list(g.outputs())
    # Add an extra edge between the two outputs so one has two neighbours.
    g.add_edge((outputs[0], outputs[1]))
    with pytest.raises(ValueError, match="exactly one"):
        compute_output_permutation(g)


def test_compute_output_permutation_non_input_neighbor() -> None:
    """Raises ValueError when an output is connected to a non-input vertex."""
    g = _extracted_graph(2)
    out0 = list(g.outputs())[0]
    nbr = list(g.neighbors(out0))[0]
    g.remove_edge((nbr, out0))
    # Wire the output to a fresh interior vertex instead of an input.
    new_v = g.add_vertex()
    g.add_edge((new_v, out0))
    with pytest.raises(ValueError, match="not an input vertex"):
        compute_output_permutation(g)


def test_compute_output_permutation_non_bijective() -> None:
    """Raises ValueError when two outputs map to the same input."""
    g = _extracted_graph(2)
    inputs = list(g.inputs())
    outputs = list(g.outputs())
    # Point both outputs at inputs[0].
    for out in outputs:
        for n in list(g.neighbors(out)):
            g.remove_edge((n, out))
        g.add_edge((inputs[0], out))
    with pytest.raises(ValueError, match="bijective"):
        compute_output_permutation(g)


def test_permutation_to_swaps_correctness() -> None:
    """Test that _permutation_to_swaps produces a correct SWAP sequence."""
    # Identity permutation: no SWAPs needed.
    assert not _permutation_to_swaps({0: 0, 1: 1, 2: 2})

    # Simple transposition.
    swaps = _permutation_to_swaps({0: 1, 1: 0})
    assert len(swaps) == 1
    assert set(swaps[0]) == {0, 1}

    # 3-cycle: needs 2 transpositions.
    perm = {0: 1, 1: 2, 2: 0}
    swaps = _permutation_to_swaps(perm)
    # Verify the SWAPs implement the permutation.
    state = list(range(3))
    for i, j in swaps:
        state[i], state[j] = state[j], state[i]
    assert state == [perm[k] for k in range(3)]

    # Larger permutation with multiple cycles.
    perm = {0: 2, 1: 0, 2: 1, 3: 4, 4: 3}
    swaps = _permutation_to_swaps(perm)
    state = list(range(5))
    for i, j in swaps:
        state[i], state[j] = state[j], state[i]
    assert state == [perm[k] for k in range(5)]


def test_permutation_to_swaps_invalid_input() -> None:
    """Test that _permutation_to_swaps rejects non-bijective input."""
    # Non-contiguous keys (missing 1).
    with pytest.raises(ValueError, match="keys to be range"):
        _permutation_to_swaps({0: 0, 2: 2})

    # Out-of-range keys (shifted by 1).
    with pytest.raises(ValueError, match="keys to be range"):
        _permutation_to_swaps({1: 1, 2: 2})

    # Values outside range.
    with pytest.raises(ValueError, match="values to be a permutation"):
        _permutation_to_swaps({0: 5, 1: 1})

    # Repeated values (not a permutation).
    with pytest.raises(ValueError, match="values to be a permutation"):
        _permutation_to_swaps({0: 1, 1: 1})
