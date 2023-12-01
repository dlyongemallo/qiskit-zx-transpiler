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

if __name__ == '__main__':
    import sys
    sys.path.append('..')

from qiskit import QuantumCircuit, transpile
from qiskit.transpiler import PassManager

from zxpass import ZXPass

zxpass = ZXPass()
pass_manager = PassManager(zxpass)


def _benchmark(subdir: str, circuit_name: str) -> None:
    qc = QuantumCircuit.from_qasm_file(f"QASMBench/{subdir}/{circuit_name}/{circuit_name}.qasm")
    opt_qc = transpile(qc, basis_gates=['u3', 'cx'], optimization_level=3)
    zx_qc = pass_manager.run(qc)

    print(f"Circuit name: {circuit_name}")
    print(f"Size - original: {qc.size()}, "
          f"optimized: {opt_qc.size()} ({qc.size() / opt_qc.size():.2f}), "
          f"zx: {zx_qc.size()} ({qc.size() / zx_qc.size():.2f})")
    print(f"Depth - original: {qc.depth()}, "
          f"optimized: {opt_qc.depth()} ({qc.depth() / opt_qc.depth():.2f}), "
          f"zx: {zx_qc.depth()} ({qc.depth() / zx_qc.depth():.2f})")
    print()


def run_benchmarks() -> None:
    # List of circuits to benchmark, based on: https://github.com/Qiskit/qiskit/issues/4990#issuecomment-1157858632
    small_benchmarks = ['wstate_n3', 'linearsolver_n3', 'fredkin_n3', 'dnn_n2', 'qrng_n4', 'adder_n4', 'deutsch_n2',
                        'cat_state_n4', 'basis_trotter_n4', 'qec_en_n5', 'toffoli_n3', 'grover_n2', 'hs4_n4', 'qaoa_n3',
                        'teleportation_n3', 'lpn_n5', 'vqe_uccsd_n4', 'quantumwalks_n2', 'variational_n4', 'qft_n4',
                        'iswap_n2', 'bell_n4', 'basis_change_n3', 'vqe_uccsd_n6', 'ising_n10', 'simon_n6', 'qpe_n9',
                        'qaoa_n6', 'bb84_n8', 'vqe_uccsd_n8', 'adder_n10', 'dnn_n8']
    medium_benchmarks = ['bv_n14', 'multiplier_n15', 'sat_n11', 'qft_n18']

    for benchmark in small_benchmarks:
        _benchmark('small', benchmark)
    for benchmark in medium_benchmarks:
        _benchmark('medium', benchmark)


if __name__ == '__main__':
    run_benchmarks()
