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

from qiskit import transpile
from qiskit.circuit import QuantumCircuit
from qiskit.transpiler import PassManager
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt  # type: ignore

from zxpass import ZXPass


pass_manager = PassManager(ZXPass())


def _benchmark(subdir: str, circuit_name: str, as_plugin: bool = False) -> Tuple[float, float, float]:
    print(f"Circuit name: {circuit_name}")

    qc = QuantumCircuit.from_qasm_file(f"QASMBench/{subdir}/{circuit_name}/{circuit_name}.qasm")
    opt_qc = transpile(qc, basis_gates=['u3', 'cx'], optimization_level=3)
    if as_plugin:
        zx_qc = transpile(qc, optimization_method="zxpass", optimization_level=3)
    else:
        zx_qc = pass_manager.run(qc)

    print(f"Size - original: {qc.size()}, "
          f"optimized: {opt_qc.size()} ({qc.size() / opt_qc.size():.2f}), "
          f"zx: {zx_qc.size()} ({qc.size() / zx_qc.size():.2f})")
    print(f"Depth - original: {qc.depth()}, "
          f"optimized: {opt_qc.depth()} ({qc.depth() / opt_qc.depth():.2f}), "
          f"zx: {zx_qc.depth()} ({qc.depth() / zx_qc.depth():.2f})")
    print(f"Number of non-local gates - original: {qc.num_nonlocal_gates()}, ", end="")
    if qc.num_nonlocal_gates() != 0:
        print(f"optimized: {opt_qc.num_nonlocal_gates()}, zx: {zx_qc.num_nonlocal_gates()}, "
              f"ratio: {opt_qc.num_nonlocal_gates() / zx_qc.num_nonlocal_gates():.2f}")
    else:
        print("optimized: 0, zx: 0")
    print()

    return (qc.depth() / opt_qc.depth(),
            qc.depth() / zx_qc.depth(),
            qc.num_nonlocal_gates() / zx_qc.num_nonlocal_gates() if zx_qc.num_nonlocal_gates() != 0 else 0)


def _save_plot(title: str, plot_index: List[str], data: Dict[str, List[float]], ylabel: str) -> None:
    width = 0.35
    fig, ax = plt.subplots()
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    x = range(len(plot_index))
    ax.bar(x, data['qiskit'], width, label='qiskit')
    ax.bar([i + width for i in x], data['pyzx'], width, label='pyzx')
    ax.set_xticks([i + width / 2 for i in x])
    ax.set_xticklabels(plot_index, rotation=90)
    ax.legend()
    plt.tight_layout()
    plt.savefig(f"{title.replace(' ', '_')}.png")


def run_benchmarks() -> None:
    # List of circuits to benchmark, based on: https://github.com/Qiskit/qiskit/issues/4990#issuecomment-1157858632
    small_benchmarks = ['wstate_n3', 'linearsolver_n3', 'fredkin_n3', 'dnn_n2', 'qrng_n4', 'adder_n4', 'deutsch_n2',
                        'cat_state_n4', 'basis_trotter_n4', 'qec_en_n5', 'toffoli_n3', 'grover_n2', 'hs4_n4', 'qaoa_n3',
                        'teleportation_n3', 'lpn_n5', 'vqe_uccsd_n4', 'quantumwalks_n2', 'variational_n4', 'qft_n4',
                        'iswap_n2', 'bell_n4', 'basis_change_n3', 'vqe_uccsd_n6', 'ising_n10', 'simon_n6', 'qpe_n9',
                        'qaoa_n6', 'bb84_n8', 'vqe_uccsd_n8', 'adder_n10', 'dnn_n8']
    medium_benchmarks = ['bv_n14', 'multiplier_n15', 'sat_n11', 'qft_n18']

    depth_ratio: Dict[str, List[float]] = {'qiskit': [], 'pyzx': []}
    num_nonlocal_ratio: Dict[str, List[float]] = {'qiskit': [], 'pyzx': []}
    plot_index = []
    for benchmark in small_benchmarks:
        qiskit_depth, zx_depth, non_local_ratio = _benchmark('small', benchmark)
        depth_ratio['pyzx'].append(zx_depth)
        depth_ratio['qiskit'].append(qiskit_depth)
        num_nonlocal_ratio['pyzx'].append(non_local_ratio)
        num_nonlocal_ratio['qiskit'].append(1 if non_local_ratio != 0 else 0)
        plot_index.append(benchmark)
    for benchmark in medium_benchmarks:
        qiskit_depth, zx_depth, non_local_ratio = _benchmark('medium', benchmark)
        depth_ratio['pyzx'].append(zx_depth)
        depth_ratio['qiskit'].append(qiskit_depth)
        num_nonlocal_ratio['pyzx'].append(non_local_ratio)
        num_nonlocal_ratio['qiskit'].append(1 if non_local_ratio != 0 else 0)
        plot_index.append(benchmark)

    _save_plot('Depth compression ratio', plot_index, depth_ratio, 'depth_ratio')
    _save_plot('Ratio of non-local gates', plot_index, num_nonlocal_ratio, 'num_nonlocal_ratio')


if __name__ == '__main__':
    run_benchmarks()
