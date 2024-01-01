# Qiskit ZX Transpiler

A transpiler pass for Qiskit which uses ZX-Calculus for circuit optimization, implemented using PyZX.

## Example usage

```python
from qiskit import QuantumCircuit
from qiskit.transpiler import PassManager
from zxpass import ZXPass

# Create a qiskit `QuantumCircuit` here...
qc = QuantumCircuit(...)

zxpass = ZXPass()
pass_manager = PassManager(zxpass)
zx_qc = pass_manager.run(qc)
```

It is also possible to initialise `ZXPass` with a custom optimization function.
(The default, if none is supplied, is to call `pyzx.simplify.full_reduce`
on the graph of the circuit.)

```python
import pyzx

def my_optimize(c: pyzx.Circuit) -> pyzx.Circuit:
    g = c.to_graph()
    # do stuff to simplify `g`...
    return pyzx.extract.extract_circuit(g)

zxpass = ZXPass(my_optimize)
pass_manager = PassManager(zxpass)
my_qc = pass_manager.run(qc)
```
## Running benchmarks

To perform some benchmarks based on the [QASMBench](https://github.com/pnnl/QASMBench) suite, run the following:

```bash
cd benchmarking
python run_benchmarks.py
```

This will output some statistics and produce 2 PNG files showing the depth compression ratio between both Qiskit- and ZX-optimized circuits and the original circuits, and ratio of non-local gates beween the Qiskit- and ZX-optimized circuits.

## Previous work

There have been two previous attempts to create a transpiler pass for Qiskit using PyZX which I'm aware of.

The first attempt was made in 2019 by
[@lia-approves](https://github.com/lia-approves), [@edasgupta](https://github.com/edasgupta), and [@ewinston](https://github.com/ewinston)
when they were interns at IBM Quantum, as documented in [this Qiskit issue](https://github.com/Qiskit/qiskit/issues/4990).
That code used qasm as an intermediate format when converting between a Qiskit `DAGCircuit` and a PyZX `Circuit`,
which is undesirable for reasons noted in that issue. Furthermore, the code is out of date with subsequent changes made to both Qiskit and PyZX.

The second attempt was made by [@gprs1809](https://github.com/gprs1809) et al. as a part of the Qiskit Advocate Mentorship Program (QAMP) in the fall of 2022.
The code is found in [this repository](https://github.com/gprs1809/ZX_to_DAG_QAMP_fall_2022).
This implementation converts a PyZX `Circuit` directly to a Qiskit `QuantumCircuit`, without going through qasm.
However, the code is incomplete and produces the wrong output for some circuits
(which may have been due to a difference between how PyZX and Qiskit implements certain gates, fixed in [this PR](https://github.com/Quantomatic/pyzx/pull/156)),
and (as far as I can tell) the code to convert in the other direction is not available.

