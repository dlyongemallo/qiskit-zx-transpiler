# Qiskit ZX Transpiler

A transpiler pass for Qiskit which uses ZX-Calculus for circuit optimization, implemented using pyzx.

Example usage:

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
