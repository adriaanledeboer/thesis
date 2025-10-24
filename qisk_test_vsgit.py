from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
import numpy as np

def main(shots=10):
    # 2 qubits, 2 classical bits
    qc = QuantumCircuit(2, 2)
    qc.h([0, 1])          # Hadamard on both qubits
    qc.measure([0, 1], [0, 1])  # measure both qubits to both classical bits

    sim = AerSimulator(method='statevector')
    job = sim.run(qc, shots=shots, memory=True)
    res = job.result()

    print("Overall success:", res.success)
    print("Overall status:", res.to_dict().get("status", ""))

    counts = res.get_counts()
    mem = res.get_memory()
    print("Counts:", counts)
    print("First few memory samples:", mem[: min(5, len(mem))])

if __name__ == "__main__":
    main()

