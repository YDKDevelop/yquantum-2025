import math
import numpy as np
import matplotlib.pyplot as plt
import random
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector
from qiskit.quantum_info.operators import Pauli
from qiskit.visualization import plot_bloch_multivector

# Constants
NUM_QUBITS = 16
NUM_LAYERS = 4
TOTAL_BITS = 16            # Number of bits per expectation value
FRACTION_BITS = 15         # Precision within each fixed-point

def toFixed(x: float) -> int:
    """Convert float to fixed-point integer."""
    fraction_mult = 1 << FRACTION_BITS
    return int(x * fraction_mult + (0.5 if x >= 0 else -0.5))

def bytes_to_angles(input_bytes: bytes) -> list[float]:
    """Convert 256-bit input (32 bytes) into 2π-scaled rotation angles."""
    assert len(input_bytes) == 32, "Input must be 256 bits (32 bytes)"
    return [(b / 255.0) * 2 * math.pi for b in input_bytes]

def build_quantum_hash_circuit(angles: list[float]) -> QuantumCircuit:
    """Build a 16-qubit, 4-layer PQC circuit using parameterized rotations."""
    qc = QuantumCircuit(NUM_QUBITS)
    
    angle_chunks = np.array_split(angles, NUM_LAYERS)

    for layer_idx, chunk in enumerate(angle_chunks):
        for i in range(NUM_QUBITS):
            angle = chunk[i % len(chunk)]
            if layer_idx % 2 == 0:
                qc.rx(angle, i)
            else:
                qc.ry(angle, i)

        # Entanglement pattern (nearest neighbor + CZ cross links)
        for i in range(NUM_QUBITS - 1):
            qc.cx(i, i + 1)
        if layer_idx % 2 == 0:
            for i in range(0, NUM_QUBITS - 2, 2):
                qc.cz(i, i + 2)

    return qc

def quantum_hash(input_bytes: bytes) -> bytes:
    """Compute a 256-bit quantum hash of 32-byte input."""
    angles = bytes_to_angles(input_bytes)
    qc = build_quantum_hash_circuit(angles)
    sv = Statevector.from_instruction(qc)

    # Measure expectation values (Z-basis)
    exps = [sv.expectation_value(Pauli("Z"), [i]).real for i in range(NUM_QUBITS)]
    fixed_exps = [toFixed(exp) for exp in exps]

    # Convert to 256-bit hash (16 values × 16 bits = 256 bits)
    output_bytes = []
    for fixed in fixed_exps:
        for j in range(TOTAL_BITS // 8):
            output_bytes.append((fixed >> (8 * j)) & 0xFF)

    return bytes(output_bytes)


# ------------------------------
# Generate a distribution from many random inputs
# ------------------------------

num_samples = 100  # number of random inputs to generate
all_fixed_values = []  # collect each of the 16 fixed-point values for each hash

for _ in range(num_samples):
    # Generate a random 32-byte input.
    input_bytes = bytes(random.randint(0, 255) for _ in range(32))
    # Compute hash (we only need the fixed_exps; you can also call quantum_hash() if desired)
    angles = bytes_to_angles(input_bytes)
    qc = build_quantum_hash_circuit(angles)
    sv = Statevector.from_instruction(qc)
    exps = [sv.expectation_value(Pauli("Z"), [i]).real for i in range(NUM_QUBITS)]
    fixed_exps = [toFixed(exp) for exp in exps]
    all_fixed_values.extend(fixed_exps)

# Plotting the histogram:
plt.figure(figsize=(10,6))
plt.hist(all_fixed_values, bins=50, edgecolor='black')
plt.xlabel("Fixed-point value (per qubit expectation)")
plt.ylabel("Frequency")
plt.title("Distribution of Fixed-point Values from Quantum Hash Outputs\n(Collected over {} random inputs)".format(num_samples))
plt.show()
