from __future__ import annotations  # <-- must be here (and before other imports)

# Qiskit packages
import qiskit, qiskit_aer#, qiskit_ibm_runtime, qiskit_ibm_catalog

# Own packages
import Query as qry
import RS_prep as rsp
import General as grl

# Other packages
import numpy as np 
import math

# Qiskit functions
from qiskit import QuantumRegister, ClassicalRegister, QuantumCircuit
from qiskit.quantum_info import Statevector, Clifford, Pauli, PauliList
from qiskit.circuit.classical import expr

# Other funtions
from typing import Optional


def initialize_adresses(n: int, 
                        register_name: Optional[str] = 'addr_reg', 
                        method: Optional[str] = 'Equal superposition', 
                        seed: Optional[int] = None, 
                        vec: Optional[np.ndarray] = None
) -> QuantumCircuit:
    """
    Initializes address state

    Args:
        n: Number of qubits in address
        register_name: Name of address register
        method: Method of determining probability amplitudes in address superposition. Can be either 'Equal superposition', 'Random' or 'Supply vector'
        seed: Seed in case of method = 'Random'
        vec: np.ndarray containing probability amplitudes in case of method = 'Supply vector'

    Returns:
        qc: QuantumCircuit containing initialized address state
    """

    addr_reg = QuantumRegister(n, register_name)
    qc = QuantumCircuit(addr_reg, name='addr initialization circuit')
    amps = np.zeros(2**n, dtype=complex)
    
    if method == 'Equal superposition':
        amps[:] = 1/np.sqrt(2**n)
    elif method == 'Random':
        rng = np.random.default_rng(seed) 
        amps = rng.random(2**n) + 1j * rng.random(2**n)
        amps = amps / np.linalg.norm(amps)
    elif method == 'Supply vector':
        if len(vec) != 2**n:
                raise ValueError("Supply vector of length 2**n.")
        amps = vec / np.linalg.norm(vec)
    else:
        raise ValueError("Method must be 'Equal superposition', 'Random' or 'Supply vector'.")

    qc.initialize(amps, addr_reg)
    addr_sv = Statevector(qc)
    
    return qc


def make_mem_bits(
    n: int,
    seed: Optional[int] = None
) -> nd.array[int]:
    """
    Creates array of length N = n**2 of randomly determined (memory) bits

    Args:
        n: Determines length of bit array: N = 2**n

    Returns:
        mem_bits: np.ndarray of length N = 2**n, containing randomly determined (memory) bits
    """
    
    np.random.seed(seed)   
    mem_bits = np.random.randint(0, 2, size=2**n, dtype=np.uint8)

    return mem_bits
    
    
def make_U_QRAM_gate(
    mem_bits: np.ndarray[int]
) -> Gate:
    """
    Build a gate that implements U_QRAM for a memory of length N=2**n.

    The returned gate acts on n+1 wires ordered as [bus | address_0..address_{n-1}].
    So note: THE BUS IS THE LSB!
    For each address x with mem_bits[x] == 1, it applies an MCX on the bus
    conditioned on the address being x.

    Args:
        mem_bits: 1D array of 0/1 values, length N (ideally a power of two).

    Returns:
        Gate: A gate labeled 'U_QRAM[N]' acting on n+1 qubits.

    Raises:
        ValueError: If any entry of mem_bits is not 0 or 1.
    """
    
    N = len(mem_bits)
    n = int(math.log2(N))

    sub = QuantumCircuit(n+1, name=f'U_QRAM[{N}]')
    controls = [sub.qubits[j] for j in range(1, n+1)]

    for x, bit in enumerate(mem_bits):
        if bit not in (0, 1):
            raise ValueError("mem_bits must contain only 0 or 1.")
        if bit == 1:
            ctrl_state = format(x, f"0{n}b")
            # print(f'control state: {ctrl_state}, controls: {controls}, bit: {bit}')
            sub.mcx(controls, sub.qubits[0], ctrl_state=ctrl_state)

    return sub.to_gate(label=sub.name)


def apply_U_QRAM(
    qc: QuantumCircuit,
    input_qubits: Sequence[Qubit],
    mem_bits: np.ndarray[int]
) -> None:
    """
    Append U_QRAM(mem_bits) to `qc`, adding a 1-qubit 'bus' and repackaging
    wires so the order is [bus | input_qubits]. Modifies `qc` in place.

    Args:
        qc: Circuit to modify.
        input_qubits: Address qubits (length n), in the desired address order.
        mem_bits: Memory bits; length must be 2**n.

    Returns:
        None
    """

    n = len(input_qubits)

    bus = QuantumRegister(1, "bus")
    qc.add_register(bus)

    addr_reg = qc.qubits[:-1]
    bus = [bus[0]]

    qc2 = grl.repackage_into_registers(qc, layout=[("bus", bus),
                                                  ("addr_reg", addr_reg)])

    # grl.print_qubit_table(qc)
    gate = make_U_QRAM_gate(mem_bits)
    targets = qc2.qubits

    qc2.append(gate, targets)
    
    qc.__dict__.update(qc2.__dict__) # Mutate the original qc to become qc2


def query_step_i(
    qc: QuantumCircuit, 
    n: int,
    save_svs=False
) -> None:
    """
    Applies step (i) of the QRAM query to qc, by performing BMs between QPU address 
    register R and RS register I. Outcomes are stored in R_clr and I_clr respectively.

    Args:
        qc: QuantumCircuit holding |ψ> ⊗ |Φ>.
        n: Size of address register (R)

    Returns:
        None
    """
    
    N = 2**n
    R = grl.get_qreg(qc, "R")
    I = grl.get_qreg(qc, "I")
    R_clr = ClassicalRegister(n, "R_clr")
    I_clr = ClassicalRegister(n, "I_clr")
    L_clr = ClassicalRegister(N, "L_clr")
    qc.add_register(R_clr, I_clr, L_clr)
    
    grl.apply_BMs(qc, R[:] + I[:], R_clr[:] + I_clr[:])

    if save_svs:
        qc.save_statevector(label="step i done", pershot=True)


def query_step_ii(
    qc: QuantumCircuit, 
    L: QuantumRegister, 
    I_clr: ClassicalRegister, 
    mem_bits: Sequence[int],
    save_svs = False
) -> None:
    """
    Applies step (ii) of the QRAM query to qc, i.e. data loading. Does so by applying 
    Z gates on qubits of L conditioned on the qubit index, the measurement outcomes (b) 
    of step (i), stored in I_clr, and the memory bits, stored in mem_bits. The function 
    works as follows:
    
    For each qubit l in L, compute x = l XOR b (where b is the integer value
    of classical register I_clr). If mem_bits[x] == 1, apply Z to L[l].
    
    Args:
        qc: QuantumCircuit after step (i) of query.
        L: QuantumRegister containing qubits to load data onto
        I_clr: ClassicalRegister holding measurement outcomes, b, from step (i)
        mem_bits: List of integers representing memory bits of QRAM
        
    Returns:
        None
    """
    n = len(I_clr)
    N = 1 << n
    if len(L) != N:
        raise ValueError(f"len(L) must be 2**len(I_clr) = {N}, got {len(L)}")
    if len(mem_bits) != N:
        raise ValueError(f"len(mem_bits) must be {N}, got {len(mem_bits)}")
        
    # Precompute, for each possible b = v, which qubits need a Z:
    #   need_z[v] = { l | mem_bits[l ^ v] == 1 }
    need_z = [[] for _ in range(N)]
    for v in range(N):
        mask = []
        for l in range(N):
            if mem_bits[l ^ v] == 1:
                mask.append(l)
        need_z[v] = mask
    # print(need_z)

    # Add conditionally-controlled Z gates for each possible value of b
    for v, indices in enumerate(need_z):
        if not indices:
            continue
        with qc.if_test((I_clr, v)):
            for l in indices:
                qc.z(L[l])

    if save_svs:
        qc.save_statevector(label="step ii done", pershot=True)


def outcome_string_to_Pin_step_iii(outcome, n):
    """ """
    N = 2**n
    num_qubits = 2*N-1 + 2*(2*N-n-2)

    x = np.zeros(num_qubits, dtype=bool)
    z = np.array([0]*(N-1) + outcome + [0]*2*(2*N-n-2), dtype=bool)
    P_in = Pauli((z, x))

    return P_in

def precompute_step_iii_corrections(n):
    """ """
    N = 2**n
    
    all_outcomes = grl.all_binary_lists(N)

    C = Clifford(rsp.make_C_NOHE_circuit(n+1 , measure=False)[0])  # note n+1 instead of n
    d = {}

    for outcome in all_outcomes:
        P_in = outcome_string_to_Pin_step_iii(outcome, n)
        P_out = P_in.evolve(C, frame="s")
        d[grl.bits_to_int(outcome)] = P_out

    return d
    

def query_step_iii(qc, n, save_svs=False):
    """ """
    def outcome_string_to_Pin_step_iii(outcome, n):
        """ """
        N = 2**n
        num_qubits = 2*N-1 + 2*(2*N-n-2)
    
        x = np.zeros(num_qubits, dtype=bool)
        z = np.array([0]*(N-1) + outcome + [0]*2*(2*N-n-2), dtype=bool)
        P_in = Pauli((z, x))
    
        return P_in
    
    def precompute_step_iii_corrections(n):
        """ """
        N = 2**n
        
        all_outcomes = grl.all_binary_lists(N)
    
        C = Clifford(rsp.make_C_NOHE_circuit(n+1 , measure=False)[0])  # note n+1 instead of n
        d = {}
    
        for outcome in all_outcomes:
            P_in = outcome_string_to_Pin_step_iii(outcome, n)
            P_out = P_in.evolve(C, frame="s")
            d[grl.bits_to_int(outcome)] = P_out
    
        return d
        
    N = 2**n
    third_wires = qc.metadata[f"C_NOHE_third_wires[{2*N}]"]

    L      = grl.get_qreg(qc, "L")
    K2     = grl.get_qreg(qc, "K2")
    A2     = grl.get_qreg(qc, "A2")
    R_clr  = grl.get_creg(qc, "R_clr")
    I_clr  = grl.get_creg(qc, "I_clr")
    L_clr  = grl.get_creg(qc, "L_clr")
    K2_clr = grl.get_creg(qc, "K2_clr")

    # measure L in X basis
    qc.h(L)
    qc.measure(L, L_clr)

    if save_svs:
        qc.save_statevector(label="meas L in X done", pershot=True)
        
    # precompute pauli corrections
    P_out_dict = precompute_step_iii_corrections(n)

    # rotate q2's in K2 back to Z frame before applying corrections
    for q2 in third_wires:
        qc.h(q2)

    if save_svs:
        qc.save_statevector(label="rotate third wires to Z done", pershot=True)
        
    # apply pauli corrections
    for outcome, P_out in P_out_dict.items():
        with qc.if_test((L_clr, outcome)):
            Zarr, Xarr = P_out.z, P_out.x
            j = 0
            for q in (K2[:] + A2[:]):
                if Zarr[j]:
                    qc.z(q)
                if Xarr[j]:
                    qc.x(q)
                j += 1

    if save_svs:
        qc.save_statevector(label="step iii pauli applied", pershot=True)

    # measure q2's in K2 in X basis
    j = 0
    for q2 in third_wires:
        qc.h(q2)
        qc.measure(q2, K2_clr[j])
        j += 1

    if save_svs:
        qc.save_statevector(label="meas third wires in X done", pershot=True)

    # apply adaptive U_NOHE inversion part
    grl.apply_adaptive_part_U_NOHE_inversion(qc, K2[:] + A2[:], K2_clr[:], n+1, save_svs=save_svs)
    
    if save_svs:
        qc.save_statevector(label="adaptive part done", pershot=True)

    # apply inverse swaps
    grl.apply_inverse_swaps(qc, K2[:])

    if save_svs:
        qc.save_statevector(label="inverse swaps done", pershot=True)
    
    # apply pauli corrections from step i
    for j in range(n):
        Rc  = R_clr[j]
        Ic  = I_clr[j]
        K2q = K2[j]
        with qc.if_test((Rc, 1)):
            qc.z(K2q)
        with qc.if_test((Ic, 1)):
            qc.x(K2q)

    if save_svs:
        qc.save_statevector(label="pauli corr from step i done", pershot=True)

    # H on bus
    bus = K2[n]
    qc.h(bus)

    if save_svs:
        qc.save_statevector(label="h on bus done", pershot=True)
        

def QRAM_query(qc_addr, qc_RS, n, mem_bits, save_svs=False, meas_output=False):
    """ """
    # combine address and resource states / circuits
    qc = qc_RS.tensor(qc_addr)
    qc.metadata = qc_RS.metadata | qc_addr.metadata

    # step i
    qry.query_step_i(qc, n, save_svs=save_svs)

    # step ii
    L = grl.get_qreg(qc, "L")
    I_clr = grl.get_creg(qc, "I_clr")
    qry.query_step_ii(qc, L, I_clr, mem_bits, save_svs=save_svs)

    # step iii
    qry.query_step_iii(qc, n, save_svs=save_svs)

    if meas_output:
        output_clr = ClassicalRegister(n+1, "output_clr")
        qc.add_register(output_clr)
        K2 = grl.get_qreg(qc, "K2")
        qc.measure(K2[:n+1], output_clr[:])

    return qc
