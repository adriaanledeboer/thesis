from __future__ import annotations  # <-- must be here (and before other imports)

# Qiskit packages
import qiskit, qiskit_aer#, qiskit_ibm_runtime, qiskit_ibm_catalog

# Own packages
from python_files import Query as qry, RS_prep as rsp, General as grl, Stabilizer_and_Graphs as sg

# Other packages
import numpy as np 
import math
import re

# Qiskit functions
from qiskit import QuantumRegister, ClassicalRegister, QuantumCircuit, transpile
from qiskit.quantum_info import Statevector, Clifford, Pauli, PauliList
from qiskit.circuit import Instruction
from qiskit.circuit.classical import expr
from qiskit_aer import AerSimulator
from qiskit.transpiler.passes import RemoveBarriers
from qiskit.transpiler import CouplingMap

# Other funtions
from typing import Optional
from IPython.display import Math
from numpy.random import default_rng
from collections import deque, Counter


def cswap_gadget(
        qc: QuantumCircuit,
        control_qubit: Qubit,
        target_qubit1: Qubit,
        target_qubit2: Qubit
) -> None:
    """
    Implements CSWAP gadget
    """
    qc.ccx(control_qubit, target_qubit1, target_qubit2)
    qc.cx(target_qubit2, target_qubit1)


def make_U_NOHE_gate(
        n: int
) -> QuantumCircuit:
    """
    Build a reusable circuit implementing U_NOHE for an n-qubit input.

    The circuit acts on `N-1` wires where `N = 2**n`:
      - wires `[0 .. n-1]`  → input qubits
      - wires `[n .. N-2]`  → ancillas (`N-1-n` of them)

    Args:
        n: Integer specifing the number of qubits of the input state
        
    Returns:
        QuantumCircuit: A qc labeled `U_NOHE[n]` that you can append to any circuit.
    """
    
    N = 2**n
    total = N - 1

    sub = QuantumCircuit(total, name=f"U_NOHE[{N}]")
    # indices inside the subcircuit (0..total-1)
    # first n are the input, rest are ancilla's
    
    # SWAPs
    for K in range(n-1, 1, -1):
        sub.swap(K, 2**K - 1)

    # CSWAPs
    for K in range(0, n - 1):                                     # 0 .. n-2
        for J in range(K + 1, n):                                 # K+1 .. n-1
            for alpha in range(2**K - 1, 2 * (2**K - 1) + 1):     # alpha:  (2**K - 1) .. (2*(2**K - 1))
                cswap_gadget(sub, 
                             alpha,
                             alpha + 2**J - 2**K,
                             alpha + 2**J)

    # print(sub)

    return sub


def apply_U_NOHE(
        qc: QuantumCircuit, 
        input_qubits: Sequence[Qubit]
) -> None:
    """
    Append U_NOHE (on the given input qubits) to `qc`, allocating the required
    ancilla qubits automatically.

    Args:
        qc: Circuit to modify (operation is appended).
        input_qubits: Ordered input qubits (length `n`). Ancillas (`2**n - 1 - n`)
            are added to a new register named `"A1"` and used as part of the gate.

    Returns:
        None. The circuit `qc` is modified in place.
    """
    
    n = len(input_qubits)
    N = 2**n
    anc_needed = N - 1 - n

    A1 = QuantumRegister(anc_needed, "A1")
    qc.add_register(A1)
    ancilla_qubits = list(A1[:])

    sub = make_U_NOHE_gate(n)
    targets = list(input_qubits) + ancilla_qubits

    qc.compose(sub, qubits=targets, inplace=True)
    # qc.append(gate, targets)


def make_Phi1(
        n: int
) -> QuantumCircuit:
    """
    Prepares the Phi1 part of the resource state, given the number of qubits in the address register: n.

    Args:
        n: Integer specifing the number of qubits in the address register

    Returns:
        qc: A QuantumCircuit on which Phi1 is prepared. Note: the registers are still labeled as in the input, so the output of U_NOHE is not grouped yet. 
    """

    I   = QuantumRegister(n, 'I')
    K1  = QuantumRegister(n, 'K1')
    bus = QuantumRegister(1, 'bus1')
    qc  = QuantumCircuit(I, K1, bus)
    
    # Make BPs
    grl.apply_make_BPs(qc, I[:] + K1[:]); qc.h(bus)
    
    # Apply U_NOHE to the COMBINED (K1 + bus) inputs
    input_subset = K1[:] + bus[:]
    apply_U_NOHE(qc, input_subset)

    N = 2**n
    I  = qc.qubits[:n]
    D_NOHE = qc.qubits[n:n+N-1]
    L = qc.qubits[n+N-1:]
    qc = grl.repackage_into_registers(qc, layout=[("I", I),
                                                  ("D_NOHE", D_NOHE),
                                                  ("L", L)])

    return qc


def G(
        qc: QuantumCircuit, 
        input_qubits: Sequence[Qubit], 
        anc_qubits: Sequence[Qubit], 
        clbit: Clbit
) -> None:
    """
    Deterministic gadget for C_NOHE.
    - Prepares a Bell-pair-like resource on anc_qubits (via H+CZ)
    - Applies only Clifford two-qubit gates
    - Measures the 3rd input in X basis (H then Z-measure)

    No Pauli corrections are applied here; byproducts are propagated virtually.
    """
    # make two-qubit entangled resource for the gadget
    grl.apply_make_BPs(qc, anc_qubits[:])  # creates a pair across anc_qubits[0] / anc_qubits[1]

    # Clifford entanglers for the gadget
    qc.cx(input_qubits[-1], input_qubits[1])
    qc.cz(input_qubits[0], anc_qubits[0])
    qc.cz(input_qubits[1], anc_qubits[1])


def make_C_NOHE_circuit(
        n: int,
        target_layer: int = None,
        measure: bool = True
) -> Tuple[QuantumCircuit, Dict[int, Tuple[int, int]], Dict[int, Dict[str, int]]]:
    """
    C_NOHE for N=2**n:
      • Lays down all gadget unitaries (Clifford only).
      • Records per-gadget metadata, including its 'layer' (layer = J), Kq0/Kq1, q2, and A2 indices.
      • Then (if measure) does X-basis SQPMs on all q2 wires (H+measure) — this is the first
        (n-1) measurement rounds, one layer at a time (ordering is preserved by the loop).

    Returns:
      sub:       the subcircuit
      mapping:   gid -> (Kq0_idx, Kq1_idx) into K2
      gadgets:   gid -> {"layer", "Kq0", "Kq1", "q2", "Aalpha", "Abeta"}
    """
    N = 2**n
    total_qubits = N - 1 + 2*(N - n - 1)
    total_clbits = 3*(N - n - 1)

    sub = QuantumCircuit(total_qubits, total_clbits, name=f"C_NOHE[{N}]")
    K2 = sub.qubits[:N-1]
    A2 = sub.qubits[N-1:]
    K2_clr = sub.clbits[:N-n-1]           # outcomes of q2 X-SQPMs
    # A2_clr = sub.clbits[N-n-1:]         # outcomes of A2 Z-SQPMs (filled in adaptive phase)

    count = 0
    mapping = {}
    third_wires = []
    info = {}
    start_gates = False
    
    for K in range(n-1, -1, -1):  # descending
        for J in range(n-1, K, -1):
            for alpha in range(2*(2**K-1), 2**K-2, -1):
                q0 = K2[alpha]
                q1 = K2[alpha + 2**J - 2**K]
                q2 = K2[alpha + 2**J]
                qubits = (q0, q1, q2)
                
                A0 = A2[2*count]
                A1 = A2[2*count+1]
                ancillas = (A0, A1)

                if count == 0:
                    layer = 0
                    gadget = 0
                    info.setdefault(layer, {})
                    info[layer][gadget] = {"qubits": qubits, "ancillas": ancillas}
                else:
                    # Check if this gadget shares any qubit with gadgets already in the current layer
                    new_layer = False
                    layer_dict = info.get(layer, {})
                    for _, gadget_dict in layer_dict.items():
                        q_prev = gadget_dict["qubits"]
                        if (q_prev[0] == q0) or (q_prev[1] == q1) or (q_prev[2] == q2):
                            new_layer = True
                            break
                
                    if new_layer:
                        layer += 1
                        gadget = 0
                        info.setdefault(layer, {})
                        info[layer][gadget] = {"qubits": qubits, "ancillas": ancillas}
                    else:
                        gadget += 1
                        info.setdefault(layer, {})  # <-- make sure dict exists
                        info[layer][gadget] = {"qubits": qubits, "ancillas": ancillas}

                # sub.save_statevector(label=f"layer: {layer}, gadget: {gadget}, count: {count}", pershot=True)

                if target_layer is None:
                    start_gates = True
                elif layer is (target_layer + 1): 
                    start_gates = True

                if start_gates:
                    G(
                        sub,
                        qubits,
                        ancillas,
                        K2_clr[count]
                    )

                third_wires.append(q2)
                mapping[count] = (alpha, alpha + 2**J - 2**K)
                
                count += 1

    # After ALL gadget unitaries: X-basis SQPMs of the collected third wires
    i = 0
    if measure:
        for layer, layer_dict in info.items():
            for g, gadget_dict in layer_dict.items(): 
                q2 = gadget_dict["qubits"][2]
                sub.h(q2)
                sub.measure(q2, K2_clr[i])
                gadget_dict["clbit"] = K2_clr[i]
                i += 1

    return sub, mapping, info


def apply_C_NOHE(
        qc: QuantumCircuit,
        input_qubits: Sequence[Qubit],
        measure = True
) -> Tuple[QuantumRegister, ClassicalRegister, Dict[int, Tuple[int,int]], Dict[int, Dict[str, int]], list[int]]:
    """
    Apply the deterministic C_NOHE to the given K2 wires (input_qubits).     
    """
    N = len(input_qubits) + 1
    if N < 1 or (N & (N-1)) != 0:
        raise ValueError("N must be a power of 2")
    n = int(math.log2(N))

    num_qubits_A2 = 2*(N-n-1)
    num_clbits_A2 = num_qubits_A2
    num_clbits_K2 = (N-1-n)

    K2 = input_qubits
    A2 = QuantumRegister(num_qubits_A2, "A2")
    K2_clr = ClassicalRegister(num_clbits_K2, "K2_clr")
    # A2_clr = ClassicalRegister(num_clbits_A2, "A2_clr")
    qc.add_register(A2, K2_clr)#, A2_clr)

    count = 0
    mapping = {}
    third_wires = []
    info = {}
    
    for K in range(n-1, -1, -1):  # descending
        for J in range(n-1, K, -1):
            for alpha in range(2*(2**K-1), 2**K-2, -1):
                q0 = K2[alpha]
                q1 = K2[alpha + 2**J - 2**K]
                q2 = K2[alpha + 2**J]
                qubits = (q0, q1, q2)
                
                A0 = A2[2*count]
                A1 = A2[2*count+1]
                ancillas = (A0, A1)

                G(
                    qc,
                    qubits,
                    ancillas,
                    K2_clr[count]
                )

                third_wires.append(q2)
                mapping[count] = (alpha, alpha + 2**J - 2**K)

                if count == 0:
                    layer = 0
                    gadget = 0
                    info.setdefault(layer, {})
                    info[layer][gadget] = {"qubits": qubits, "ancillas": ancillas}
                else:
                    # Check if this gadget shares any qubit with gadgets already in the current layer
                    new_layer = False
                    layer_dict = info.get(layer, {})
                    for _, gadget_dict in layer_dict.items():
                        q_prev = gadget_dict["qubits"]
                        if (q_prev[0] == q0) or (q_prev[1] == q1) or (q_prev[2] == q2):
                            new_layer = True
                            break
                
                    if new_layer:
                        layer += 1
                        gadget = 0
                        info.setdefault(layer, {})
                        info[layer][gadget] = {"qubits": qubits, "ancillas": ancillas}
                    else:
                        gadget += 1
                        info.setdefault(layer, {})  # <-- make sure dict exists
                        info[layer][gadget] = {"qubits": qubits, "ancillas": ancillas}

                # qc.save_statevector(label=f"layer: {layer}, gadget: {gadget}, count: {count}", pershot=True)
                
                count += 1

    # After ALL gadget unitaries: X-basis SQPMs of the collected third wires
    i = 0
    if measure:
        for layer, layer_dict in info.items():
            for g, gadget_dict in layer_dict.items(): 
                q2 = gadget_dict["qubits"][2]
                qc.h(q2)
                qc.measure(q2, K2_clr[i])
                gadget_dict["clbit"] = K2_clr[i]
                i += 1
    
    if qc.metadata is None:
        qc.metadata = {}
    qc.metadata[f"C_NOHE_mapping[{N}]"] = mapping
    qc.metadata[f"C_NOHE_info[{N}]"] = info
    qc.metadata[f"C_NOHE_third_wires[{N}]"] = third_wires

    return A2, K2_clr, mapping, info, third_wires


def make_Phi2(
        n: int
) -> Tuple[QuantumCircuit, QuantumRegister, QuantumRegister, ClassicalRegister, Dict[int, Tuple[int,int]], Dict[int, Dict[str, int]], list[int]]:
    """
    Makes Phi 2 for a query with n input qubits.
    """
    N = 2**n
    Dprime = QuantumRegister(2*N-1, "Dprime")
    K2 = QuantumRegister(2*N-1, "K2")
    qc = QuantumCircuit(Dprime, K2)
    
    grl.apply_make_BPs(qc, Dprime[:] + K2[:])
    A2, K2_clr, mapping, info, third_wires = apply_C_NOHE(qc, input_qubits=K2[:])

    return qc, K2, A2, K2_clr, mapping, info, third_wires


def merge_phi1_phi2_normal(
        qc1: QuantumCircuit, 
        qc2: QuantumCircuit, 
        n: int, 
        save_svs: bool =False
) -> QuantumCircuit:
    """
    Merges phi1 and phi2 quantumcircuits for n query qubits on a new qc. 
    Optionally save intermediate statevectors
    """

    def outcome_string_to_Pin(
            outcome: list[int], 
            n: int
    ) -> Pauli:
        """
        Converts D_NOHE and D' measurement outcome to an equivalent Pauli (P_in) applied to K2.
        D_NOHE contributes Z's, while D' contributes X's.

        Assumes the case for n address qubits.

        Returns P_in
        """
        N = 2**n
        num_A2_qubits = 2*(2*N-n-2)
    
        # separate d_nohe values from dpime values
        d_nohe_outcomes = outcome[:N-1]
        dprime_outcomes = outcome[N-1:]
    
        # make P_in
        z = np.array(d_nohe_outcomes + N*[0] + num_A2_qubits*[0], dtype=bool)
        x = np.array(dprime_outcomes + num_A2_qubits*[0], dtype=bool)  
        P_in = Pauli((z, x))
    
        return P_in
    
    def precompute_merge_corrections(
            n: int
    ) -> dict[int, Pauli]:
        """
        Assumes n address qubits in the query. Precomputes the Pauli corrections on K2 and A2 (P_out)
        corresponding to all possible D_NOHE and D' measurement outcomes.

        Returns a dict where a key is a full D_NOHE and D' outcome (in base 10 form) and the values are 
        the P_out's.
        """
        N = 2**n
    
        num_d_nohe_qubits = N-1
        num_dprime_qubits = 2*N-1
        num_qubits        = num_d_nohe_qubits + num_dprime_qubits
        
        all_outcomes = grl.all_binary_lists(num_qubits)
    
        C = Clifford(rsp.make_C_NOHE_circuit(n+1 , measure=False)[0])  # note n+1 instead of n
        d = {}
    
        for outcome in all_outcomes:
            P_in = outcome_string_to_Pin(outcome, n)
            P_out = P_in.evolve(C, frame="s")
            d[grl.bits_to_int(outcome)] = P_out
    
        return d
        
    # tensor phi1 and phi2
    qc = qc2.tensor(qc1)
    qc.metadata = qc2.metadata | qc1.metadata

    N = 2**n
    third_wires = qc.metadata[f"C_NOHE_third_wires[{2*N}]"]
    
    D_NOHE = grl.get_qreg(qc, "D_NOHE")
    L      = grl.get_qreg(qc, "L")
    Dprime = grl.get_qreg(qc, "Dprime")
    K2     = grl.get_qreg(qc, "K2")
    A2     = grl.get_qreg(qc, "A2")
    K2_clr = grl.get_creg(qc, "K2_clr")

    if save_svs:
        qc.save_statevector(label="start merge", pershot=True)
    
    # apply CZs between D_NOHE + L and Dprime
    qc.cz(D_NOHE[:] + L[:], Dprime[:])
    
    # add cregs for D_NOHE and Dprime, then measure in X basis
    merge_clr = ClassicalRegister(3*N-2, "merge_clr")
    qc.add_register(merge_clr)
    
    qc.h(D_NOHE[:] + Dprime[:])
    qc.measure(D_NOHE[:] + Dprime[:], merge_clr[:])

    if save_svs:
        qc.save_statevector(label="partial BMs done", pershot=True)

    # precompute pauli corrections
    P_out_dict = precompute_merge_corrections(n)

    # rotate q2's in K2 back to Z frame before applying corrections
    for q2 in third_wires:
        qc.h(q2)

    # apply pauli corrections
    for outcome, P_out in P_out_dict.items():
        with qc.if_test((merge_clr, outcome)):
            Zarr, Xarr = P_out.z, P_out.x
            j = 0
            for q in (K2[:] + A2[:]):
                if Zarr[j]:
                    qc.z(q)
                if Xarr[j]:
                    qc.x(q)
                j += 1

    # measure q2's in K2 in X basis
    j = 0
    for q2 in third_wires:
        qc.h(q2)
        qc.measure(q2, K2_clr[j])
        j += 1

    if save_svs:
        qc.save_statevector(label="merging pauli corr done", pershot=True)

    return qc


def make_phi_normal(
        n: int, 
        save_svs: bool = False
) -> QuantumCircuit:
    """
    Makes a qc holding the resource state phi for the case of n address qubits. 
    Optionally save intermediate statevectors
    """
    
    qc1 = make_Phi1(n)
    if save_svs:
        qc1.save_statevector(label="phi1 done", pershot=True)
        
    qc2, K2, A2, K2_clr, mapping, info, third_wires = make_Phi2(n)
    if save_svs:
        qc2.save_statevector(label="phi2 done", pershot=True)
    
    qc = merge_phi1_phi2_normal(qc1, qc2, n, save_svs=save_svs)
    if save_svs:
        qc.save_statevector(label="merge done", pershot=True)

    return qc
