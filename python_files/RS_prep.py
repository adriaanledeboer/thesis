from __future__ import annotations  # <-- must be here (and before other imports)

# Qiskit packages
import qiskit, qiskit_aer#, qiskit_ibm_runtime, qiskit_ibm_catalog

# Own packages
import Query as qry
import RS_prep as rsp
import General as grl
import Stabilizer_and_Graphs as sg

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
) -> Gate:
    """
    Build a reusable gate implementing U_NOHE for an n-qubit input.

    The gate acts on `N-1` wires where `N = 2**n`:
      - wires `[0 .. n-1]`  → input qubits
      - wires `[n .. N-2]`  → ancillas (`N-1-n` of them)

    Args:
        n: Integer specifing the number of qubits of the input state
        
    Returns:
        Gate: A gate labeled `U_NOHE[n]` that you can append to any circuit.
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

    return sub#.to_gate(label=sub.name)


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
    measure: Optional[bool] = True
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
) -> Tuple[QuantumRegister, ClassicalRegister, ClassicalRegister, Dict[int, Tuple[int,int]]]:
    """
    Apply the deterministic C_NOHE to the given K2 wires (input_qubits). 
    Returns A2 (ancillas), K2_clr (X-SQPM outcomes from G), A2_clr (placeholders for adaptive phase), and 'mapping'.
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

    return A2, K2_clr, mapping, info, third_wires#, A2_clr


def make_Phi2(
    n: int
) -> QuantumCircuit:
    """
    """
    N = 2**n
    Dprime = QuantumRegister(2*N-1, "Dprime")
    K2 = QuantumRegister(2*N-1, "K2")
    qc = QuantumCircuit(Dprime, K2)
    
    grl.apply_make_BPs(qc, Dprime[:] + K2[:])
    A2, K2_clr, mapping, info, third_wires = apply_C_NOHE(qc, input_qubits=K2[:])

    return qc, K2, A2, K2_clr, mapping, info, third_wires


def translate_pauli_off_Dprime(
    P: Pauli, 
    S_list: PauliList, 
    Dprime_local
) -> Pauli:
    """
    Translate a Pauli off the Dprime register modulo a stabilizer group.
    
    Given a Pauli `P` and stabilizer generators `S_list`, find an equivalent Pauli
    `Q ≡ P (mod ⟨S_list⟩)` with no X/Z support on indices `Dprime_local`. Phases are ignored;
    a GF(2) linear system is solved to cancel Dprime support.
    
    Args:
        P: Pauli to be translated (Qiskit `Pauli`, little-endian).
        S_list: Stabilizer generators as a `PauliList` (little-endian).
        Dprime_local: Iterable of local qubit indices comprising Dprime.
    
    Returns:
        Q: A Pauli with zero support on Dprime (same length as `P`).
    
    Raises:
        ValueError: If no combination of stabilizers can remove support on Dprime.
    """
    
    n = P.num_qubits
    D = np.array(sorted(Dprime_local), dtype=int)
    Xs = S_list.x[:, D]  # (m, |D|)
    Zs = S_list.z[:, D]  # (m, |D|)
    M = np.vstack([Xs.T, Zs.T])  # (2|D|, m)
    b = np.hstack([P.x[D], P.z[D]]) % 2
    r = sg.gf2_solve(M, b)
    if r is None:
        raise ValueError("Cannot remove support on Dprime for this Pauli with given stabilizers.")
    qx = P.x.copy(); qz = P.z.copy()
    for j, bit in enumerate(r):
        if bit:
            qx ^= S_list.x[j]
            qz ^= S_list.z[j]
    Q = Pauli((qz, qx))
    assert not (Q.x[D].any() or Q.z[D].any())
    
    return Q


def precompute_responses(
    stab_strings: List[str], 
    Dprime_local: List[int], 
    K2_local: List[int], 
    A2_local: List[int], 
    big_endian: bool):
    """
    Precompute Pauli response patterns for feed-forward, removing Dprime support.
    
    Normalizes input stabilizer strings over Dprime|K2|A2 into Qiskit little-endian,
    builds `PauliList` S_list, and for each Dprime_i returns Paulis `Qz`, `Qx` such that:
    - `Qz ≡ Z_i (mod S)` and has no support on Dprime
    - `Qx ≡ X_i (mod S)` and has no support on Dprime
    
    Args:
        stab_strings: Iterable[str] of Pauli strings over Dprime|K2|A2.
                      If `big_endian=True`, leftmost char is local qubit 0; otherwise
                      rightmost char is local qubit 0 (Qiskit format).
        Dprime_local: Local indices of Dprime.
        K2_local: Local indices of K2 (unused except for sanity sizing).
        A2_local: Local indices of A2 (unused except for sanity sizing).
        big_endian: Whether `stab_strings` are in big-endian local order.
    
    Returns:
        Tuple[PauliList, List[Pauli], List[Pauli]]:
            - S_list: Stabilizer generators (little-endian).
            - Z_basis: List of `Qz` responses for each i in Dprime.
            - X_basis: List of `Qx` responses for each i in Dprime.
    
    Raises:
        AssertionError: If any stabilizer string length mismatches d+k+a.
    """
    
    # Normalize to Qiskit little-endian labels
    if big_endian:
        stab_strings_le = [s[::-1] for s in stab_strings]   # reverse each label
    else:
        stab_strings_le = stab_strings

    S_list = PauliList(stab_strings_le)

    # sanity: local sizes
    d = len(Dprime_local); k = len(K2_local); a = len(A2_local)
    n = d + k + a
    assert all(len(s) == n for s in stab_strings_le), "Stabilizer length mismatch."

    # Build basis responses off Dprime
    Z_basis, X_basis = [], []
    for i in Dprime_local:
        lab = ['I']*n; lab[i] = 'Z'
        Qz = translate_pauli_off_Dprime(Pauli(''.join(lab)), S_list, Dprime_local)
        Z_basis.append(Qz)

        lab = ['I']*n; lab[i] = 'X'
        Qx = translate_pauli_off_Dprime(Pauli(''.join(lab)), S_list, Dprime_local)
        X_basis.append(Qx)

    return S_list, Z_basis, X_basis


def apply_local_pauli_qcontrolled(
    qc: QuantumCircuit,
    Q: Pauli,
    local_to_global: List[int],
    valid_global_targets_set: Set[int],
    control_qubit: Qubit | int,
) -> None:
    """
    Quantum-ify classical feed-forward:
      For each non-identity factor in Pauli Q (local frame), map it to the
      corresponding *global* wire and apply the Pauli as a gate controlled
      by `control_qubit`.

    Z  -> CZ(control, target)
    X  -> CX(control, target)
    Y  -> S(target); CX(control, target); Sdg(target)

    Args:
        qc: Circuit to modify in-place.
        Q: Pauli in the local (concatenated Dprime|K2|A2) frame.
        local_to_global: map local qubit index -> global qubit index in `qc`.
        valid_global_targets_set: allowed global targets (e.g. K2 ∪ A2 indices).
        control_qubit: the *quantum* control (Qubit or int index in `qc.qubits`).

    Notes:
        - Must be placed BEFORE measuring `control_qubit`.
        - If target == control, the gate is skipped (no valid self-controlled 1q gate).
    """
    # Normalize control to a Qubit object
    ctrl = qc.qubits[control_qubit] if isinstance(control_qubit, int) else control_qubit

    # Iterate over local qubits of Q
    for t_local in range(Q.num_qubits):
        x = bool(Q.x[t_local])
        z = bool(Q.z[t_local])
        if not (x or z):
            continue

        g = local_to_global[t_local]
        if g not in valid_global_targets_set:
            continue

        tgt = qc.qubits[g]
        if tgt is ctrl:
            # Classical control allowed same-wire; quantum control doesn't.
            # Skip or handle outside (e.g., refactor circuit) as needed.
            continue

        if x and z:
            # Controlled-Y on target: Y = S X S† (global phase ignored)
            qc.s(tgt)
            qc.cx(ctrl, tgt)
            qc.sdg(tgt)
        elif x:
            qc.cx(ctrl, tgt)
        else:  # z and not x
            qc.cz(ctrl, tgt)


def add_feedforward_corrections(
    qc: QuantumCircuit,
    stab_strings: List[Tuple[str, str]],                
    Dprime_global: List[int], 
    K2_global: List[int], 
    A2_global: List[int], 
    D_global: List[int]
) -> QuantumCircuit:
    """
    Add measurement-driven feed-forward Pauli corrections to K2 and A2.
    
    Given BIG-ENDIAN stabilizer strings over Dprime|K2|A2 and an existing circuit `qc` that has
    already measured Dprime (X-basis → Dprime_clr) and D (Z-basis → D_clr), this:
    1) Builds local→global index maps for (Dprime, K2, A2).
    2) Precomputes response patterns (Z/X) with `big_endian=True`.
    3) Applies conditional single-qubit Paulis to K2∪A2 using the recorded bits:
       - For each Dprime_i X-basis outcome bit: apply Z-pattern.
       - For each D_i  Z-basis outcome bit: apply X-pattern.
    
    Args:
        qc: Quantum circuit that already includes the Dprime (X) and D (Z) measurements.
        stab_strings: Stabilizer strings over Dprime|K2|A2 (big-endian layout).
        Dprime_global: Global qubit indices for the Dprime register.
        K2_global: Global qubit indices for the K2 register.
        A2_global: Global qubit indices for the A2 register.
        D_global: Global qubit indices for the D register (measured in Z).
        Dprime_clr: Classical bits holding Dprime X-basis outcomes (bit i ↔ Dprime_i).
        D_clr: Classical bits holding D Z-basis outcomes (bit i ↔ D_i).
    
    Returns:
        The input `qc` with conditional Pauli corrections appended in-place.
    """
    
    d = len(D_global); dprime = len(Dprime_global); k = len(K2_global); a = len(A2_global)
    Dprime_local = list(range(0, dprime))
    K2_local     = list(range(dprime, dprime+k))
    A2_local     = list(range(dprime+k, dprime+k+a))

    # local→global map (matches your physical indexing)
    local_to_global = list(Dprime_global) + list(K2_global) + list(A2_global)
    valid_global_targets = set(K2_global) | set(A2_global)

    # NEW: say big_endian=True here
    _, Z_basis, X_basis = precompute_responses(
        stab_strings, Dprime_local, K2_local, A2_local, big_endian=True
    )

    D, Dprime = grl.get_qreg(qc, "D"), grl.get_qreg(qc, "Dprime")
    # Feed-forward:
    # measurement on Dprime (X-basis): bit 1 ⇒ as-if Z on Dprime_i
    for i in range(d):
        apply_local_pauli_qcontrolled(qc, Z_basis[i],
                                        local_to_global, valid_global_targets,
                                        Dprime[i])

    # measurement on D (Z-basis): bit 1 ⇒ as-if X on Dprime_i (1–1 mapping)
    for i in range(d):
        apply_local_pauli_qcontrolled(qc, X_basis[i],
                                        local_to_global, valid_global_targets,
                                        D[i])

    return qc


def _lookup_reg(
    qc: QuantumCircuit, 
    name: str
) -> QuantumRegister:
    """
    Resolve a quantum register by name, with smart aliases for Dprime.
    
    Tries exact name matches in `qc.qregs`, then common aliases for Dprime (e.g., "Dprime",
    "D’", "D′", "Dp"). If still not found, sanitizes non-alphanumeric chars to underscores
    and retries. Raises if no register is found.
    
    Args:
        qc: QuantumCircuit containing the registers.
        name: Desired register name (e.g., "Dprime", "K2", "A2", "Dprime").
    
    Returns:
        QuantumRegister corresponding to `name` (or its alias).
    
    Raises:
        KeyError: If the register cannot be found.
    """
    
    regs = {r.name: r for r in qc.qregs}
    if name in regs:
        return regs[name]
    # common aliases for Dprime
    aliases = {
        "Dprime": "Dprime",
        "D’": "Dprime",   # curly apostrophe
        "D′": "Dprime",   # prime symbol
        "Dp": "Dprime",
    }
    alt = aliases.get(name, None)
    if alt and alt in regs:
        return regs[alt]
    # fallback: sanitize non-alnum to underscores
    sanitized = re.sub(r"\W+", "_", name)
    if sanitized in regs:
        return regs[sanitized]
    raise KeyError(f"QuantumRegister '{name}' not found. Available: {list(regs)}")


def global_indices_for_register(
    qc: QuantumCircuit, 
    reg_name: str
) -> List[int]:
    """
    Get global qubit indices for a named register.
    
    Looks up `reg_name` (supports Dprime aliases) and maps each local bit reg[i] to its global
    qubit index in `qc.qubits`.
    
    Args:
        qc: QuantumCircuit containing the register.
        reg_name: Register name (e.g., "D", "Dprime", "K2", "A2").
    
    Returns:
        List[int]: Global qubit indices for the register, in local order.
    """
    
    reg = _lookup_reg(qc, reg_name)
    # Map each local bit reg[i] to its global index in qc.qubits
    return [qc.find_bit(reg[i]).index for i in range(reg.size)]


def get_global_indices_dict(
    qc: QuantumCircuit
) -> Dict[str, List[int]]:
    """
    Collect global qubit indices for the canonical registers.
    
    Convenience wrapper returning the global index lists for D, Dprime, K2, and A2 using
    `global_indices_for_register`, with Dprime resolved via aliases when needed.
    
    Args:
        qc: QuantumCircuit containing the registers.
    
    Returns:
        Dict[str, List[int]]: Mapping {"D": [...], "Dprime": [...], "K2": [...], "A2": [...]}.
    """
    
    return {
        "D":   global_indices_for_register(qc, "D"),
        "Dprime":  global_indices_for_register(qc, "Dprime"),  # will resolve to Dprime/Dp/etc.
        "K2":  global_indices_for_register(qc, "K2"),
        "A2":  global_indices_for_register(qc, "A2"),
    }

    
def build_phi2_with_graph_state(
    qc2: QuantumCircuit, 
    n: int, 
) -> Tuple[QuantumCircuit, List[str], List[Tuple[str, str]]]:
    """
    Build the phi2 stage as a graph-state pipeline derived from an input Clifford circuit.
    
    This function takes a Clifford circuit `qc2` acting on `tot_qubits` qubits and:
    1) Simulates it as a stabilizer tableau to extract the Φ₂ stabilizer generators.
    2) Converts those generators to graph form (θ, adjacency, and local Clifford (LC) corrections).
    3) Constructs the corresponding 1D CZ graph-state circuit |G⟩ and the dagger of the LC
       corrections L†, then composes them to realize an equivalent implementation of `qc2`.
    4) Repackages the qubits into three named quantum registers — Dprime, K2, and A2 — with sizes
       determined by `n` (where N = 2**n), and appends classical registers to hold potential
       measurement outcomes on K2 and A2.
    
    Concretely:
    - A stabilizer tableau is built for `qc2` (`tab_phi2`), and `sg.run_clifford_with_meas_as_tableau`
      returns an initial sign/phase vector `s0` together with Φ₂ generators `phi2_gens`.
    - `sg.to_graph_form` yields a graph-state description plus local Clifford ops.
    - The circuit `qc_pipeline` is created by composing the 1D CZ graph-state circuit `qc_G_1D`
      followed by the local corrections dagger `qc_Ldag`.
    - Qubits are repartitioned into:
        • Dprime : first (2*N - 1) qubits
        • K2 : next (2*N - 1) qubits
        • A2 : remaining qubits
      and classical registers are added with lengths:
        • K2_clr : (N - 1 - n)
        • A2_clr : 2*(2*N - 1)
    
    Args:
       qc2: QuantumCircuit containing phi2, whose graph-state realization will be built.
       n:  Log₂ system size parameter; sets N = 2**n and determines the sizes of the Dprime, K2, A2
           partitions and the associated classical registers.
    
    Returns:
        qc_pipeline: The assembled `qc_pipeline` with registers (Dprime, K2, A2) and classical registers
                     (K2_clr, A2_clr) added.
        s0: Measurement outcomes of C_NOHE in tableasu simulation
        phi2_gens: Stabilizer generators of phi 2, given outcomes s0
    """
    
    N = 2**n
    tot_qubits = len(qc2.qubits)
    
    tab_phi2 = sg.Tableau(n=tot_qubits, seed=123)
    s0, phi2_gens = sg.run_clifford_with_meas_as_tableau(qc2, tab_phi2)
    
    theta, R, LC_ops, Sg = sg.to_graph_form(phi2_gens)
    
    # build circuits for |G> and L^†
    qc_G_1D = sg.build_1D_CZ_circuit(theta)
    qc_Ldag = sg.local_ops_to_circuit_dagger(LC_ops)
    
    # apply |G> then L^†
    qc_pipeline = QuantumCircuit(tot_qubits)
    qc_pipeline.compose(qc_G_1D, qubits=range(tot_qubits), inplace=True)
    qc_pipeline.compose(qc_Ldag, qubits=range(tot_qubits), inplace=True)
    
    # restructure qc_pipeline
    Dprime = qc_pipeline.qubits[:2*N-1]
    K2 = qc_pipeline.qubits[2*N-1:2*(2*N-1)]
    A2 = qc_pipeline.qubits[2*(2*N-1):]
    qc_pipeline = grl.repackage_into_registers(qc_pipeline, layout=[("Dprime", Dprime),
                                                                    ("K2", K2),
                                                                    ("A2", A2)])
    
    num_clbits_A2, num_clbits_K2 = 2*(2*N-n-2), 2*N-1 - (n+1)
    K2_clr = ClassicalRegister(num_clbits_K2, "K2_clr")
    A2_clr = ClassicalRegister(num_clbits_A2, "A2_clr")
    qc_pipeline.add_register(K2_clr, A2_clr)
    
    return qc_pipeline, s0, phi2_gens


def merge_phi1_phi2(
    qc_pipeline: QuantumCircuit, 
    qc1: QuantumCircuit, 
    n: int, 
    phi2_gens: Sequence[Tuple[str, str]]
) -> QuantumCircuit:
    """
    Merges circuits for phi1 and phi2 according to description in the paper. Does
    so by first tensoring the circuits, then performing partial BMs between D and 
    Dprime. Then feeds forward measurement outcomes of partial BMs to apply Pauli 
    corrections to registers K2 and A2, such that all BM outcomes were effectively 0.

    Args:
        qc_pipeline: QuantumCircuit containing phi2 (built from graph form)
        qc1: QuantumCircuit containing phi1
        n: Integer specifying size of corresponding address register
        phi2_gens: list of generators of phi2 (including signs)

    Returns:
        qc: QuantumCircuit containing merged phi1 and phi2
    """
    
    N = 2**n
    
    # tensor circuits for phi1 and phi2
    qc = qc_pipeline.tensor(qc1)

    # get and add registers for partial BMs
    D, Dprime = grl.get_qreg(qc, "D"), grl.get_qreg(qc, "Dprime")
    D_clr = ClassicalRegister(N-1, "D_NOHE_clr")
    Dprime_clr = ClassicalRegister(2*N-1, "Dprime_clr")
    qc.add_register(D_clr, Dprime_clr)

    # perform partial BMs
    qc.cz(D[:], Dprime[:])
    qc.h(D[:N-1] + Dprime[:])

    # prepare input for feedforward
    stab_strings = [b for _, b in phi2_gens]

    idx = get_global_indices_dict(qc)
    D_global       = idx["D"][:N-1]
    Dprime_global  = idx["Dprime"]   
    K2_global      = idx["K2"]
    A2_global      = idx["A2"]

    # feedforward
    qc = add_feedforward_corrections(
        qc,
        stab_strings=stab_strings,
        Dprime_global=Dprime_global,
        K2_global=K2_global,
        A2_global=A2_global,
        D_global=D_global
    )

    qc.measure(D[:N-1] + Dprime[:], D_clr[:] + Dprime_clr[:])
    
    return qc


def make_phi(
    n: int
) -> Tuple[QuantumCircuit, List[int]]:
    """
    Prepares full resource state, |Φ>, for an address register of size n. The function
    first makes phi1 and phi2. Then rebuilds phi2 from it's graph form with a measurment
    free circuit, because that way the circuit holds a phi2 with a generator list that 
    corresponds to known measurement outcomes of C_NOHE, contained in s0. (If we do not
    do this, and keep actual measurement operations in the system, the stabilizer group
    changes when a simulaton runs these measurements)
    
    Note that the circuit still contains registers K2 and A2. 

    Args:
        n: Integer specifying size of corresponding address register

    Returns:
        qc: QuantumCircuit containing |Φ>
        s0: Measurement outcomes of C_NOHE
    """
    
    N = 2**n

    # make phi1
    qc1 = rsp.make_Phi1(n)
    I  = qc1.qregs[0][:]
    D = [q for reg in qc1.qregs[1:] for q in reg]
    qc1 = grl.repackage_into_registers(qc1, layout=[("I", I),
                                                  ("D", D)])
    
    # make phi2, get measurements and stabilizers
    qc2, mapping = rsp.make_Phi2(n)
    qc_pipeline, s0, phi2_gens = build_phi2_with_graph_state(qc2, n)

    # merge phi1 and phi2
    qc = merge_phi1_phi2(qc_pipeline, qc1, n, phi2_gens)

    # repackage qubit registers to match paper 
    I, D, Dprime, K2,  A2 = grl.get_qreg(qc, "I"), grl.get_qreg(qc, "D"), grl.get_qreg(qc, "Dprime"), grl.get_qreg(qc, "K2"), grl.get_qreg(qc, "A2")
    D_NOHE, L = D[:N-1], D[N-1:]
    I, Dprime, K2, A2 = I[:], Dprime[:], K2[:], A2[:]
    qc = grl.repackage_into_registers(qc, layout=[("I", I),
                                                  ("D_NOHE", D_NOHE),
                                                  ("L", L),
                                                  ("Dprime", Dprime),
                                                  ("K2", K2),
                                                  ("A2", A2)])

    return qc, s0, mapping


def merge_phi1_phi2_normal(qc1, qc2, n, save_svs=False):
    """
    """

    def outcome_string_to_Pin(outcome, n):
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
    
    def precompute_merge_corrections(n):
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

def make_phi_normal(n, save_svs=False):
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
