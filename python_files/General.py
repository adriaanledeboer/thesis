from __future__ import annotations  # <-- must be here (and before other imports)

# Qiskit packages
import qiskit, qiskit_aer#, qiskit_ibm_runtime, qiskit_ibm_catalog

# Own packages
from . import Query as qry
from . import RS_prep as rsp
from . import General as grl

# Other packages
import numpy as np 
import math
import matplotlib.pyplot as plt

# Qiskit functions
from qiskit import QuantumRegister, ClassicalRegister, QuantumCircuit, transpile
from qiskit.quantum_info import Statevector, DensityMatrix, partial_trace,  Clifford, Pauli, PauliList
from qiskit.circuit import Parameter, ParameterVector, Instruction
from qiskit_aer import AerSimulator
from qiskit.circuit.classical import expr
from qiskit.visualization import plot_state_city, plot_state_hinton

# Other funtions
from typing import Optional
from IPython.display import Math
from numpy.random import default_rng
from collections import deque, Counter
from itertools import product


def qubit_inventory(
    qc: QuantumCircuit
) -> list:
    """
    Returns a list containing the qubit inventory information of QuantumCircuit (qc)
    """
    
    rows = []
    for gidx, q in enumerate(qc.qubits):
        # find owning register & index
        for reg in qc.qregs:
            try:
                ridx = reg.index(q)
                rows.append({
                    "repr": str(q),           # e.g. "a[0]"
                    "global": gidx,           # position in qc.qubits
                    "reg": reg.name,          # register name
                    "reg_idx": ridx,          # index within that register
                })
                break
            except ValueError:
                continue

    return rows


def _clbit_inventory(qc: QuantumCircuit):
    """Build rows like qubit_inventory(qc) but for classical bits."""
    rows = []
    # global index is the position in qc.clbits
    for g_idx, bit in enumerate(qc.clbits):
        reg_name = ""
        reg_idx = ""
        # find the classical register (if any) that contains this bit
        for creg in qc.cregs:
            if bit in creg:
                reg_name = creg.name
                reg_idx = list(creg).index(bit)
                break
        rows.append({
            "repr": repr(bit),
            "global": g_idx,
            "reg": reg_name,
            "reg_idx": reg_idx
        })
    return rows

def _print_rows(rows, title: str):
    """Pretty-print rows with the same columns you use for qubits."""
    # handle empty gracefully
    if not rows:
        print(f"{title}\n(no bits)\n")
        return

    w_repr = max(len("qubit"), *(len(r["repr"]) for r in rows))
    w_glob = max(len("global"), *(len(str(r["global"])) for r in rows))
    w_reg  = max(len("register"), *(len(r["reg"]) for r in rows))
    w_ridx = max(len("reg_idx"), *(len(str(r["reg_idx"])) for r in rows))

    header = (f"{title:<{w_repr}}  "
              f"{'global':>{w_glob}}  "
              f"{'register':<{w_reg}}  "
              f"{'reg_idx':>{w_ridx}}")
    print(header)
    print("-" * len(header))
    for r in rows:
        print(f"{r['repr']:<{w_repr}}  "
              f"{r['global']:>{w_glob}}  "
              f"{r['reg']:<{w_reg}}  "
              f"{r['reg_idx']:>{w_ridx}}")
    print()  # spacing after table

def print_qubit_table(qc: QuantumCircuit) -> None:
    """
    Prints table of qubit inventory of QuantumCircuit (qc),
    and additionally a table for clbits.
    """
    # Original qubit table
    qrows = qubit_inventory(qc)
    _print_rows(qrows, "qubit")

    # New clbit table
    crows = _clbit_inventory(qc)
    _print_rows(crows, "clbit")


def repackage_into_registers(
    qc: QuantumCircuit,
    layout: Sequence[Tuple[str, Sequence[Qubit]]],
) -> QuantumCircuit:
    """
    Return a new circuit whose **quantum registers** are exactly as specified by
    `layout`, without changing any operations. Classical registers are preserved.

    Args:
        qc: Source circuit.
        layout: Sequence of `(name, qubits)` pairs. The concatenated `qubits`
            must be a **permutation** of `qc.qubits` (covers all, no duplicates).

    Returns:
        A new `QuantumCircuit` with registers created and ordered per `layout`.

    Raises:
        ValueError: If `layout` is not a permutation/partition of `qc.qubits`.

    Example:
        # make registers R1, R2 from arbitrary qubits
        qc2 = repackage_into_registers(qc, [("R1", [b[1], a[0]]), ("R2", [b[0], b[2]])])
    """
    
    old_order = list(qc.qubits)

    # validate layout covers all qubits exactly once
    flat: list[Qubit] = []
    for _name, qs in layout:
        flat.extend(qs)
    if len(flat) != len(old_order) or len(set(flat)) != len(flat) or set(flat) != set(old_order):
        raise ValueError("`layout` must be a permutation partition of qc.qubits")

    # build new regs
    new_qregs = []
    new_linear = []
    for name, qs in layout:
        r = QuantumRegister(len(qs), name)
        new_qregs.append(r)
        new_linear.extend(r[:])

    qc_new = QuantumCircuit(*new_qregs, *qc.cregs)

    # map old -> new
    pos_in_new = {orig: new for new, orig in zip(new_linear, flat)}
    targets = [pos_in_new[q] for q in old_order]
    qc_new.compose(qc, qubits=targets, clbits=qc_new.clbits, inplace=True)
    return qc_new
    

def make_BPs(
    n: int
) -> Gate:
    """
    Return a Gate that creates n Bell pairs on 2n wires: pairs (i, i+n) via H on all and CZ between halves.

    Args:
        n: Number of Bell pairs (uses 2*n qubits).

    Returns:
        Gate acting on 2*n qubits.
    """
    
    total = 2*n
    sub = QuantumCircuit(total, name=f'make_{n}_BPs')
    # First 0, ..., n-1 qubits are entangled with the last n, ..., 2*n-1 qubits

    sub.h(sub.qubits)
    sub.cz(sub.qubits[:n], sub.qubits[n:])

    return sub#.to_gate(label=sub.name)


def apply_make_BPs(
    qc: QuantumCircuit,
    input_qubits: Sequence[Qubit]
) -> None:
    """
    Append the n-pair Bell creator to `qc`, pairing the first half of `input_qubits` with the second half.

    Args:
        qc: Circuit to modify.
        input_qubits: Qubits (length 2n), paired as (i, i+n).

    Raises:
        ValueError: If `len(input_qubits)` is odd.
    """
    
    tot_num_qubits = len(input_qubits)
    if (tot_num_qubits % 2) != 0:
        raise ValueError("Total number of qubits must be even")
        
    n = int(tot_num_qubits / 2)


    sub = make_BPs(n)
    targets = list(input_qubits)

    qc.compose(sub, qubits=targets, inplace=True)


def prepare_BMs(n: int) -> Gate:
    """Gate: pairwise CZ between halves then H on all (2n wires)."""
    sub = QuantumCircuit(2*n, name=f"Prepare {n} BMs")
    sub.cz(sub.qubits[:n], sub.qubits[n:])
    sub.h(sub.qubits)
    return sub#.to_gate(label=sub.name)
    

def apply_BMs(
    qc: QuantumCircuit,
    qubits: Sequence[Qubit],
    clbits: Sequence[Clbit]
) -> None:
    """
    Append BMs on `qubits`, then measure the **entire** circuit globally.
    Requires: `qc` already has at least as many classical bits as qubits.
    """
    if len(qubits) % 2 != 0:
        raise ValueError("`qubits` length must be even (2n).")
    if len(clbits) != len(qubits):
        raise ValueError(
            f"Number of qubits and measurement bits do not match: {len(qubits)} qubits and {len(clbits)} classical bits"
            "Allocate them before calling."
        )

    # 1) Append the quantum-only block
    n = int(len(qubits) / 2)
    sub = prepare_BMs(n)
    qc.compose(sub, qubits=list(qubits), inplace=True)

    # 2) Add global measurement
    qc.measure(qubits, clbits)

    # qc.save_statevector('post_meas', pershot=True)


def get_qreg(circ: QuantumCircuit, name: str) -> QuantumRegister:
    return next(r for r in circ.qregs if r.name == name)
    

def get_creg(circ: QuantumCircuit, name: str) -> ClassicalRegister:
    return next(r for r in circ.cregs if r.name == name)


def bits_to_int(lsb_first):
    return sum(b << i for i, b in enumerate(lsb_first))

    
def all_binary_lists(n: int):
    """Return 2^n lists (length n) with first position flipping fastest."""
    return [list(bits)[::-1] for bits in product([0, 1], repeat=n)]
        

def apply_adaptive_part_U_NOHE_inversion(
    qc: QuantumCircuit,
    input_qubits: Sequence[Qubit],
    input_clbits: Sequence[Clbit],
    n: int,
    save_svs = False
) -> None:
    """
    Adaptive SQPMs for NOHE inversion:
      • Choose A2 measurement bases (X vs Z) based on prior X-measurements (K2_clr).
      • Do NOT apply Pauli corrections mid-circuit.
      • After all measurements, apply one pass of conditional Zs on the relevant K2 qubits.
    """
    N = 2**n
    mapping = qc.metadata[f"C_NOHE_mapping[{N}]"]
    info = qc.metadata[f"C_NOHE_info[{N}]"]
    third_wires = qc.metadata[f"C_NOHE_third_wires[{N}]"]

    expected_qubits = N - 1 + 2*(N - n - 1)
    expected_clbits = (N - n - 1)
    if len(input_qubits) != expected_qubits:
        raise ValueError(f"Need {expected_qubits} qubits, got {len(input_qubits)}")
    if len(input_clbits) != expected_clbits:
        raise ValueError(f"Need {expected_clbits} clbits, got {len(input_clbits)}")

    def find_loc_bit_idx(qc, q, reg_name):
        """ Assumes q is in only 1 reg """
        reg = get_qreg(qc, reg_name)
        return qc.find_bit(q).registers[-1][-1]

    def precompute_P_outs_for_layer(n, layer, local_indices, expected_qubits):
        """ Returns a dict, where the keys are the alpha beta combs and the items are the P_outs """
         # make Clifford to propagate Pauli byproduct through
    
        C = Clifford(rsp.make_C_NOHE_circuit(n, target_layer=layer, measure=False)[0])
    
        num_outcomes = len(local_indices)
        outcome_combs = all_binary_lists(num_outcomes)
    
        d = {}
        for outcome_comb in outcome_combs:
            x = np.zeros(expected_qubits, dtype=bool)
            z = np.zeros(expected_qubits, dtype=bool)
            
            for j in range(num_outcomes):
                if outcome_comb[j] == 1:
                    loc_idx = local_indices[j]
                    z[loc_idx] = True
                    
            P_in = Pauli((z, x))
            P_out = P_in.evolve(C, frame="s")
            d[bits_to_int(outcome_comb)] = P_out
    
        return d

    K2      = list(input_qubits[:N-1])
    A2      = list(input_qubits[N-1:])
    K2_clr  = list(input_clbits[:])

    count = 0
    
    for layer, layer_dict in info.items():
        num_gadgets = len(layer_dict)
        local_indices = [] 
        A2_clr = ClassicalRegister(2 * num_gadgets, f"A2_layer_{layer}_clr")
        qc.add_register(A2_clr)
        i = 0
        
        for gadget, gadget_dict in layer_dict.items():
            q0, q1, q2 = gadget_dict["qubits"]
            A0, A1     = gadget_dict["ancillas"]
            Kc         = gadget_dict["clbit"]
            Ac0, Ac1   = A2_clr[2*i], A2_clr[2*i+1]

            # measure A0, A1 in X or Z
            with qc.if_test((Kc, 0)) as else_:
                qc.measure([A0, A1], [Ac0, Ac1])
            with else_:
                qc.h(A0)
                qc.h(A1)
                qc.measure([A0, A1], [Ac1, Ac0])   # reversed roles of alpha and beta

            local_idx_q0 = find_loc_bit_idx(qc, q0, "K2")
            local_idx_q1 = find_loc_bit_idx(qc, q1, "K2")

            local_indices.append(local_idx_q0)
            local_indices.append(local_idx_q1)

            if save_svs:
                qc.save_statevector(label=f"adapt layer: {layer}, gadget: {gadget}, count: {count}", pershot=True)

            i += 1
            count += 1
            
        # print(local_indices)
        
        # get corrections from propagated Pauli byproduct
        P_out_dict = precompute_P_outs_for_layer(n, layer, local_indices, expected_qubits)

        # rotate q2's in K2 back to Z frame before applying corrections
        for q2 in third_wires:
            qc.h(q2)

        # apply corrections
        for outcome_comb, P_out in P_out_dict.items():
            with qc.if_test((A2_clr, outcome_comb)):
                Zarr, Xarr = P_out.z, P_out.x
                for j, q in enumerate(K2):
                        if Zarr[j]:
                            qc.z(q)
                        if Xarr[j]:
                            qc.x(q)

        # measure q2's in K2 in X basis
        j = 0
        for q2 in third_wires:
            qc.h(q2)
            qc.measure(q2, K2_clr[j])
            j += 1

        if save_svs:
            qc.save_statevector(label=f"adapt layer: {layer} done", pershot=True)


def apply_inverse_swaps(
    qc: QuantumCircuit,
    input_qubits: Sequence[Qubit]
) -> None:
    """
    Undo the SWAP layer from Eq. (9) so the recovered |ψ⟩ sits on the LSB wires.
    Forward used K = n-1..2, so inverse is K = 2..n-1.
    """
    N_minus_1 = len(input_qubits)
    N = N_minus_1 + 1
    n = int(math.log2(N))

    for K in range(2, n):   # K = 2 .. n-1
        qc.swap(input_qubits[K], input_qubits[2**K - 1])

    # for K in range(n - 1, 1, -1):  # descending
    #     qc.swap(input_qubits[K], input_qubits[2**K - 1])
    

def make_U_NOHE_inversion(
    n: int
) -> QuantumCircuit:
    """
    """

    N = 2**n
    total_qubits = N-1
    sub = QuantumCircuit(total_qubits, name=f"U_NOHE[{N}]^-1")
    sub = repackage_into_registers(sub, layout=[("K2", sub.qubits[:])])
    K2 = sub.qubits

    # C_NOHE on K2
    A2, K2_clr, mapping, info, third_wires = rsp.apply_C_NOHE(sub, K2[:])

    # Adaptive part
    apply_adaptive_part_U_NOHE_inversion(sub, K2[:] + A2[:], K2_clr[:], n)
    
    # Apply SWAPs
    apply_inverse_swaps(sub, input_qubits=K2)

    return sub


def apply_U_NOHE_inversion(
    qc: QuantumCircuit,
    input_qubits: Sequence[Qubit]
) -> None:
    """
    """
    K2 = input_qubits
    N = len(K2) + 1
    n = int(math.log2(N))

    # num_qubits_A2 = 2*(N-n-1)
    # num_clbits_A2 = num_qubits_A2
    # num_clbits_K2 = (N-1-n)

    # K2 = input_qubits
    # A2 = QuantumRegister(num_qubits_A2, "A2")
    # K2_clr = ClassicalRegister(num_clbits_K2, "K2_clr")
    # # A2_clr = ClassicalRegister(num_clbits_A2, "A2_clr")
    # qc.add_register(A2, K2_clr)#, A2_clr)

    # sub = make_U_NOHE_inversion(n)
    # target_qubits = K2[:] + A2[:]
    # target_clbits = K2_clr[:] #+ A2_clr[:]

    # qc.compose(sub, qubits=target_qubits, clbits=target_clbits, inplace=True)


    A2, K2_clr, mapping, info, third_wires = rsp.apply_C_NOHE(qc, input_qubits=K2[:])

    apply_adaptive_part_U_NOHE_inversion(qc, K2[:] + A2[:], K2_clr[:], n)

    apply_inverse_swaps(qc, input_qubits=K2[:])


    


def counts_to_dm(counts, n):
    probs = np.zeros(2**n, float)
    total = sum(counts.values())
    for bitstr, c in counts.items():
        idx = int(bitstr.replace(' ', ''), 2)   # adjust if you need reversed order
        probs[idx] = c/total
    return DensityMatrix(np.diag(probs).astype(complex))

def plot_dm_heatmap(rho, which="abs", max_dim=512):
    A = rho.data
    d = A.shape[0]
    if A.shape[0] > max_dim:
        raise MemoryError(f"Matrix is {A.shape[0]}×{A.shape[0]}; cap is {max_dim}. "
                          "Trace out to a small subsystem first.")
    if which == "abs":
        M = np.abs(A)
    elif which == "real":
        M = np.real(A)
    elif which == "imag":
        M = np.imag(A)
    else:
        raise ValueError("which ∈ {'abs','real','imag'}")
    
    fig, ax = plt.subplots(figsize=(10, 10))
    im = ax.imshow(M, aspect='equal', interpolation='nearest')

    # colorbar + title/labels
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.formatter.set_powerlimits((0, 0))
    ax.set_title(f"Density matrix heatmap ({which})")
    ax.set_xlabel("|j⟩")
    ax.set_ylabel("|i⟩")
    fig.tight_layout()

    ax.set_xticks(np.arange(-0.5, d, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, d, 1), minor=True)
    ax.grid(which='minor', linestyle='-', linewidth=0.3)
    ax.tick_params(which='minor', bottom=False, left=False)

def reduced_dm(rho: DensityMatrix, keep_qubits):
    n = rho.num_qubits
    trace_out = [q for q in range(n) if q not in keep_qubits]
    return partial_trace(rho, trace_out)
    

def simulate_circuit(qc, shots=10, method="statevector", print_mem=False):
    """ """
    t0 = time.time()
    # simulate
    sim = AerSimulator(method= "statevector")
    tqc = transpile(qc, backend=sim, optimization_level=0)
    res = sim.run(tqc, shots=shots, memory=True).result()
    t1 = time.time()
    
    print("Success:", res.success)
    print("Overall status:", res.to_dict().get("status", ""))
    print(f"Elapsed time is {(t1-t0)/60} min")
    
    mem    = res.get_memory()
    counts = res.get_counts()

    if print_mem:
        for m in mem:
            print(m)

    return mem, counts
