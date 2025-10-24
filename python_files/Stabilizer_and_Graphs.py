from __future__ import annotations  # <-- must be here (and before other imports)

# Qiskit packages
import qiskit, qiskit_aer#, qiskit_ibm_runtime, qiskit_ibm_catalog

# Own packages|
from python_files import Query as qry, RS_prep as rsp, General as grl, Stabilizer_and_Graphs as sg

# Other packages
import numpy as np 
import math
import time

# Qiskit functions
from qiskit import QuantumRegister, ClassicalRegister, QuantumCircuit, transpile
from qiskit.quantum_info import Statevector
from qiskit.circuit import Instruction
from qiskit.circuit.classical import expr
from qiskit_aer import AerSimulator
from qiskit.transpiler.passes import RemoveBarriers

# Other funtions
from typing import Optional
from IPython.display import Math
from numpy.random import default_rng
from collections import deque, Counter


# ---------------------------------- Tableau Class ----------------------------------

class Tableau:
    def __init__(self, n, seed=None):
        self.n = n
        self.X = np.zeros((2*n, n), dtype=np.uint8)
        self.Z = np.zeros((2*n, n), dtype=np.uint8)
        self.r = np.zeros(2*n, dtype=np.uint8)
        self.rng = default_rng(seed)
        # standard initial tableau for |0>^n
        for j in range(n):
            self.X[j, j] = 1          # destabilizers: X on each qubit
            self.Z[n + j, j] = 1      # stabilizers:   Z on each qubit

    # ---------- gates ----------
    def H(self, a):
        x = self.X[:, a].copy()
        z = self.Z[:, a].copy()
        self.r ^= (x & z)  # flip phase if both 1 before swap
        self.X[:, a], self.Z[:, a] = z, x

    def S(self, a):  # Phase
        self.r ^= (self.X[:, a] & self.Z[:, a])
        self.Z[:, a] ^= self.X[:, a]

    def CNOT(self, a, b):
        # phase update per paper: r ^= x_a & z_b & (x_b ^ z_a ^ 1)
        self.r ^= (self.X[:, a] & self.Z[:, b] & (self.X[:, b] ^ self.Z[:, a] ^ 1))
        self.X[:, b] ^= self.X[:, a]
        self.Z[:, a] ^= self.Z[:, b]

    def CZ(self, a, b):
        self.H(b); self.CNOT(a, b); self.H(b)

    def Z_gate(self, a):
        # Conjugation by Z on qubit a:
        # flips the sign of any row whose Pauli has X on qubit a (X or Y).
        self.r ^= self.X[:, a]

    # rowsum (h <- h + i) with full phase logic
    def _rowsum_rows(self, h, i):
        # Use int arrays for arithmetic in {0,1} -> integers
        x1 = self.X[i].astype(np.int8);  z1 = self.Z[i].astype(np.int8)
        x2 = self.X[h].astype(np.int8);  z2 = self.Z[h].astype(np.int8)

        # g(x1,z1,x2,z2) per paper
        g = np.zeros(self.n, dtype=np.int8)

        mask00 = (x1 == 0) & (z1 == 0)
        mask11 = (x1 == 1) & (z1 == 1)
        mask10 = (x1 == 1) & (z1 == 0)
        mask01 = (x1 == 0) & (z1 == 1)

        g[mask11] = (z2 - x2)[mask11]
        g[mask10] = (z2 * (2*x2 - 1))[mask10]
        g[mask01] = (x2 * (1 - 2*z2))[mask01]
        # mask00 contributes 0

        total = (2*int(self.r[h]) + 2*int(self.r[i]) + int(g.sum())) % 4
        self.r[h] = np.uint8(0 if total == 0 else 1)

        # XOR the Paulis
        self.X[h] ^= self.X[i]
        self.Z[h] ^= self.Z[i]

    # helper: rowsum into a temporary scratch row (xh, zh, rh)
    def _rowsum_into_temp(self, xh, zh, rh, i):
        x1 = self.X[i].astype(np.int8);  z1 = self.Z[i].astype(np.int8)
        x2 = xh.astype(np.int8);         z2 = zh.astype(np.int8)

        g = np.zeros(self.n, dtype=np.int8)
        mask00 = (x1 == 0) & (z1 == 0)
        mask11 = (x1 == 1) & (z1 == 1)
        mask10 = (x1 == 1) & (z1 == 0)
        mask01 = (x1 == 0) & (z1 == 1)

        g[mask11] = (z2 - x2)[mask11]
        g[mask10] = (z2 * (2*x2 - 1))[mask10]
        g[mask01] = (x2 * (1 - 2*z2))[mask01]

        total = (2*int(rh) + 2*int(self.r[i]) + int(g.sum())) % 4
        rh = np.uint8(0 if total == 0 else 1)

        xh ^= self.X[i]
        zh ^= self.Z[i]
        return xh, zh, rh

    # ---------- Z-basis measurement on qubit a ----------
    def measure_z(self, a):
        n = self.n
        # Case I: exists stabilizer row p with X[p, a] == 1  -> random outcome, update state
        Xpa = self.X[n:, a]  # stabilizer block rows n..2n-1
        p_offsets = np.flatnonzero(Xpa)
        if p_offsets.size > 0:
            p = n + int(p_offsets[0])  # smallest such p
            # For all i != p with X[i,a]==1, do rowsum(i, p)
            col = self.X[:, a]
            one_idxs = np.flatnonzero(col)
            for i in one_idxs:
                if i != p:
                    self._rowsum_rows(i, p)
            # Copy row p -> row p-n
            self.X[p - n, :] = self.X[p, :]
            self.Z[p - n, :] = self.Z[p, :]
            self.r[p - n] = self.r[p]
            # Set row p to have Z on qubit a, all else 0; rp is random bit
            self.X[p, :].fill(0)
            self.Z[p, :].fill(0)
            self.Z[p, a] = 1
            self.r[p] = np.uint8(self.rng.integers(0, 2))
            return int(self.r[p])  # outcome bit

        # Case II: no stabilizer row has X[p,a]==1  -> determinate; compute outcome via scratch sum
        # Scratch row starts as identity (all zero, rh=0)
        xh = np.zeros(self.n, dtype=np.uint8)
        zh = np.zeros(self.n, dtype=np.uint8)
        rh = np.uint8(0)
        # For each destabilizer row i with X[i,a]==1, add the corresponding stabilizer row (i+n)
        xi_col = self.X[:n, a]
        for i in np.flatnonzero(xi_col):
            xh, zh, rh = self._rowsum_into_temp(xh, zh, rh, i + n)
        # Outcome is rh
        return int(rh)

    def stabilizers(self, remove_signs=False):
        # rows n..2n-1 -> list of (sign, pauli_string)
        out = []
        for i in range(self.n, 2*self.n):
            paulis = []
            sign = '-' if self.r[i] else '+'
            for j in range(self.n):
                x, z = self.X[i, j], self.Z[i, j]
                paulis.append('I' if (x|z)==0 else ('X' if (x==1 and z==0)
                    else ('Z' if (x==0 and z==1) else 'Y')))
            if not remove_signs:
                out.append((sign, ''.join(paulis)))
            else:
                out.append(''.join(paulis))
        return out


# ---------------------------------- Updating Tableau Based on Clifford Circuit ---------------------------------- 

def run_clifford_with_meas_as_tableau(qc: QuantumCircuit, tab, remove_signs=False):
    """
    Execute a Clifford(+measure) circuit on a Tableau(n) state simulator.
    Returns:
        meas_outcomes: dict {classical_bit_index: 0/1}
        final_stabilizers: list of (sign, pauli_string) from tab.stabilizers()
    """
    meas_outcomes = {}

    # Iterate over the instructions
    for inst, qargs, cargs in qc.data:
        name = inst.name
        # flat indices for qubits / clbits
        qs = [qc.find_bit(q).index for q in qargs]

        if name == "h":
            tab.H(qs[0])

        elif name in ("s",):  # if you also use S in your circuits
            tab.S(qs[0])

        elif name == "z":                  
            tab.Z_gate(qs[0])

        elif name in ("cx", "cnot"):
            tab.CNOT(qs[0], qs[1])

        elif name == "cz":
            tab.CZ(qs[0], qs[1])

        elif name == "measure":
            # Z-basis measurement on that qubit (collapse)
            bit = tab.measure_z(qs[0])
            c = qc.find_bit(cargs[0]).index
            meas_outcomes[c] = bit

        else:
            raise ValueError(f"Unsupported op in tableau driver: {name}")

    return meas_outcomes, tab.stabilizers(remove_signs=remove_signs)

def pretty_stabs(gens):
    for sign, pauli in gens:
        print(f"{sign} {pauli}")
        

# ---------------------------------- Finding LC-eq Graph state and LC-circuit ---------------------------------- 

# ---------- GF(2) utilities ----------
def gf2_inv(A):
    A = np.array(A, dtype=np.uint8) % 2
    n = A.shape[0]
    M = np.concatenate([A, np.eye(n, dtype=np.uint8)], axis=1)
    r = 0
    for c in range(n):
        piv = None
        for rr in range(r, n):
            if M[rr, c]:
                piv = rr; break
        if piv is None:
            raise np.linalg.LinAlgError("singular over GF(2)")
        if piv != r:
            M[[r, piv]] = M[[piv, r]]
        for rr in range(n):
            if rr != r and M[rr, c]:
                M[rr, :] ^= M[r, :]
        r += 1
    return M[:, n:]

def gf2_rank(A):
    A = np.array(A, dtype=np.uint8) % 2
    A = A.copy()
    n, m = A.shape
    r = 0
    for c in range(m):
        piv = None
        for rr in range(r, n):
            if A[rr, c]:
                piv = rr; break
        if piv is None: 
            continue
        if piv != r:
            A[[r, piv]] = A[[piv, r]]
        for rr in range(n):
            if rr != r and A[rr, c]:
                A[rr, :] ^= A[r, :]
        r += 1
    return r

# ---------- Stabilizers → [Z;X] with phases ----------
PAULI2ZX = {'I': (0,0), 'X': (0,1), 'Z': (1,0), 'Y': (1,1)}

def pauli_to_zx(pstr):
    z = np.fromiter((PAULI2ZX[p][0] for p in pstr), dtype=np.uint8)
    x = np.fromiter((PAULI2ZX[p][1] for p in pstr), dtype=np.uint8)
    return z, x

def stabilizers_to_Sr(generators):
    n = len(generators)
    Z = np.zeros((n, n), dtype=np.uint8)
    X = np.zeros((n, n), dtype=np.uint8)
    r = np.zeros(n, dtype=np.uint8)
    for j, (sgn, pstr) in enumerate(generators):
        r[j] = 0 if sgn == '+' else 1
        z, x = pauli_to_zx(pstr)
        if len(z) != n:
            raise ValueError("Generator length mismatch.")
        Z[:, j] = z
        X[:, j] = x
    S = np.vstack([Z, X]) % 2
    return S, r

def apply_H_with_r(S, r, q):
    n = S.shape[0] // 2
    r ^= (S[n+q, :] & S[q, :])
    S[[q, n+q], :] = S[[n+q, q], :]

def apply_S_with_r(S, r, q):
    n = S.shape[0] // 2
    r ^= (S[n+q, :] & S[q, :])
    S[q, :] ^= S[n+q, :]

def to_graph_form(final_gens):
    """
    Returns:
      theta  : adjacency (n×n) of the LC-equivalent graph state |G>
      R      : generator basis change over GF(2)
      LC_ops : per-qubit string over {'H','S','Z'} s.t. L maps |Phi2> -> |G>
      Sg     : stabilizer in graph form [theta; I]
    """
    S, r = stabilizers_to_Sr(final_gens)
    n = S.shape[0] // 2
    LC_ops = [''] * n

    def try_sequences(q, seqs):
        best_rank = -1
        best_S = best_r = best_seq = None
        for seq in seqs:
            S_tmp = S.copy(); r_tmp = r.copy()
            for g in seq:
                if g == 'H': apply_H_with_r(S_tmp, r_tmp, q)
                elif g == 'S': apply_S_with_r(S_tmp, r_tmp, q)
                else: raise ValueError(f"Unknown gate '{g}'")
            rank_tmp = gf2_rank(S_tmp[n:, :])
            if rank_tmp > best_rank:
                best_rank, best_S, best_r, best_seq = rank_tmp, S_tmp, r_tmp, seq
        return best_rank, best_S, best_r, best_seq

    SEQS = ["H", "S", "HS", "SH", "HSH", "SHS"]
    while gf2_rank(S[n:, :]) < n:
        base_rank = gf2_rank(S[n:, :])
        improved = False
        for q in range(n):
            rank_new, S_new, r_new, seq = try_sequences(q, SEQS)
            if rank_new > base_rank:
                S, r = S_new, r_new
                LC_ops[q] += seq
                improved = True
                break
        if not improved:
            raise RuntimeError("Could not make X invertible with local {H,S}.")

    X = S[n:, :] % 2
    Z = S[:n, :] % 2
    Xinv = gf2_inv(X)
    Zp  = (Z @ Xinv) % 2
    Xp  = (X @ Xinv) % 2

    I = np.eye(n, dtype=np.uint8)
    if not np.array_equal(Xp, I): raise AssertionError("Postcondition X' = I failed.")
    if not np.array_equal(Zp, Zp.T): raise AssertionError("Theta not symmetric.")
    if np.any(np.diag(Zp)): raise AssertionError("Theta has nonzero diagonal.")

    theta = Zp
    Sg = np.vstack([theta, I]).astype(np.uint8)
    R = Xinv % 2

    # propagate signs through R to fix '+' signs in graph form
    z_cols = Z; x_cols = X
    omega = np.zeros((n, n), dtype=np.uint8)
    for i in range(n):
        zi = z_cols[:, i]; xi = x_cols[:, i]
        for k in range(i+1, n):
            zk = z_cols[:, k]; xk = x_cols[:, k]
            omega[i, k] = (int(zi @ xk) + int(xi @ zk)) & 1

    r_prime = np.zeros(n, dtype=np.uint8)
    for j in range(n):
        c = R[:, j]
        val = int(np.dot(c, r) & 1)
        ones = np.flatnonzero(c)
        for a in range(len(ones)):
            i = ones[a]
            for b in range(a+1, len(ones)):
                k = ones[b]
                val ^= int(omega[min(i,k), max(i,k)])
        r_prime[j] = val & 1

    for j in range(n):
        if r_prime[j]:
            LC_ops[j] += 'Z'

    return theta % 2, R % 2, LC_ops, Sg

# ---------- Graph primitives ----------
def connected_components(theta):
    T = (np.array(theta, dtype=np.uint8) % 2)
    n = T.shape[0]
    deg = T.sum(axis=1)
    nodes = [i for i in range(n) if deg[i] > 0]
    seen, comps = set(), []
    for s in nodes:
        if s in seen: continue
        comp, stack = [], [s]
        seen.add(s)
        while stack:
            u = stack.pop()
            comp.append(u)
            for v in np.where(T[u]==1)[0]:
                if v not in seen:
                    seen.add(v); stack.append(v)
        comps.append(sorted(comp))
    return comps

def local_complement(theta, v):
    T = theta.copy()
    nbrs = np.where(T[v]==1)[0]
    for i in range(len(nbrs)):
        a = nbrs[i]
        for j in range(i+1, len(nbrs)):
            b = nbrs[j]
            T[a,b] ^= 1; T[b,a] ^= 1
    return T

def pivot_on_edge(theta, u, v):
    T = local_complement(theta, u)
    T = local_complement(T, v)
    T = local_complement(T, u)
    return T

def apply_pivot_update_ops(theta, LC_ops, u, v):
    T = theta.copy()
    # LC(u)
    nbrs = np.where(T[u]==1)[0]
    LC_ops[u] += 'HSH'
    for w in nbrs: LC_ops[w] += 'S'
    T = local_complement(T, u)
    # LC(v)
    nbrs = np.where(T[v]==1)[0]
    LC_ops[v] += 'HSH'
    for w in nbrs: LC_ops[w] += 'S'
    T = local_complement(T, v)
    # LC(u)
    nbrs = np.where(T[u]==1)[0]
    LC_ops[u] += 'HSH'
    for w in nbrs: LC_ops[w] += 'S'
    T = local_complement(T, u)
    return T, LC_ops

def is_path_on_nodes(theta, nodes):
    if not nodes: return True, []
    sub = theta[np.ix_(nodes, nodes)]
    if not np.array_equal(sub, sub.T): return False, None
    if np.any(np.diag(sub)): return False, None
    deg = sub.sum(axis=1); m = len(nodes)
    if m == 1: return True, [nodes[0]]
    if sorted(deg.tolist()) != [1,1]+[2]*(m-2): return False, None
    ends = [i for i,d in enumerate(deg) if d==1]
    start = ends[0]; order_loc=[start]; seen={start}
    while len(order_loc)<m:
        cur = order_loc[-1]
        nxt = [t for t in np.where(sub[cur]==1)[0] if t not in seen]
        if not nxt: break
        order_loc.append(nxt[0]); seen.add(nxt[0])
    if len(order_loc)!=m: return False, None
    return True, [nodes[i] for i in order_loc]

def x_measure_on_graph(theta, v):
    T = local_complement(theta, v)
    keep = [i for i in range(T.shape[0]) if i!=v]
    return T[np.ix_(keep, keep)], keep

def prune_to_must_keep(theta, LC_ops, must_keep_original, original_labels=None):
    """
    Deterministically delete (X-measure) every vertex whose ORIGINAL index is not in must_keep_original.
    Returns: (pruned_theta, pruned_ops, survivors_orig)
    """
    n = theta.shape[0]
    if original_labels is None:
        original_labels = list(range(n))
    T = theta.copy()
    ops = LC_ops[:]
    labels = original_labels[:]
    must_keep_set = set(must_keep_original)

    changed = True
    while changed:
        changed = False
        for i_cur in range(T.shape[0]):
            if i_cur >= T.shape[0]:
                break
            if labels[i_cur] not in must_keep_set:
                T, keep = x_measure_on_graph(T, i_cur)
                ops    = [ops[j]    for j in keep]
                labels = [labels[j] for j in keep]
                changed = True
                break
    return T, ops, labels

# ---------- Circuit builders ----------
try:
    from qiskit import QuantumCircuit
    HAVE_QISKIT = True
except Exception:
    HAVE_QISKIT = False

def build_1D_CZ_circuit(theta, order=None):
    if not HAVE_QISKIT:
        return None
    n = theta.shape[0]
    qc = QuantumCircuit(n, name="Prep_|G>_1D")
    for q in range(n):
        qc.h(q)
    if order is not None:
        for i in range(len(order) - 1):
            a, b = order[i], order[i + 1]
            qc.cz(a, b)
        return qc
    # Fallback: general graph (if no order supplied)
    for i in range(n):
        for j in range(i + 1, n):
            if theta[i, j]:
                qc.cz(i, j)
    return qc

def local_ops_to_circuit_dagger(LC_ops):
    if not HAVE_QISKIT:
        return None
    n = len(LC_ops)
    qc = QuantumCircuit(n, name="L^dagger")
    inv_map = {'H': 'H', 'S': 'Sdg', 'Z': 'Z'}
    for q, seq in enumerate(LC_ops):
        inv_seq = [inv_map[g] for g in seq]
        for g in reversed(inv_seq):
            if g == 'H':   qc.h(q)
            elif g == 'Sdg': qc.sdg(q)
            elif g == 'Z':   qc.z(q)
            else: raise ValueError(f"Unknown inverse gate '{g}'")
    return qc

# ---------------------------------- Finding LC-eq Path/Linear Graph state ---------------------------------- 

# ---------- Registers → index sets ----------
def global_indices_for_register(qc, reg_name):
    reg = next(r for r in qc.qregs if r.name == reg_name)
    return [qc.find_bit(q).index for q in reg]

def measured_qubits_in_register(qc, reg_name):
    reg = next(r for r in qc.qregs if r.name == reg_name)
    reg_qubits = set(reg)
    measured = set()
    for inst, qargs, _ in qc.data:
        if inst.name == "measure" and qargs[0] in reg_qubits:
            measured.add(qargs[0])
    return [qc.find_bit(q).index for q in measured]

def build_must_keep(qc, include_A2=True):
    Dprime = set(global_indices_for_register(qc, "D'"))
    K2_all = set(global_indices_for_register(qc, "K2"))
    K2_meas = set(measured_qubits_in_register(qc, "K2"))
    K2_keep = K2_all - K2_meas
    keep = set(Dprime) | set(K2_keep)
    if include_A2:
        A2 = set(global_indices_for_register(qc, "A2"))
        keep |= A2
    return sorted(keep)

# ---------- For Debugging ----------
def print_component_degrees(theta, nodes, survivors_orig=None, label="component"):
    """
    Print degree stats of the induced subgraph on 'nodes'.
      theta           : full adjacency (0/1 np.array)
      nodes           : list of CURRENT indices forming the component
      survivors_orig  : optional list mapping CURRENT index -> ORIGINAL/GLOBAL index
    """
    nodes = list(nodes)
    sub = theta[np.ix_(nodes, nodes)]
    deg = sub.sum(axis=1).astype(int)

    print(f"\nDegrees on {label} (size={len(nodes)}):")
    print(" current idx -> deg:", list(zip(nodes, map(int, deg))))

    if survivors_orig is not None:
        print(" global idx  -> deg:", [(survivors_orig[i], int(d)) for i, d in zip(nodes, deg)])

    cnt = Counter(map(int, deg))
    print(" degree histogram:", dict(sorted(cnt.items())))
    print(" max degree:", int(deg.max()) if len(deg) else 0)
    print(" #endpoints (deg=1):", cnt.get(1, 0))
    print(" #internal  (deg=2):", cnt.get(2, 0))

    endpoints_cur = [nodes[k] for k, d in enumerate(deg) if d == 1]
    print(" endpoints (current):", endpoints_cur)
    if survivors_orig is not None:
        endpoints_glob = [survivors_orig[i] for i in endpoints_cur]
        print(" endpoints (global) :", endpoints_glob)

# ---------- Helpers for IDA ----------
def _excess_and_deg(theta, nodes):
    sub = theta[np.ix_(nodes, nodes)]
    deg = sub.sum(axis=1).astype(int)
    excess = int(np.maximum(0, deg - 2).sum())
    return excess, deg

def _heuristic(theta, nodes):
    excess, _ = _excess_and_deg(theta, nodes)
    return (excess + 1) // 2  # each move removes at most 2 excess

def _encode(theta, nodes):
    sub = theta[np.ix_(nodes, nodes)]
    bits = []
    m = len(nodes)
    for i in range(m):
        for j in range(i+1, m):
            bits.append(sub[i, j])
    return bytes(np.packbits(np.array(bits, dtype=np.uint8)))

def _apply_LC_update(theta, LC_ops, v):
    T = theta.copy()
    nbrs = np.where(T[v]==1)[0]
    LC_ops[v] += 'HSH'
    for u in nbrs: LC_ops[u] += 'S'
    T = local_complement(T, v)
    return T, LC_ops

def ida_star_to_path(theta, LC_ops, nodes, time_limit_sec=180, max_depth=400):
    nodes = list(nodes)
    ok, order = is_path_on_nodes(theta, nodes)
    if ok: return theta, LC_ops, order

    start_h = _heuristic(theta, nodes)
    enc0 = _encode(theta, nodes)
    t0 = time.time()

    bound = start_h
    while bound <= max_depth and (time.time() - t0) < time_limit_sec:
        seen = {enc0: 0}
        stack = [(theta.copy(), LC_ops[:], 0)]
        while stack and (time.time() - t0) < time_limit_sec:
            T, ops, g = stack.pop()
            ok, order = is_path_on_nodes(T, nodes)
            if ok:
                return T, ops, order
            h = _heuristic(T, nodes)
            f = g + h
            if f > bound:
                continue

            sub = T[np.ix_(nodes, nodes)]
            m = len(nodes)

            # Edge pivots first (order by touching high-degree vertices)
            deg = sub.sum(axis=1).astype(int)
            order_u = np.argsort(-deg)
            for i_local in order_u:
                u = nodes[i_local]
                nbrs_local = np.where(sub[i_local]==1)[0]
                nbrs_local = sorted(nbrs_local, key=lambda j: int(sub[j].sum()), reverse=True)
                for j_local in nbrs_local:
                    v = nodes[j_local]
                    T2, ops2 = apply_pivot_update_ops(T, ops[:], u, v)
                    k = _encode(T2, nodes)
                    g2 = g + 1
                    if k in seen and seen[k] <= g2: continue
                    seen[k] = g2
                    stack.append((T2, ops2, g2))

            # Also try a few single-vertex LCs (top-degree vertices)
            for i_local in order_u[:min(4, m)]:
                v = nodes[i_local]
                T2, ops2 = _apply_LC_update(T, ops[:], v)
                k = _encode(T2, nodes)
                g2 = g + 1
                if k in seen and seen[k] <= g2: continue
                seen[k] = g2
                stack.append((T2, ops2, g2))
        bound += 1
    return None, None, None


# ---------------------------------- Check if Stabilizer Group Contains Generator ----------------------------------

# ---------- GF(2) solver ----------
def gf2_solve(A, b):
    """Solve A c = b over GF(2). Return one solution c or None if inconsistent."""
    A = (A % 2).astype(np.uint8)
    b = (b % 2).astype(np.uint8)
    A = A.copy(); b = b.copy()
    n, m = A.shape
    row = 0
    pivcol = [-1]*m
    for col in range(m):
        # find pivot
        piv = None
        for r in range(row, n):
            if A[r, col]:
                piv = r; break
        if piv is None:
            continue
        # swap to 'row'
        if piv != row:
            A[[row,piv]] = A[[piv,row]]
            b[row], b[piv] = b[piv], b[row]
        # eliminate other rows
        for r in range(n):
            if r != row and A[r, col]:
                A[r, :] ^= A[row, :]
                b[r]     ^= b[row]
        pivcol[col] = row
        row += 1
        if row == n: break
    # check consistency
    for r in range(row, n):
        if A[r].sum()==0 and b[r]:
            return None
    # back-substitute: pick zeros for free vars
    c = np.zeros(m, dtype=np.uint8)
    for col in range(m-1, -1, -1):
        r = pivcol[col]
        if r == -1:
            c[col] = 0
        else:
            # A[r,col] == 1; compute rhs minus other cols
            s = b[r]
            for j in range(col+1, m):
                if A[r, j] and c[j]:
                    s ^= 1
            c[col] = s
    return c

# ---------- pauli helpers ----------
def pauli_to_vec(sign_char, pstr):
    """Return (r, a) with r∈{0,1} for sign (+->0, -->1) and a=[z;x]∈GF(2)^{2n}."""
    r = 0 if sign_char == '+' else 1
    z, x = pauli_to_zx(pstr)
    return r, np.concatenate([z, x]).astype(np.uint8)

def gens_to_S_and_signs(gens):
    """gens: list of (sign_char, pstr). Return S (2n×n) with columns as generators, and r (n,) sign bits."""
    n = len(gens)
    Z = np.zeros((n, n), dtype=np.uint8)
    X = np.zeros((n, n), dtype=np.uint8)
    r = np.zeros(n, dtype=np.uint8)
    for j, (sgn, pstr) in enumerate(gens):
        r[j] = 0 if sgn == '+' else 1
        z,x = pauli_to_zx(pstr)
        Z[:, j] = z; X[:, j] = x
    S = np.vstack([Z, X]) % 2
    return S, r

def symplectic_inner(a, b, n):
    """a,b are 2n binary column vectors. Return a^T J b mod 4 (but value is 0 or 2 since entries in {0,1})."""
    z_a, x_a = a[:n], a[n:]
    z_b, x_b = b[:n], b[n:]
    # a^T J b = z_a·x_b - x_a·z_b over integers; mod 4 we only need parity of (z·x + x·z) then map 1->1, 0->0
    # For commuting stabilizer gens this is even; but keep general:
    v = (int(z_a @ x_b) - int(x_a @ z_b))  # integer
    # Reduce to {0,2} mod 4:
    return 2 * (v & 1)

def product_phase_mod4(c, S, r_bits):
    """
    Phase of ∏ g_i^{c_i}: i^p where p in {0,2}. p = 2*sum c_i r_i + sum_{j<k} c_j c_k a_j^T J a_k  (mod 4).
    Return p (0 or 2).
    """
    n = S.shape[0] // 2
    p = 0
    # generator sign contributions (-1)^{r_i} = i^{2 r_i}
    p = (p + 2*int(np.dot(c, r_bits) % 2)) % 4
    # pairwise commutation phase
    for j in range(len(c)):
        if not c[j]: continue
        a_j = S[:, j]
        for k in range(j+1, len(c)):
            if not c[k]: continue
            a_k = S[:, k]
            p = (p + symplectic_inner(a_j, a_k, n)) % 4
    return p  # 0 -> +1; 2 -> -1

def stabilizer_group_contains(target, gens):
    """
    True iff 'target' (sign, pstr) is in the stabilizer group generated by 'gens' with the given sign.
    """
    (r_tar, a_tar) = pauli_to_vec(*target)
    S, r_bits = gens_to_S_and_signs(gens)
    n = S.shape[0] // 2
    # solve S c = a_tar over GF(2)
    c = gf2_solve(S, a_tar)
    if c is None:
        return False
    # compute phase i^p of the product ∏ g_i^{c_i}
    p = product_phase_mod4(c, S, r_bits)   # p ∈ {0,2}
    sign_of_product = 0 if p == 0 else 1   # 0->'+', 1->'-'
    return (sign_of_product == r_tar)

