import trueq as tq
import trueq.simulation as tqs
import trueq.math as tqm
from trueq import Gate
import random
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from scipy.linalg import polar
################ DEFINE GATE SETS AND MATCHES #####################
non_paulis = [tq.Gate.s, tq.Gate.cliff8, tq.Gate.cliff10, tq.Gate.cliff11]
paulis = [tq.Gate.x, tq.Gate.y, tq.Gate.z, tq.Gate.id]
clifford_gates = [
    tq.Gate.cliff0, tq.Gate.cliff1, tq.Gate.cliff2, tq.Gate.cliff3,
    tq.Gate.cliff4, tq.Gate.cliff5, tq.Gate.cliff6, tq.Gate.cliff7,
    tq.Gate.cliff8, tq.Gate.cliff9, tq.Gate.cliff10, tq.Gate.cliff11,
    tq.Gate.cliff13, tq.Gate.cliff14, tq.Gate.cliff15,
    tq.Gate.cliff16, tq.Gate.cliff17, tq.Gate.cliff18, tq.Gate.cliff19,
    tq.Gate.cliff20, tq.Gate.cliff21, tq.Gate.cliff22, tq.Gate.cliff23
]

easy_gates = paulis + non_paulis
one_qubit_hard_gates = [tq.Gate.t, tq.Gate.h]
multi_qubit_hard_gates = [tq.Gate.cx]
hard_gates = multi_qubit_hard_gates + one_qubit_hard_gates

pauli_cycles = [tq.Cycle({0:pauli}) for pauli in paulis]
dihedral_cycles = [tq.Cycle({0:easy}) for easy in easy_gates]
two_pauli_cycles = [tq.Cycle({0:pauli1, 1:pauli2}) for pauli1 in paulis for pauli2 in paulis]
match_i = tqs.GateMatch(tq.Gate.i)
match_t = tqs.GateMatch(tq.Gate.t)
match_h = tqs.GateMatch(tq.Gate.h)
match_easy_gates = tqs.GateMatch(easy_gates)
match_cnot= tqs.GateMatch(Gate.cx)
match_clifford = tqs.GateMatch(clifford_gates)

################ PROCESS FIDELITY CALCULATION #####################

h_circs = [tq.Circuit([{0:easy},{0:tq.Gate.h}]) for easy in paulis]
t_circs = [tq.Circuit([{0:easy},{0:tq.Gate.t}]) for easy in easy_gates]
i_circs = [tq.Circuit([{0:easy}]) for easy in paulis]
cnot_circs = [tq.Circuit([{0:easy_1, 1:easy_2},{(0,1):tq.Gate.cx}]) for easy_1 in paulis for easy_2 in paulis]

circs = {Gate.h: h_circs, Gate.t: t_circs, Gate.i: i_circs, Gate.cx: cnot_circs}

def get_hard_cycles(circuit: tq.Circuit):

    return [circuit[2*k+1] for k in range(circuit.n_cycles//2 -1)]

def get_dressed_cycles(circuit: tq.Circuit):

    return [circuit[2*k+1] for k in range(circuit.n_cycles//2 -1)]


def process_fidelity(circuit: tq.Circuit, simulator):
    ideal_unitary_matrix = tq.Simulator().operator(circuit= circuit).upgrade().mat()
    ideal_unitary_superop = tqm.Superop.from_rowstack(ideal_unitary_matrix) 
        
    noisy_unitary_matrix = simulator.operator(circuit= circuit).upgrade().mat()
    noisy_unitary_superop = tqm.Superop.from_rowstack(noisy_unitary_matrix) 


    proc_fid = (ideal_unitary_superop.adj @ noisy_unitary_superop).fidelity 
    return proc_fid


def hard_gates_fid(noisy_sim):
    dic = {}
    cnot_fid = sum([process_fidelity(circ, noisy_sim) for circ in cnot_circs]) / len(cnot_circs)
    t_fid = sum([process_fidelity(circ, noisy_sim) for circ in t_circs]) / len(t_circs)
    h_fid = sum([process_fidelity(circ, noisy_sim) for circ in h_circs]) / len(h_circs)
    i_fid = sum([process_fidelity(circ, noisy_sim) for circ in i_circs]) / len(i_circs)
    dic[tq.Gate.cx] = cnot_fid
    dic[tq.Gate.t] = t_fid
    dic[tq.Gate.h] = h_fid
    dic[tq.Gate.id] = i_fid
    return dic

################ GATE AVERAGING AND GAUGE OPTIMIZATION #####################

def average_gate_set(hard_gate, noisy_sim):

    def hard_cycle(hard_gate):
        if hard_gate == Gate.h:
            return tq.Cycle({0: Gate.h})
        if hard_gate == Gate.t:
            return tq.Cycle({0: Gate.t})
        if hard_gate == Gate.cx:
            return tq.Cycle({(0,1): Gate.cx})
        if hard_gate == Gate.i:
            return tq.Cycle({0: Gate.i})

        else:
            return None
    
    randomizing_cycles = pauli_cycles if hard_gate == Gate.h or hard_gate == Gate.i else dihedral_cycles if hard_gate == Gate.t else two_pauli_cycles

    lst = []

    if hard_gate == Gate.i:
        for r_1 in randomizing_cycles:
            for r_2 in randomizing_cycles:
                for r_3 in randomizing_cycles:
                    for r_4 in randomizing_cycles:
                        circ = tq.Circuit([r_1, r_2, r_3, r_4])
                        ideal_PTM = tqm.Superop.from_rowstack(tq.Simulator().operator(circuit=circ).upgrade().mat()).ptm
                        noisy_PTM = tqm.Superop.from_rowstack(noisy_sim.operator(circuit=circ).upgrade().mat()).ptm
                        mat = ideal_PTM.conj().T @ noisy_PTM
                        lst.append(mat)
    
    else:

        for r_1 in randomizing_cycles:
            for r_2 in randomizing_cycles:
                for r_3 in randomizing_cycles:
                    for r_4 in randomizing_cycles:
                        circ = tq.Circuit([r_1, hard_cycle(hard_gate), r_2, hard_cycle(hard_gate), r_3, hard_cycle(hard_gate), r_4, hard_cycle(hard_gate)])
                        ideal_PTM = tqm.Superop.from_rowstack(tq.Simulator().operator(circuit=circ).upgrade().mat()).ptm
                        noisy_PTM = tqm.Superop.from_rowstack(noisy_sim.operator(circuit=circ).upgrade().mat()).ptm
                        mat = ideal_PTM.conj().T @ noisy_PTM
                        mat[np.abs(mat) < 1e-6] = 0
                        lst.append(mat)
    
    avg =  sum(lst) / len(lst)
    return avg

def eigenvector_closest_to_one(A):
    # Compute eigenvalues and eigenvectors
    vals, vecs = np.linalg.eig(A)

    # Find index of eigenvalue closest to 1
    idx = np.argmin(np.abs(vals - 1))

    # Corresponding eigenvalue & eigenvector
    eigenvalue = vals[idx]
    eigenvector = vecs[:, idx]

    return eigenvalue, eigenvector

def unvec(matrix: np.array):
    n = int(np.sqrt(np.shape(matrix)[0]))
    return matrix.T.reshape(n,n)

def unitary_gauge(hard_gate, noisy_sim):
    m = 4 if hard_gate == Gate.cx else 2
    avg = average_gate_set(hard_gate, noisy_sim)
    choi = tqm.Superop.from_ptm(avg).choi/m
    eigenval, eigenvec = eigenvector_closest_to_one(choi)
    print(eigenval)
    u, p = polar(unvec(eigenvec*np.sqrt(m)))
    return u

def gauge_unitaries(noisy_sim):
    dic = {}
    gauge_CNOT = unitary_gauge(Gate.cx,noisy_sim)
    print("CNOT gauge computed")
    gauge_H = unitary_gauge(Gate.h,noisy_sim)
    print("H gauge computed")
    gauge_I = unitary_gauge(Gate.i,noisy_sim)
    print("I gauge computed")
    gauge_T = unitary_gauge(Gate.t,noisy_sim)
    print("T gauge computed")
    dic[tq.Gate.cx] = gauge_CNOT
    dic[tq.Gate.h] = gauge_H
    dic[tq.Gate.id] = gauge_I
    dic[tq.Gate.t] = gauge_T
    return dic

def process_fidelity_gauge(simulator):
    dic = {}
    gauge = gauge_unitaries(simulator)
    h_gauge_ptm = tqm.Superop.from_unitary(gauge[tq.Gate.h]).ptm
    cnot_gauge_ptm = tqm.Superop.from_unitary(gauge[tq.Gate.cx]).ptm
    i_gauge_ptm = tqm.Superop.from_unitary(gauge[tq.Gate.id]).ptm
    t_gauge_ptm = tqm.Superop.from_unitary(gauge[tq.Gate.t]).ptm
    gauge_ptms = {tq.Gate.h: h_gauge_ptm, tq.Gate.t: t_gauge_ptm, tq.Gate.cx: cnot_gauge_ptm, tq.Gate.id: i_gauge_ptm}
    for gate in gauge_ptms.keys():
        fidelities = []
        for circuit in circs[gate]:
            ideal_unitary_matrix = tq.Simulator().operator(circuit=circuit).upgrade().mat()
            ideal_unitary_superop = tqm.Superop.from_rowstack(ideal_unitary_matrix) 
            ideal_unitary_ptm = ideal_unitary_superop.ptm
            ideal_gauged_ptm = gauge_ptms[gate].conj().T @ ideal_unitary_ptm @ gauge_ptms[gate]
            ideal_gauged_superop = tqm.Superop.from_ptm(ideal_gauged_ptm)
            noisy_unitary_matrix = simulator.operator(circuit=circuit).upgrade().mat()
            noisy_unitary_superop = tqm.Superop.from_rowstack(noisy_unitary_matrix) 
            proc_fid = (ideal_gauged_superop.adj @ noisy_unitary_superop).fidelity 
            fidelities.append(proc_fid)
        dic[gate] = sum(fidelities) / len(fidelities)
    return dic, gauge

def gauge_fidelity(gauge: dict):
    dic = {}
    for gate in gauge.keys():
        dic[gate] = tqm.Superop.from_unitary(gauge[gate]).fidelity
    return dic

## TEST DECAYS ON CNOT, H and T
h_circ = tq.Cycle({0: Gate.h})
t_circ = tq.Cycle({0: Gate.t})
cnot_circ = tq.Cycle({(0,1): Gate.cx})
i_circ = tq.Cycle({0: Gate.i})
################### CYCLE BENCHMARKING #####################

def calculate_process_fidelity(circuits: tq.CircuitCollection):

    fit = circuits.fit()
    return 1- fit.array('e_F', 'labels').vals[0]

def run_cb_on_cycle(cycle: tq.Cycle, n_decays, n_randomizations, m_vals, noisy_sim):
    
    cb_circuits = tq.make_cb(cycles= cycle, n_circuits= n_randomizations, n_random_cycles= m_vals, n_decays=n_decays)
    noisy_sim.run(cb_circuits, n_shots=np.inf)
    return calculate_process_fidelity(cb_circuits)

def run_cb_on_cycle_identity(n_randomizations, m_vals, noisy_sim):
    
    cb_circuits = tq.make_cb(cycles= {}, n_circuits= n_randomizations, n_random_cycles= m_vals, n_decays=4, twirl=tq.Twirl('P',0))
    noisy_sim.run(cb_circuits, n_shots=np.inf)
    return calculate_process_fidelity(cb_circuits)

def cycle_benchmarking_estimators(noisy_sim):
    print("Starting Cycle Benchmarking Estimators...")
    dic = {}

    # Run CNOT gate CB 10 times
    cnot_results = []
    print("Running CNOT gate CB 10 times...")
    for i in range(10):
        print(f"  Run {i+1}/10...")
        cnot_cb_result = run_cb_on_cycle(cnot_circ, n_decays=16, n_randomizations=1000, m_vals=[4,20,40,80], noisy_sim=noisy_sim)
        cnot_results.append(cnot_cb_result)
        

    cnot_average = np.mean(cnot_results)
    cnot_std = np.std(cnot_results)
    print(f"\nCNOT gate CB results: {cnot_results}")
    print(f"CNOT gate CB average infidelity: {cnot_average}")
    print(f"CNOT gate CB std dev: {cnot_std}")
    dic[Gate.cx] = cnot_average

    # Run H gate CB 10 times
    h_results = []
    print("Running H gate CB 10 times...")
    for i in range(10):
        print(f"  Run {i+1}/10...")
        h_cb_result = run_cb_on_cycle(h_circ, n_decays=4, n_randomizations=1000, m_vals=[4,40,80,120], noisy_sim=noisy_sim)
        h_results.append(h_cb_result)

    h_average = np.mean(h_results)
    h_std = np.std(h_results)
    print(f"\nH gate CB results: {h_results}")
    print(f"H gate CB average fidelity: {h_average}")
    print(f"H gate CB std dev: {h_std}")
    dic[Gate.h] = h_average

    # Run T gate CB 10 times
    t_results = []
    print("Running T gate CB 10 times...")

    for i in range(10):
        print(f"  Run {i+1}/10...")
        t_cb_result = run_cb_on_cycle(t_circ, n_decays=4, n_randomizations=1000, m_vals=[8,40,80,120], noisy_sim=noisy_sim)
        t_results.append(t_cb_result)

    t_average = np.mean(t_results)
    t_std = np.std(t_results)
    print(f"\nT gate CB results: {t_results}")
    print(f"T gate CB average infidelity: {t_average}")
    print(f"T gate CB std dev: {t_std}")
    dic[Gate.t] = t_average

    # Run I gate CB 10 times
    i_results = []
    print("Running I gate CB 10 times...")
    for i in range(10):
        print(f"  Run {i+1}/10...")
        i_cb_result = run_cb_on_cycle_identity(n_randomizations=1000, m_vals=[4,40,80,120], noisy_sim=noisy_sim)
        i_results.append(i_cb_result)

    i_average = np.mean(i_results)
    i_std = np.std(i_results)
    print(f"\nI gate CB results: {i_results}")
    print(f"I gate CB average infidelity: {i_average}")
    print(f"I gate CB std dev: {i_std}")
    dic[Gate.i] = i_average

    return dic

################ ZXZXZ Decomposition #####################

config = tq.Config.from_yaml(
    """
    Mode: ZXZXZ
    Gates:
      - Z:
          Hamiltonian:
          - ['Z', 'phi']
      - X90:
          Hamiltonian:
          - ['X', 90]
      - CNOT:
          Matrix:
          - [1, 0, 0, 0]
          - [0, 1, 0, 0]
          - [0, 0, 0, 1]
          - [0, 0, 1, 0]
    """
)


transpiler = tq.Compiler.from_config(config)

def ZXZXZ_decompose(circ):
    lst = []
    key = circ.key
    hard_cycles = get_hard_cycles(circ)
    n = len(hard_cycles)
    compiled_circ = transpiler.compile(circ)
    for i in range(n):
        for easy_cycle in compiled_circ[0+10*i:5+10*i]:
            lst.append(easy_cycle)
        lst.append(hard_cycles[i])
    for last_easy_cycle in compiled_circ[0+10*n: 5+10*n]:
        lst.append(last_easy_cycle)
    
    final = tq.Circuit(lst, key=key)
    final.measure_all()
    return final

def ZXZXZ_decompose3(circ):
    lst = []
    key = circ.key
    hard_cycles = get_hard_cycles(circ)
    n = len(hard_cycles)
    compiled_circ = transpiler.compile(circ)
    for i in range(n):
        for easy_cycle in compiled_circ[0+10*i:5+10*i]:
            lst.append(easy_cycle)
        lst.append(hard_cycles[i])
    for last_easy_cycle in compiled_circ[0+10*n: 5+10*n]:
        lst.append(last_easy_cycle)
    
    final = tq.Circuit(lst, key=key)
    return final

def hard_gates_fid_ZXZXZ(noisy_sim):
    dic = {}
    cnot_fid = sum([process_fidelity(ZXZXZ_decompose3(circ).append({(0,1):tq.Gate.cx}), noisy_sim) for circ in cnot_circs]) / len(cnot_circs)
    t_fid = sum([process_fidelity(ZXZXZ_decompose3(circ).append({0:tq.Gate.t}), noisy_sim) for circ in t_circs]) / len(t_circs)
    h_fid = sum([process_fidelity(ZXZXZ_decompose3(circ).append({0:tq.Gate.h}), noisy_sim) for circ in h_circs]) / len(h_circs)
    i_fid = sum([process_fidelity(ZXZXZ_decompose3(circ).append({0:tq.Gate.i}), noisy_sim) for circ in i_circs]) / len(i_circs)
    dic[tq.Gate.cx] = cnot_fid
    dic[tq.Gate.t] = t_fid
    dic[tq.Gate.h] = h_fid
    dic[tq.Gate.id] = i_fid
    return dic

def ZXZXZ_decompose_collection(circ_collection):
    decomposed_collection = []
    for circ in circ_collection:
        decomposed_collection.append(ZXZXZ_decompose(circ))
    return tq.CircuitCollection(decomposed_collection)

def run_cb_on_cycleZXZXZ(cycle: tq.Cycle, n_decays, n_randomizations, m_vals, noisy_sim):
    
    cb_circuits = tq.make_cb(cycles= cycle, n_circuits= n_randomizations, n_random_cycles= m_vals, n_decays=n_decays)
    cb_circuits = ZXZXZ_decompose_collection(cb_circuits)
    noisy_sim.run(cb_circuits, n_shots=np.inf)
    return calculate_process_fidelity(cb_circuits)

def run_cb_on_cycleZXZXZ_identity(n_randomizations, m_vals, noisy_sim):
    
    cb_circuits = tq.make_cb(cycles={}, n_circuits= n_randomizations, n_random_cycles= m_vals, n_decays=4, twirl=tq.Twirl('P',0))
    cb_circuits = ZXZXZ_decompose_collection(cb_circuits)
    noisy_sim.run(cb_circuits, n_shots=np.inf)
    return calculate_process_fidelity(cb_circuits)

def cycle_benchmarking_estimators_ZXZXZ(noisy_sim):
    print("Starting Cycle Benchmarking Estimators for the ZXZXZ decomposition...")
    dic = {}
    # Run CNOT gate CB 10 times
    cnot_results = []
    print("Running CNOT gate CB 10 times...")
    for i in range(10):
        print(f"  Run {i+1}/10...")
        cnot_cb_result = run_cb_on_cycleZXZXZ(cnot_circ, n_decays=16, n_randomizations=1000, m_vals=[4,20,40,80], noisy_sim=noisy_sim)
        cnot_results.append(cnot_cb_result)
        

    cnot_average = np.mean(cnot_results)
    cnot_std = np.std(cnot_results)
    print(f"\nCNOT gate CB results: {cnot_results}")
    print(f"CNOT gate CB average infidelity: {cnot_average}")
    print(f"CNOT gate CB std dev: {cnot_std}")
    dic[Gate.cx] = cnot_average

    # Run H gate CB 10 times
    h_results = []
    print("Running H gate CB 10 times...")
    for i in range(10):
        print(f"  Run {i+1}/10...")
        h_cb_result = run_cb_on_cycleZXZXZ(h_circ, n_decays=4, n_randomizations=1000, m_vals=[4,40,80,120], noisy_sim=noisy_sim)
        h_results.append(h_cb_result)

    h_average = np.mean(h_results)
    h_std = np.std(h_results)
    print(f"\nH gate CB results: {h_results}")
    print(f"H gate CB average fidelity: {h_average}")
    print(f"H gate CB std dev: {h_std}")
    dic[Gate.h] = h_average

    # Run T gate CB 10 times
    t_results = []
    print("Running T gate CB 10 times...")

    for i in range(10):
        print(f"  Run {i+1}/10...")
        t_cb_result = run_cb_on_cycleZXZXZ(t_circ, n_decays=4, n_randomizations=1000, m_vals=[8,40,80,120], noisy_sim=noisy_sim)
        t_results.append(t_cb_result)

    t_average = np.mean(t_results)
    t_std = np.std(t_results)
    print(f"\nT gate CB results: {t_results}")
    print(f"T gate CB average infidelity: {t_average}")
    print(f"T gate CB std dev: {t_std}")
    dic[Gate.t] = t_average

    # Run I gate CB 10 times
    i_results = []
    print("Running I gate CB 10 times...")
    for i in range(10):
        print(f"  Run {i+1}/10...")
        i_cb_result = run_cb_on_cycleZXZXZ_identity(n_decays=4, n_randomizations=1000, m_vals=[4,40,80,120], noisy_sim=noisy_sim)
        i_results.append(i_cb_result)

    i_average = np.mean(i_results)
    i_std = np.std(i_results)
    print(f"\nI gate CB results: {i_results}")
    print(f"I gate CB average infidelity: {i_average}")
    print(f"I gate CB std dev: {i_std}")
    dic[Gate.i] = i_average

    return dic


def average_gate_set_ZXZXZ(hard_gate, noisy_sim):

    def hard_cycle(hard_gate):
        if hard_gate == Gate.h:
            return tq.Cycle({0: Gate.h})
        if hard_gate == Gate.t:
            return tq.Cycle({0: Gate.t})
        if hard_gate == Gate.cx:
            return tq.Cycle({(0,1): Gate.cx})
        if hard_gate == Gate.i:
            return tq.Cycle({0: Gate.i})

        else:
            return None
    
    randomizing_cycles = pauli_cycles if hard_gate == Gate.h or hard_gate == Gate.i else dihedral_cycles if hard_gate == Gate.t else two_pauli_cycles

    lst = []

    if hard_gate == Gate.i:
        for r_1 in randomizing_cycles:
            for r_2 in randomizing_cycles:
                for r_3 in randomizing_cycles:
                    for r_4 in randomizing_cycles:
                        circ = tq.Circuit([r_1, r_2, r_3, r_4])
                        circ = ZXZXZ_decompose3(circ)
                        ideal_PTM = tqm.Superop.from_rowstack(tq.Simulator().operator(circuit=circ).upgrade().mat()).ptm
                        noisy_PTM = tqm.Superop.from_rowstack(noisy_sim.operator(circuit=circ).upgrade().mat()).ptm
                        mat = ideal_PTM.conj().T @ noisy_PTM
                        lst.append(mat)
    
    else:

        for r_1 in randomizing_cycles:
            for r_2 in randomizing_cycles:
                for r_3 in randomizing_cycles:
                    for r_4 in randomizing_cycles:
                        circ = tq.Circuit([r_1, hard_cycle(hard_gate), r_2, hard_cycle(hard_gate), r_3, hard_cycle(hard_gate), r_4, hard_cycle(hard_gate)])
                        circ = ZXZXZ_decompose3(circ)
                        ideal_PTM = tqm.Superop.from_rowstack(tq.Simulator().operator(circuit=circ).upgrade().mat()).ptm
                        noisy_PTM = tqm.Superop.from_rowstack(noisy_sim.operator(circuit=circ).upgrade().mat()).ptm
                        mat = ideal_PTM.conj().T @ noisy_PTM
                        mat[np.abs(mat) < 1e-6] = 0
                        lst.append(mat)
    
    avg =  sum(lst) / len(lst)
    return avg


def unitary_gauge_ZXZXZ(hard_gate, noisy_sim):
    m = 4 if hard_gate == Gate.cx else 2
    avg = average_gate_set_ZXZXZ(hard_gate, noisy_sim)
    choi = tqm.Superop.from_ptm(avg).choi/m
    eigenval, eigenvec = eigenvector_closest_to_one(choi)
    print(eigenval)
    u, p = polar(unvec(eigenvec*np.sqrt(m)))
    return u

def gauge_unitaries_ZXZXZ(noisy_sim):
    dic = {}
    gauge_CNOT = unitary_gauge_ZXZXZ(Gate.cx,noisy_sim)
    print("CNOT gauge computed")
    gauge_H = unitary_gauge_ZXZXZ(Gate.h,noisy_sim)
    print("H gauge computed")
    gauge_I = unitary_gauge_ZXZXZ(Gate.i,noisy_sim)
    print("I gauge computed")
    gauge_T = unitary_gauge_ZXZXZ(Gate.t,noisy_sim)
    print("T gauge computed")
    dic[tq.Gate.cx] = gauge_CNOT
    dic[tq.Gate.h] = gauge_H
    dic[tq.Gate.id] = gauge_I
    dic[tq.Gate.t] = gauge_T
    return dic

def process_fidelity_gauge_ZXZXZ(simulator):
    dic = {}
    gauge = gauge_unitaries_ZXZXZ(simulator)
    h_gauge_ptm = tqm.Superop.from_unitary(gauge[tq.Gate.h]).ptm
    cnot_gauge_ptm = tqm.Superop.from_unitary(gauge[tq.Gate.cx]).ptm
    i_gauge_ptm = tqm.Superop.from_unitary(gauge[tq.Gate.id]).ptm
    t_gauge_ptm = tqm.Superop.from_unitary(gauge[tq.Gate.t]).ptm
    gauge_ptms = {tq.Gate.h: h_gauge_ptm, tq.Gate.t: t_gauge_ptm, tq.Gate.cx: cnot_gauge_ptm, tq.Gate.id: i_gauge_ptm}
    
    # Map gates to their append syntax
    gate_appends = {
        tq.Gate.h: lambda circ: ZXZXZ_decompose3(circ).append({0: tq.Gate.h}),
        tq.Gate.t: lambda circ: ZXZXZ_decompose3(circ).append({0: tq.Gate.t}),
        tq.Gate.cx: lambda circ: ZXZXZ_decompose3(circ).append({(0,1): tq.Gate.cx}),
        tq.Gate.id: lambda circ: ZXZXZ_decompose3(circ).append({0: tq.Gate.i})
    }
    
    for gate in gauge_ptms.keys():
        fidelities = []
        for circuit in circs[gate]:
            # Decompose and append hard gate
            full_circuit = gate_appends[gate](circuit)
            
            ideal_unitary_matrix = tq.Simulator().operator(circuit=full_circuit).upgrade().mat()
            ideal_unitary_superop = tqm.Superop.from_rowstack(ideal_unitary_matrix) 
            ideal_unitary_ptm = ideal_unitary_superop.ptm
            ideal_gauged_ptm = gauge_ptms[gate].conj().T @ ideal_unitary_ptm @ gauge_ptms[gate]
            ideal_gauged_superop = tqm.Superop.from_ptm(ideal_gauged_ptm)
            noisy_unitary_matrix = simulator.operator(circuit=full_circuit).upgrade().mat()
            noisy_unitary_superop = tqm.Superop.from_rowstack(noisy_unitary_matrix) 
            proc_fid = (ideal_gauged_superop.adj @ noisy_unitary_superop).fidelity 
            fidelities.append(proc_fid)
        dic[gate] = sum(fidelities) / len(fidelities)
    return dic, gauge