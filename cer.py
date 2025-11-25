from config import *

def two_qubit_paulis():
    two_paulis = {}
    for p1 in paulis:
        for p2 in paulis:
            two_paulis[(p1,p2)] = tq.Gate(np.kron(p1.mat, p2.mat))
    return two_paulis

def two_qubit_easy_gates():
    two_paulis = {}
    for p1 in easy_gates:
        for p2 in easy_gates:
            two_paulis[(p1,p2)] = tq.Gate(np.kron(p1.mat, p2.mat))
    return two_paulis

two_q_easy_gates = two_qubit_easy_gates()
two_q_paulis = two_qubit_paulis()

def ptms():
    ptm_dict = {}
    for gate in easy_gates:
        ptm_dict[gate] = tqm.Superop.from_unitary(
            tq.Simulator().operator(circuit=tq.Circuit([{0: gate}])).mat()).ptm
    for gate in hard_gates:
        ptm_dict[gate] = tqm.Superop.from_unitary(
            tq.Simulator().operator(circuit=tq.Circuit([{0: gate}])).mat()).ptm
    for (p1, p2), gate in two_q_easy_gates.items():
        ptm_dict[two_q_easy_gates[(p1, p2)]] = tqm.Superop.from_unitary(
            tq.Simulator().operator(circuit=tq.Circuit([{0: gate}])).mat()).ptm
    return ptm_dict

ptm = ptms()

def correction_gate(hard_gate: Gate, twirling_gate: Gate):
    return hard_gate.adj @ twirling_gate.adj @ hard_gate

def noisy_gates_dict(sim: tq.Simulator):
    noisy_gates = {}
    for gate in easy_gates:
        noisy_gates[gate] = tq.Gate(sim.operator(tq.Circuit([{0: gate}])).mat())
    for gate in hard_gates:
        noisy_gates[gate] = tq.Gate(sim.operator(tq.Circuit([{0: gate}])).mat())
    for p1 in easy_gates:
        for p2 in easy_gates:
            noisy_gates[two_q_easy_gates[(p1, p2)]] = tq.Gate(sim.operator(tq.Circuit([{0: p1, 1: p2}])).mat())
    return noisy_gates

def effective_dressed_cycle_noise(hard_gate: tq.Gate, easy_gate: tq.Gate, noisy_gates: dict):
    
    lst = []
    if hard_gate == Gate.h:
        for p1 in paulis:
            for p2 in paulis:
                circ = tq.Circuit([{0:p1@noisy_gates[hard_gate]@noisy_gates[correction_gate(hard_gate,p1)@easy_gate@p2]@p2.adj}])
                lst.append(tqm.Superop.from_unitary(tq.Simulator().operator(circuit=circ).mat()).ptm)    
    
    
    if hard_gate == Gate.t:
        for p1 in easy_gates:
            for p2 in easy_gates:
                circ = tq.Circuit([{0:p1@noisy_gates[hard_gate]@noisy_gates[correction_gate(hard_gate,p1)@easy_gate@p2]@p2.adj}])
                lst.append(tqm.Superop.from_unitary(tq.Simulator().operator(circuit=circ).mat()).ptm)
    
    if hard_gate == Gate.cnot:
        for (p1,p2) in two_q_paulis.keys():
            for (p3,p4) in two_q_paulis.keys():
                circ = tq.Circuit([{0:two_q_paulis[(p1,p2)]@noisy_gates[hard_gate]@noisy_gates[correction_gate(hard_gate,two_q_paulis[(p1,p2)])@easy_gate@two_q_paulis[(p3,p4)]]@two_q_paulis[(p3,p4)].adj}])
                lst.append(tqm.Superop.from_unitary(tq.Simulator().operator(circuit=circ).mat()).ptm)
    
    return tqm.Superop.from_ptm(np.conj(ptm[easy_gate]).T@np.conj(ptm[hard_gate]).T@np.mean(lst, axis=0)).ptm