from config import *
from functools import reduce

def get_dressed_cycles(circuit: tq.Circuit):
    return [tq.Circuit(circuit[2*k:2*k+2]) for k in range(circuit.n_cycles//2 -1)] + [[circuit[-2]]]

def tensor_product_list(tensor_list):
    return reduce(np.kron, tensor_list)


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

def effective_first_dressed_gate(hard_gate: tq.Gate, easy_gate: tq.Gate, noisy_gates: dict):
    
    lst = []
    if hard_gate == Gate.h:
        for p1 in paulis:
            circ = tq.Circuit([{0:p1@noisy_gates[hard_gate]@noisy_gates[correction_gate(hard_gate,p1)@easy_gate]}])
            lst.append(tqm.Superop.from_unitary(tq.Simulator().operator(circuit=circ).mat()).ptm)    

    
    if hard_gate == Gate.t:
        for p1 in easy_gates:
            circ = tq.Circuit([{0:p1@noisy_gates[hard_gate]@noisy_gates[correction_gate(hard_gate,p1)@easy_gate]}])
            lst.append(tqm.Superop.from_unitary(tq.Simulator().operator(circuit=circ).mat()).ptm)
    
    if hard_gate == Gate.cnot:
        for (p1,p2) in two_q_paulis.keys():
            circ = tq.Circuit([{0:two_q_paulis[(p1,p2)]@noisy_gates[hard_gate]@noisy_gates[correction_gate(hard_gate,two_q_paulis[(p1,p2)])@easy_gate]}])
            lst.append(tqm.Superop.from_unitary(tq.Simulator().operator(circuit=circ).mat()).ptm)
    
    return tqm.Superop.from_ptm(np.mean(lst, axis=0)).ptm

def effective_first_dressed_gate_noise(hard_gate: tq.Gate, easy_gate: tq.Gate, noisy_gates: dict):
    
    lst = []
    if hard_gate == Gate.h:
        for p1 in paulis:
            circ = tq.Circuit([{0:p1 @ noisy_gates[hard_gate] @ noisy_gates[correction_gate(hard_gate, p1) @ easy_gate]}])
            lst.append(tqm.Superop.from_unitary(tq.Simulator().operator(circuit=circ).mat()).ptm)

    if hard_gate == Gate.t:
        for p1 in easy_gates:
            circ = tq.Circuit([{0:p1 @ noisy_gates[hard_gate] @ noisy_gates[correction_gate(hard_gate, p1) @ easy_gate]}])
            lst.append(tqm.Superop.from_unitary(tq.Simulator().operator(circuit=circ).mat()).ptm)

    if hard_gate == Gate.cnot:
        for (p1, p2) in two_q_paulis.keys():
            circ = tq.Circuit([{0:two_q_paulis[(p1, p2)] @ noisy_gates[hard_gate] @ noisy_gates[correction_gate(hard_gate, two_q_paulis[(p1, p2)]) @ easy_gate]}])
            lst.append(tqm.Superop.from_unitary(tq.Simulator().operator(circuit=circ).mat()).ptm)

    return tqm.Superop.from_ptm(np.conj(ptm[easy_gate]).T @ np.conj(ptm[hard_gate]).T @ np.mean(lst, axis=0)).ptm

def effective_first_dressed_cycle(dressed_cycle: list, noisy_gates: dict):

    lst = []

    hard_cycle = dressed_cycle[1]
    easy_cycle = dressed_cycle[0]

    for qubit, hard_gate in sorted(hard_cycle.gates.items()):
        
        if hard_gate in [Gate.h, Gate.t]:
            easy_gate = easy_cycle.gates[qubit]
            lst.append(effective_first_dressed_gate(hard_gate, easy_gate, noisy_gates=noisy_gates))

        if hard_gate == Gate.cx:
            easy_gate_0 = easy_cycle.gates[(qubit[0],)]
            easy_gate_1 = easy_cycle.gates[(qubit[1],)]
            lst.append(effective_dressed_gate(hard_gate, tq.Gate(np.kron(easy_gate_0.mat, easy_gate_1.mat)), noisy_gates=noisy_gates))
    
    return tensor_product_list(lst)


def effective_first_dressed_cycle_noise(dressed_cycle: list, noisy_gates: dict):

    lst = []

    hard_cycle = dressed_cycle[1]
    easy_cycle = dressed_cycle[0]

    for qubit, hard_gate in sorted(hard_cycle.gates.items()):

        if hard_gate in [Gate.h, Gate.t]:
            easy_gate = easy_cycle.gates[qubit]
            lst.append(effective_first_dressed_gate_noise(hard_gate, easy_gate, noisy_gates=noisy_gates))

        if hard_gate == Gate.cx:
            easy_gate_0 = easy_cycle.gates[(qubit[0],)]
            easy_gate_1 = easy_cycle.gates[(qubit[1],)]
            lst.append(
                effective_dressed_gate_noise(
                    hard_gate,
                    tq.Gate(np.kron(easy_gate_0.mat, easy_gate_1.mat)),
                    noisy_gates=noisy_gates,
                )
            )

    return tensor_product_list(lst)


def effective_dressed_gate(hard_gate: tq.Gate, easy_gate: tq.Gate, noisy_gates: dict):
    
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
    
    return tqm.Superop.from_ptm(np.mean(lst, axis=0)).ptm

def effective_dressed_gate_noise(hard_gate: tq.Gate, easy_gate: tq.Gate, noisy_gates: dict):
    
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

def effective_dressed_cycle(dressed_cycle: list, noisy_gates: dict):

    lst = []

    hard_cycle = dressed_cycle[1]
    easy_cycle = dressed_cycle[0]

    for qubit, hard_gate in sorted(hard_cycle.gates.items()):
        
        if hard_gate in [Gate.h, Gate.t]:
            easy_gate = easy_cycle.gates[qubit]
            lst.append(effective_dressed_gate(hard_gate, easy_gate, noisy_gates=noisy_gates))

        if hard_gate == Gate.cx:
            easy_gate_0 = easy_cycle.gates[(qubit[0],)]
            easy_gate_1 = easy_cycle.gates[(qubit[1],)]
            lst.append(effective_dressed_gate(hard_gate, tq.Gate(np.kron(easy_gate_0.mat, easy_gate_1.mat)), noisy_gates=noisy_gates))
    
    return tensor_product_list(lst)

def effective_dressed_cycle_noise(dressed_cycle: list, noisy_gates: dict):

    lst = []

    hard_cycle = dressed_cycle[1]
    easy_cycle = dressed_cycle[0]

    for qubit, hard_gate in sorted(hard_cycle.gates.items()):
        
        if hard_gate in [Gate.h, Gate.t]:
            easy_gate = easy_cycle.gates[qubit]
            lst.append(effective_dressed_gate_noise(hard_gate, easy_gate, noisy_gates=noisy_gates))

        if hard_gate == Gate.cx:
            easy_gate_0 = easy_cycle.gates[(qubit[0],)]
            easy_gate_1 = easy_cycle.gates[(qubit[1],)]
            lst.append(effective_dressed_gate_noise(hard_gate, tq.Gate(np.kron(easy_gate_0.mat, easy_gate_1.mat)), noisy_gates=noisy_gates))
    
    return tensor_product_list(lst)


def effective_last_dressed_gate(easy_gate: tq.Gate, noisy_gates: dict):
    
    lst = []

    for p1 in paulis:
        circ = tq.Circuit([{0:noisy_gates[easy_gate @ p1]@p1.adj}])
        lst.append(tqm.Superop.from_unitary(tq.Simulator().operator(circuit=circ).mat()).ptm)    
    return tqm.Superop.from_ptm(np.mean(lst, axis=0)).ptm

def effective_last_dressed_gate_noise(easy_gate: tq.Gate, noisy_gates: dict):
    lst = []
    for p1 in paulis:
        circ = tq.Circuit([{0: noisy_gates[easy_gate @ p1] @ p1.adj}])
        lst.append(
            tqm.Superop.from_unitary(
                tq.Simulator().operator(circuit=circ).mat()
            ).ptm
        )

    return tqm.Superop.from_ptm(
        np.conj(ptm[easy_gate]).T @ np.mean(lst, axis=0)
    ).ptm


def effective_last_dressed_cycle(last_dressed_cycle: list, noisy_gates: dict):
    
    lst = []

    easy_cycle = last_dressed_cycle[0]

    for qubit, easy_gate in sorted(easy_cycle.gates.items()):
        
        easy_gate = easy_cycle.gates[qubit]
        lst.append(effective_last_dressed_gate(easy_gate, noisy_gates=noisy_gates))
    
    return tensor_product_list(lst)

def effective_last_dressed_cycle_noise(last_dressed_cycle: list, noisy_gates: dict):
    lst = []
    easy_cycle = last_dressed_cycle[0]

    for qubit, easy_gate in sorted(easy_cycle.gates.items()):
        easy_gate = easy_cycle.gates[qubit]
        lst.append(
            effective_last_dressed_gate_noise(easy_gate, noisy_gates=noisy_gates)
        )

    return tensor_product_list(lst)

def effective_circuit(circuit: tq.Circuit, noisy_gates: dict):

    dressed_cycles = get_dressed_cycles(circuit)

    lst = []
    
    lst.append(effective_first_dressed_cycle(dressed_cycles[0], noisy_gates=noisy_gates))

    for drs_cycle in dressed_cycles[1:-1]:
        lst.append(effective_dressed_cycle(drs_cycle, noisy_gates=noisy_gates))

    lst.append(effective_last_dressed_cycle(dressed_cycles[-1], noisy_gates=noisy_gates))

    lst = reversed(lst)
    
    return tqm.Superop.from_ptm(reduce(np.matmul, lst))