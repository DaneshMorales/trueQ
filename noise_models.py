import trueq as tq
import trueq.math as tqm
import trueq.simulation as tqs
from config import *



t_gate_rotation = 0.180213
h_gate_rotation = 0.020135
cnot_gate_rotation = 0.073675


# T gate infidelity     = 0.00499996373841
# H gate infidelity     = 0.00099999586597
# CNOT gate infidelity  = 0.01000003672978 

circs = [tq.Circuit({0: i}) for i in easy_gates]

strength = [1,2,3,4]
strength_scales = [0.5, 1.0, 1.5, 10]

def process_fidelity(circuit: tq.Circuit, simulator):
    ideal_unitary_matrix = tq.Simulator().operator(circuit= circuit).upgrade().mat()
    ideal_unitary_superop = tqm.Superop.from_rowstack(ideal_unitary_matrix) 
        
    noisy_unitary_matrix = simulator.operator(circuit= circuit).upgrade().mat()
    noisy_unitary_superop = tqm.Superop.from_rowstack(noisy_unitary_matrix) 


    proc_fid = (ideal_unitary_superop.adj @ noisy_unitary_superop).fidelity 
    return proc_fid
################ Gate independent noise model #####################

gate_ind_angle =  [1.81215 * scale for scale in strength_scales]


def gate_independent_simulators():
    gate_ind_simulators = {}

    # Strength 1
    sim1 = tq.Simulator()
    easy_dict_1 = {gate: tq.Gate.rp("X", 1.81215 * 0.5) @ gate.mat for gate in clifford_gates}
    def gate_replace_1(gate):
        return easy_dict_1[gate]
    noisy_sim_1 = (
        sim1.add_gate_replace(gate_replace_1, match=match_clifford)
            .add_overrotation(single_sys=t_gate_rotation, match=match_t)
            .add_overrotation(single_sys=h_gate_rotation, match=match_h)
            .add_overrotation(single_sys = cnot_gate_rotation, multi_sys=cnot_gate_rotation, match=match_cnot)
    )
    infid_1 = 1 - sum(process_fidelity(circ, noisy_sim_1) for circ in circs) / len(circs)
    gate_ind_simulators[1] = (noisy_sim_1, infid_1)

    # Strength 2
    sim2 = tq.Simulator()
    easy_dict_2 = {gate: tq.Gate.rp("X", 1.81215) @ gate.mat for gate in clifford_gates}
    def gate_replace_2(gate):
        return easy_dict_2[gate]
    noisy_sim_2 = (
        sim2.add_gate_replace(gate_replace_2, match=match_clifford)
            .add_overrotation(single_sys=t_gate_rotation, match=match_t)
            .add_overrotation(single_sys=h_gate_rotation, match=match_h)
            .add_overrotation(single_sys = cnot_gate_rotation, multi_sys=cnot_gate_rotation, match=match_cnot)
    )
    infid_2 = 1 - sum(process_fidelity(circ, noisy_sim_2) for circ in circs) / len(circs)
    gate_ind_simulators[2] = (noisy_sim_2, infid_2)

    # Strength 3
    sim3 = tq.Simulator()
    easy_dict_3 = {gate: tq.Gate.rp("X", 1.81215 * 1.5) @ gate.mat for gate in clifford_gates}
    def gate_replace_3(gate):
        return easy_dict_3[gate]
    noisy_sim_3 = (
        sim3.add_gate_replace(gate_replace_3, match=match_clifford)
            .add_overrotation(single_sys=t_gate_rotation, match=match_t)
            .add_overrotation(single_sys=h_gate_rotation, match=match_h)
            .add_overrotation(single_sys = cnot_gate_rotation, multi_sys=cnot_gate_rotation, match=match_cnot)
    )
    infid_3 = 1 - sum(process_fidelity(circ, noisy_sim_3) for circ in circs) / len(circs)
    gate_ind_simulators[3] = (noisy_sim_3, infid_3)

    # Strength 4
    sim4 = tq.Simulator()
    easy_dict_4 = {gate: tq.Gate.rp("X", 1.81215 * 10) @ gate.mat for gate in clifford_gates}
    def gate_replace_4(gate):
        return easy_dict_4[gate]
    noisy_sim_4 = (
        sim4.add_gate_replace(gate_replace_4, match=match_clifford)
            .add_overrotation(single_sys=t_gate_rotation, match=match_t)
            .add_overrotation(single_sys=h_gate_rotation, match=match_h)
            .add_overrotation(single_sys = cnot_gate_rotation, multi_sys=cnot_gate_rotation, match=match_cnot)
    )
    infid_4 = 1 - sum(process_fidelity(circ, noisy_sim_4) for circ in circs) / len(circs)
    gate_ind_simulators[4] = (noisy_sim_4, infid_4)

    return gate_ind_simulators

sim_gate_ind = gate_independent_simulators()

################ Gate dependent noise model #####################

gate_dep_strengths =  [0.0121405 * scale for scale in strength_scales]


def gate_dependent_simulators():

    gate_dep_simulators = {}
    for label, dep_strength in zip(strength, gate_dep_strengths):

        sim = tq.Simulator()
        
        noisy_sim = (
            sim.add_overrotation(single_sys=dep_strength, match=match_clifford)
               .add_overrotation(single_sys=t_gate_rotation, match=match_t)
               .add_overrotation(single_sys=h_gate_rotation, match=match_h)
               .add_overrotation(single_sys = cnot_gate_rotation, multi_sys=cnot_gate_rotation, match=match_cnot)
        )
        infidelity = 1 - sum([process_fidelity(circ, noisy_sim) for circ in circs]) / len(circs)
        gate_dep_simulators[label] = (noisy_sim, infidelity)
    return gate_dep_simulators

sim_gate_dep = gate_dependent_simulators()

################ ZXZXZ Decomposition noise model #####################

x_rotation_strengths = [0.014236438 * scale for scale in strength_scales]

def zxzxz_simulators():

    zxzxz_dict = {gate: ZXZXZ_decompose(tq.Circuit([{0:gate}])) for gate in clifford_gates}

    circs = [ZXZXZ_decompose(tq.Circuit([{0:easy}])) for easy in easy_gates]
    zxzxz_simulators = {}
    noisy_easy_gates = {}

    for label, x_strength in zip(strength, x_rotation_strengths):

        sim = tq.Simulator()
        
        noisy_sim = (
            sim.add_overrotation(single_sys=x_strength, match=tqs.GateMatch(tq.Gate.sx))
                .add_overrotation(single_sys=t_gate_rotation, match=match_t)
                .add_overrotation(single_sys=h_gate_rotation, match=match_h)
                .add_overrotation(single_sys = cnot_gate_rotation, multi_sys=cnot_gate_rotation, match=match_cnot)
        )
        infidelity = 1 - sum([process_fidelity(circ, noisy_sim) for circ in circs]) / len(circs)

        zxzxz_simulators[label] = (noisy_sim, infidelity)
        noisy_easy_gates[label] = {gate: zxzxz_simulators[label][0].operator(zxzxz_dict[gate]).mat() for gate in clifford_gates}

    def gate_replace_1(gate):
        return tq.Gate(noisy_easy_gates[1][gate])

    def gate_replace_2(gate):
        return tq.Gate(noisy_easy_gates[2][gate])

    def gate_replace_3(gate):
        return tq.Gate(noisy_easy_gates[3][gate])

    def gate_replace_4(gate):
        return tq.Gate(noisy_easy_gates[4][gate])

    sim1 = tq.Simulator()
    noisy_sim_1 = (
            sim1.add_gate_replace(gate_replace_1, match=match_clifford)
                .add_overrotation(single_sys=t_gate_rotation, match=match_t)
                .add_overrotation(single_sys=h_gate_rotation, match=match_h)
                .add_overrotation(single_sys = cnot_gate_rotation, multi_sys=cnot_gate_rotation, match=match_cnot)
        )

    sim2 = tq.Simulator()
    noisy_sim_2 = (
            sim2.add_gate_replace(gate_replace_2, match=match_clifford)
                .add_overrotation(single_sys=t_gate_rotation, match=match_t)
                .add_overrotation(single_sys=h_gate_rotation, match=match_h)
                .add_overrotation(single_sys = cnot_gate_rotation, multi_sys=cnot_gate_rotation, match=match_cnot)
        )

    sim3 = tq.Simulator()
    noisy_sim_3 = (
            sim3.add_gate_replace(gate_replace_3, match=match_clifford)
                .add_overrotation(single_sys=t_gate_rotation, match=match_t)
                .add_overrotation(single_sys=h_gate_rotation, match=match_h)
                .add_overrotation(single_sys = cnot_gate_rotation, multi_sys=cnot_gate_rotation, match=match_cnot)
        )

    sim4 = tq.Simulator()
    noisy_sim_4 = (
            sim4.add_gate_replace(gate_replace_4, match=match_clifford)
                .add_overrotation(single_sys=t_gate_rotation, match=match_t)
                .add_overrotation(single_sys=h_gate_rotation, match=match_h)
                .add_overrotation(single_sys = cnot_gate_rotation, multi_sys=cnot_gate_rotation, match=match_cnot)
        )

    final_sims = {}
    zxzxz_simulators_final = {}

    final_sims[1] = noisy_sim_1
    final_sims[2] = noisy_sim_2
    final_sims[3] = noisy_sim_3
    final_sims[4] = noisy_sim_4

    for label in strength:
        zxzxz_simulators_final[label] = (final_sims[label], zxzxz_simulators[label][1])

    return zxzxz_simulators_final

sim_ZXZXZ = zxzxz_simulators()

################ Twisted ZXZXZ Decomposition #####################

def rotation_about_axis(theta: float, n: np.ndarray) -> np.ndarray:
    """
    Return the 2x2 unitary V = exp(-i * theta/2 * (n · σ)),
    where n is a 3D unit vector and σ = (X, Y, Z).
    """
    n = np.asarray(n, dtype=float)
    n = n / np.linalg.norm(n)  # ensure unit length

    nx, ny, nz = n

    H = nx * Gate.x.mat + ny * Gate.y.mat + nz * Gate.z.mat  # generator n · σ

    c = np.cos(theta / 2.0)
    s = np.sin(theta / 2.0)

    V = c * Gate.i.mat - 1j * s * H
    return V

phi = 0.08  ## RAD, which is approximately 4.58 degrees tilted off the x-axis to the y plane       
n = np.array([np.cos(phi), np.sin(phi), 0.0])
n = n / np.linalg.norm(n)

x_strength_tilted = 0.02236255
x_rotation_strengths_tilted = [x_strength_tilted * scale for scale in strength_scales]

def zxzxz_simulators_tilted():

    zxzxz_dict = {gate: ZXZXZ_decompose(tq.Circuit([{0:gate}])) for gate in clifford_gates}
    zxzxz_simulators = {}
    circs = [ZXZXZ_decompose(tq.Circuit([{0: easy}])) for easy in easy_gates]
    noisy_easy_gates = {}

    # Strength 1 (0.5 * x_strength_tilted)
    sim1 = tq.Simulator()
    V1 = rotation_about_axis(x_strength_tilted * 0.5, n)
    easy_dict_1 = {gate: tq.Gate(V1) @ gate.mat for gate in clifford_gates}
    def gate_replace_1(gate):
        return easy_dict_1[gate]
    noisy_sim_1 = (
        sim1.add_gate_replace(gate_replace_1, match=tqs.GateMatch(tq.Gate.sx))
            .add_overrotation(single_sys=t_gate_rotation, match=match_t)
            .add_overrotation(single_sys=h_gate_rotation, match=match_h)
            .add_overrotation(single_sys = cnot_gate_rotation, multi_sys=cnot_gate_rotation, match=match_cnot)
    )
    infid_1 = 1 - sum(process_fidelity(circ, noisy_sim_1) for circ in circs) / len(circs)
    zxzxz_simulators[1] = (noisy_sim_1, infid_1)

    # Strength 2 (1.0 * x_strength_tilted)
    sim2 = tq.Simulator()
    V2 = rotation_about_axis(x_strength_tilted * 1.0, n)
    easy_dict_2 = {gate: tq.Gate(V2) @ gate.mat for gate in clifford_gates}
    def gate_replace_2(gate):
        return easy_dict_2[gate]
    noisy_sim_2 = (
        sim2.add_gate_replace(gate_replace_2, match=tqs.GateMatch(tq.Gate.sx))
            .add_overrotation(single_sys=t_gate_rotation, match=match_t)
            .add_overrotation(single_sys=h_gate_rotation, match=match_h)
            .add_overrotation(single_sys = cnot_gate_rotation, multi_sys=cnot_gate_rotation, match=match_cnot)
    )
    infid_2 = 1 - sum(process_fidelity(circ, noisy_sim_2) for circ in circs) / len(circs)
    zxzxz_simulators[2] = (noisy_sim_2, infid_2)

    # Strength 3 (1.5 * x_strength_tilted)
    sim3 = tq.Simulator()
    V3 = rotation_about_axis(x_strength_tilted * 1.5, n)
    easy_dict_3 = {gate: tq.Gate(V3) @ gate.mat for gate in clifford_gates}
    def gate_replace_3(gate):
        return easy_dict_3[gate]
    noisy_sim_3 = (
        sim3.add_gate_replace(gate_replace_3, match=tqs.GateMatch(tq.Gate.sx))
            .add_overrotation(single_sys=t_gate_rotation, match=match_t)
            .add_overrotation(single_sys=h_gate_rotation, match=match_h)
            .add_overrotation(single_sys = cnot_gate_rotation, multi_sys=cnot_gate_rotation, match=match_cnot)
    )
    infid_3 = 1 - sum(process_fidelity(circ, noisy_sim_3) for circ in circs) / len(circs)
    zxzxz_simulators[3] = (noisy_sim_3, infid_3)

    # Strength 4 (2.0 * x_strength_tilted)
    sim4 = tq.Simulator()
    V4 = rotation_about_axis(x_strength_tilted * 2.0, n)
    easy_dict_4 = {gate: tq.Gate(V4) @ gate.mat for gate in clifford_gates}
    def gate_replace_4(gate):
        return easy_dict_4[gate]
    noisy_sim_4 = (
        sim4.add_gate_replace(gate_replace_4, match=tqs.GateMatch(tq.Gate.sx))
            .add_overrotation(single_sys=t_gate_rotation, match=match_t)
            .add_overrotation(single_sys=h_gate_rotation, match=match_h)
            .add_overrotation(single_sys = cnot_gate_rotation, multi_sys=cnot_gate_rotation, match=match_cnot)
    )
    infid_4 = 1 - sum(process_fidelity(circ, noisy_sim_4) for circ in circs) / len(circs)
    zxzxz_simulators[4] = (noisy_sim_4, infid_4)

    for label in strength:
        noisy_easy_gates[label] = {gate: zxzxz_simulators[label][0].operator(zxzxz_dict[gate]).mat() for gate in clifford_gates}

    def gate_replace_1_tilted(gate):
        return tq.Gate(noisy_easy_gates[1][gate])

    def gate_replace_2_tilted(gate):
        return tq.Gate(noisy_easy_gates[2][gate])

    def gate_replace_3_tilted(gate):
        return tq.Gate(noisy_easy_gates[3][gate])

    def gate_replace_4_tilted(gate):
        return tq.Gate(noisy_easy_gates[4][gate])

    sim1 = tq.Simulator()
    noisy_sim_1 = (
            sim1.add_gate_replace(gate_replace_1_tilted, match=match_clifford)
                .add_overrotation(single_sys=t_gate_rotation, match=match_t)
                .add_overrotation(single_sys=h_gate_rotation, match=match_h)
                .add_overrotation(single_sys = cnot_gate_rotation, multi_sys=cnot_gate_rotation, match=match_cnot)
        )

    sim2 = tq.Simulator()
    noisy_sim_2 = (
            sim2.add_gate_replace(gate_replace_2_tilted, match=match_clifford)
                .add_overrotation(single_sys=t_gate_rotation, match=match_t)
                .add_overrotation(single_sys=h_gate_rotation, match=match_h)
                .add_overrotation(single_sys = cnot_gate_rotation, multi_sys=cnot_gate_rotation, match=match_cnot)
        )

    sim3 = tq.Simulator()
    noisy_sim_3 = (
            sim3.add_gate_replace(gate_replace_3_tilted, match=match_clifford)
                .add_overrotation(single_sys=t_gate_rotation, match=match_t)
                .add_overrotation(single_sys=h_gate_rotation, match=match_h)
                .add_overrotation(single_sys = cnot_gate_rotation, multi_sys=cnot_gate_rotation, match=match_cnot)
        )

    sim4 = tq.Simulator()
    noisy_sim_4 = (
            sim4.add_gate_replace(gate_replace_4_tilted, match=match_clifford)
                .add_overrotation(single_sys=t_gate_rotation, match=match_t)
                .add_overrotation(single_sys=h_gate_rotation, match=match_h)
                .add_overrotation(single_sys = cnot_gate_rotation, multi_sys=cnot_gate_rotation, match=match_cnot)
        )

    final_sims = {}
    zxzxz_simulators_final = {}

    final_sims[1] = noisy_sim_1
    final_sims[2] = noisy_sim_2
    final_sims[3] = noisy_sim_3
    final_sims[4] = noisy_sim_4

    for label in strength:
        zxzxz_simulators_final[label] = (final_sims[label], zxzxz_simulators[label][1])

    return zxzxz_simulators_final

sim_ZXZXZ_tilted = zxzxz_simulators_tilted()


   
