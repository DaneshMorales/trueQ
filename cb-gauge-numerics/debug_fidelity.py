
import trueq as tq
import trueq.math as tqm
import trueq.simulation as tqs
from config import *
from noise_models import gate_dependent_simulators, gate_independent_simulators

print("Checking Gate Dependent Simulators...")
sims_dep = gate_dependent_simulators()
circ = tq.Circuit([{0: tq.Gate.x}])

for strength, (sim, infid) in sims_dep.items():
    op = sim.operator(circuit=circ).upgrade().mat()
    # Check if operator is different
    print(f"Strength {strength}: Operator[0,0] = {op[0,0]}")
    fid = process_fidelity(circ, sim)
    print(f"Strength {strength}: Fidelity = {fid}")

print("\nChecking Gate Independent Simulators...")
sims_ind = gate_independent_simulators()
for strength, (sim, infid) in sims_ind.items():
    op = sim.operator(circuit=circ).upgrade().mat()
    print(f"Strength {strength}: Operator[0,0] = {op[0,0]}")
    fid = process_fidelity(circ, sim)
    print(f"Strength {strength}: Fidelity = {fid}")
