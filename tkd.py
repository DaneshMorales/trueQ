import trueq as tq
import trueq.math as tqm

############ CIRCUIT EXTENSIONS ############

def _unitary(self: tq.Circuit):
    sim = tq.Simulator()
    op = sim.operator(circuit=self)
    return op.mat()

def ptm(self: tq.Circuit, sim = tq.Simulator()):
    sim = tq.Simulator()
    ptm = tqm.Superop.from_rowstack(sim.operator(circuit=self).upgrade().mat()).ptm
    return ptm

def ptm_plot(self: tq.Circuit, sim = tq.Simulator()):
    sim = tq.Simulator()
    ptm = tqm.Superop.from_rowstack(sim.operator(circuit=self).upgrade().mat()).plot_ptm()
    return ptm

tq.Circuit.unitary = property(_unitary)
tq.Circuit.ptm = property(ptm)
tq.Circuit.ptm_plot = property(ptm_plot)

############ GATE EXTENSIONS ############

def ptm(self: tq.Gate, sim = tq.Simulator()):
    sim = tq.Simulator()
    ptm = tqm.Superop.from_rowstack(sim.operator(circuit=tq.Circuit({0: self})).upgrade().mat()).ptm
    return ptm

def ptm_plot(self: tq.Gate, sim = tq.Simulator()):
    sim = tq.Simulator()
    ptm = tqm.Superop.from_rowstack(sim.operator(circuit=tq.Circuit({0: self})).upgrade().mat()).plot_ptm()
    return ptm

tq.Gate.ptm = property(ptm)
tq.Gate.ptm_plot = property(ptm_plot)


