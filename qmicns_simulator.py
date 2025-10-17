import numpy as np
import matplotlib.pyplot as plt
import time
from tqdm import trange


from brian2 import (
    start_scope, NeuronGroup, Synapses, SpikeMonitor, StateMonitor,
    run, defaultclock, ms, mV, volt, Network, prefs
)

from qiskit import QuantumCircuit, Aer
from qiskit.quantum_info import Statevector, Operator, Pauli


# Parameters (editable)
N_NEURONS = 1000
p_conn = 0.02            # connectivity prob
sim_time = 2.0           # seconds
dt_ms = 0.1              # integration step (ms)
T_w_ms = 10.0            # coupling window (ms)
n_qubits = 6             # number of qubits in quantum layer (keep small for RAM)
alpha = 1.2              # mapping gain from firing rate -> rotation angle
feedback_gain = 0.8      # how strongly qubit expectation drives neuronal current
subpop_fraction = 0.1    # fraction of neurons that read into a given qubit



start_scope()
defaultclock.dt = dt_ms * ms

tau = 10.0 * ms
eqs = '''
dv/dt = (-(v - -65*mV) + I)/tau : volt
I : volt
'''
G = NeuronGroup(N_NEURONS, eqs, threshold='v>-50*mV', reset='v=-65*mV', method='euler')
G.v = -65.0 * mV
G.I = 0.0 * mV

# Synapses (sparse random)
S = Synapses(G, G, on_pre='I += 1.0*mV')
S.connect(p=p_conn)

# Monitors

spikemon = SpikeMonitor(G)

# record voltages of a small subset

record_idx = np.arange(min(50, N_NEURONS))
statemon = StateMonitor(G, 'v', record=record_idx)

# For network operations

net = Network(G, S, spikemon, statemon)


# Quantum setup (statevector)
backend = Aer.get_backend('aer_simulator_statevector')

# initial state |0...0>

sv = Statevector.from_label('0' * n_qubits)

def apply_rotation_and_noise(statevector, target_qubit, theta):
    """Apply Ry(theta) on target_qubit and return new Statevector (no noise here).
       For noise, one would apply channels (requires density matrix)."""
    qc = QuantumCircuit(n_qubits)
    qc.ry(theta, target_qubit)
    # Use Statevector to apply unitary
    U = Operator(qc)
    new_sv = statevector.evolve(U)
    return new_sv

def measure_expectation_z(statevector, target_qubit):
  
    # expectation of Z on qubit = <psi| Z_k |psi>
    # Build Pauli Z on target qubit
  
    pauli = 1
    # Using tensor products: build operator as numpy (but qiskit provides tools)
    # Simpler: compute reduced density matrix of qubit k and then <Z> = rho_00 - rho_11
  
    sv = statevector.data
    dim = 2 ** n_qubits
  
    # reshape amplitude vector to (2, 2^{n-1}) and compute populations by basis index
    # Easier approach using Statevector.probabilities_dict? We'll compute reduced density:
  
    from qiskit.quantum_info import partial_trace, DensityMatrix
    dm = DensityMatrix(statevector)
    reduced = partial_trace(dm, [i for i in range(n_qubits) if i != target_qubit])
  
    # reduced is 2x2 density matrix
  
    rho = reduced.data
    expZ = np.real(rho[0,0] - rho[1,1])
    return float(expZ)



window_counts = []
window_start = 0.0
coupling_times = np.arange(0.0, sim_time*1000.0, T_w_ms)  

# mapping: population indices for each qubit

rng = np.random.default_rng(42)
pop_indices = []
pop_size = max(1, int(subpop_fraction * N_NEURONS))
for q in range(n_qubits):
    pop_indices.append(rng.choice(N_NEURONS, size=pop_size, replace=False))

# storage for metrics

time_points = []
global_rates = []
qubit_expectations = [[] for _ in range(n_qubits)]

# network-run loop with manual stepping

t = 0.0
dt = dt_ms
n_steps = int((sim_time*1000.0) / dt_ms)

print("Starting simulation: neurons=", N_NEURONS, "qubits=", n_qubits)
for step in trange(n_steps):
  
    # advance Brian2 by dt
  
    net.run(dt * ms, report=None)  
    t += dt
   
    if (t * 1.0) % T_w_ms < 1e-6 or abs((t * 1.0) % T_w_ms - T_w_ms) < 1e-6:
        # compute firing rate per qubit population over last window
        now = float(t)
        time_points.append(now)
        # compute spikes in last T_w_ms
        window_start_ms = now - T_w_ms
        # global firing rate (spikes per neuron per second)
        recent_spikes = spikemon.count - 0  # total counts per neuron
        total_spikes_in_window = 0
        # brute force compute spikes timestamps in interval
        spike_times = spikemon.t/ms
        spike_indices = spikemon.i
        mask = (spike_times >= window_start_ms) & (spike_times <= now)
        total_spikes_in_window = np.sum(mask)
        global_rate = total_spikes_in_window / (N_NEURONS * (T_w_ms/1000.0))  # Hz per neuron
        global_rates.append(global_rate)
        
     
        for q in range(n_qubits):
            inds = pop_indices[q]
          
            # spikes for those neurons in window
          
            maskq = (spike_times >= window_start_ms) & (spike_times <= now) & np.isin(spike_indices, inds)
            count_q = np.sum(maskq)
            rate_q = count_q / (len(inds) * (T_w_ms/1000.0)) 
            theta = alpha * np.tanh(0.01 * rate_q)
            sv = apply_rotation_and_noise(sv, q, theta)
            expZ = measure_expectation_z(sv, q)
            qubit_expectations[q].append(expZ)
            
            # retro-feedback: apply small current to a random subset of that pop
            # compute feedback current in mV
          
            feedback_current = feedback_gain * expZ * 1.0  
            # scale to mV units later
            # apply to a subset (first 10 indices) - here we implement simply by setting I for those neurons
            if len(inds) > 0:
                subinds = inds[:max(1, min(10, len(inds)))]
              
                # convert to brian indices and add current (vectorized)
              
                G.I[subinds] += feedback_current * mV  
              
                # NOTE: simplified model



plt.figure(figsize=(10,6))
plt.subplot(2,1,1)
plt.plot(time_points, global_rates, '-o')
plt.title('Global firing rate (Hz per neuron) per coupling window')
plt.xlabel('time (ms)')
plt.ylabel('rate (Hz)')

plt.subplot(2,1,2)
for q in range(n_qubits):
    plt.plot(time_points, qubit_expectations[q], label=f'Q{q}')
plt.title('Qubit <Z> expectations')
plt.xlabel('time (ms)')
plt.legend()
plt.tight_layout()
plt.show()
