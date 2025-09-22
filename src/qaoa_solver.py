import numpy as np
from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator
from qiskit.quantum_info import SparsePauliOp
from scipy.optimize import minimize

from .maxcut_utils import brute_force_maxcut, compute_cut_value


class QAOAMaxCutSolver:

    def __init__(self, graph, p=1, shots=4096, seed=None):
        self.graph = graph
        self.p = p
        self.shots = shots
        self.n_qubits = graph.number_of_nodes()
        self.nodes = list(graph.nodes())
        self.simulator = AerSimulator()
        self.seed = seed
        self.cost_history = []

    def _build_qaoa_circuit(self, gamma, beta):
        qc = QuantumCircuit(self.n_qubits)

        qc.h(range(self.n_qubits))

        for layer in range(self.p):
            for u, v, data in self.graph.edges(data=True):
                i = self.nodes.index(u)
                j = self.nodes.index(v)
                w = data.get('weight', 1.0)
                qc.cx(i, j)
                qc.rz(2 * gamma[layer] * w, j)
                qc.cx(i, j)

            for i in range(self.n_qubits):
                qc.rx(2 * beta[layer], i)

        qc.measure_all()
        return qc

    def _evaluate_cost(self, bitstring):
        partition = [int(b) for b in bitstring]
        return compute_cut_value(self.graph, partition)

    def _objective(self, params):
        gamma = params[:self.p]
        beta = params[self.p:]

        qc = self._build_qaoa_circuit(gamma, beta)

        job = self.simulator.run(qc, shots=self.shots, seed_simulator=self.seed)
        result = job.result()
        counts = result.get_counts()

        total_cost = 0
        for bitstring, count in counts.items():
            # qiskit gives bitstrings reversed for some reason
            bits = bitstring[::-1]
            cost = self._evaluate_cost(bits)
            total_cost += cost * count

        avg_cost = total_cost / self.shots
        self.cost_history.append(-avg_cost)
        # print(f"  iteration {len(self.cost_history)}: avg_cost = {avg_cost:.4f}")

        return -avg_cost

    def solve(self, maxiter=300):
        self.cost_history = []

        rng = np.random.default_rng(self.seed)
        gamma0 = rng.uniform(0, 2 * np.pi, self.p)
        beta0 = rng.uniform(0, np.pi, self.p)
        x0 = np.concatenate([gamma0, beta0])

        res = minimize(
            self._objective,
            x0,
            method='COBYLA',
            options={'maxiter': maxiter, 'rhobeg': 0.5}
        )

        opt_gamma = res.x[:self.p]
        opt_beta = res.x[self.p:]

        qc = self._build_qaoa_circuit(opt_gamma, opt_beta)
        job = self.simulator.run(qc, shots=self.shots, seed_simulator=self.seed)
        result = job.result()
        counts = result.get_counts()

        best_bitstring = max(counts, key=lambda b: self._evaluate_cost(b[::-1]))
        best_partition = tuple(int(b) for b in best_bitstring[::-1])
        best_cut = compute_cut_value(self.graph, best_partition)

        exact_cut, exact_partition = brute_force_maxcut(self.graph)
        approx_ratio = best_cut / exact_cut if exact_cut > 0 else 1.0

        print(f"  qaoa done: cut={best_cut}/{exact_cut}, ratio={approx_ratio:.4f}")

        return {
            'best_partition': best_partition,
            'best_cut_value': best_cut,
            'exact_cut_value': exact_cut,
            'exact_partition': exact_partition,
            'approx_ratio': approx_ratio,
            'optimal_gamma': opt_gamma,
            'optimal_beta': opt_beta,
            'cost_history': self.cost_history.copy(),
            'optimization_result': res,
            'measurement_counts': counts,
        }


def run_single_experiment(n_nodes, p, edge_prob=0.5, seed=None):
    from .maxcut_utils import generate_random_graph

    G = generate_random_graph(n_nodes, edge_prob, seed=seed)
    solver = QAOAMaxCutSolver(G, p=p, seed=seed)
    result = solver.solve()
    return result
