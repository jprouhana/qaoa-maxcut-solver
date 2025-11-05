# QAOA Max-Cut Solver

Implementation of the Quantum Approximate Optimization Algorithm (QAOA) for solving the Max-Cut problem on various graph structures. Built as part of an independent study on quantum-classical hybrid optimization at Arizona State University.

## Background

The **Max-Cut problem** asks: given a graph, find a partition of vertices into two sets that maximizes the number of edges crossing the partition. This is an NP-hard combinatorial optimization problem with applications in circuit design, statistical physics, and network analysis.

**QAOA** (Farhi et al., 2014) is a variational quantum algorithm that uses alternating layers of problem-specific and mixer unitaries to find approximate solutions to combinatorial optimization problems. The circuit depth is controlled by a parameter `p` — higher values of `p` allow the algorithm to explore more of the solution space but require more quantum resources.

### How QAOA Works

1. Encode the Max-Cut objective as a cost Hamiltonian $H_C$
2. Prepare an initial state $|+\rangle^{\otimes n}$
3. Apply $p$ alternating layers of $e^{-i\gamma H_C}$ and $e^{-i\beta H_M}$
4. Measure and use a classical optimizer (COBYLA) to tune $\gamma$ and $\beta$
5. Repeat until convergence

## Project Structure

```
qaoa-maxcut-solver/
├── src/
│   ├── qaoa_solver.py       # Main QAOA implementation
│   ├── maxcut_utils.py      # Graph generation and classical solver
│   └── benchmarking.py      # Benchmarking and plotting utilities
├── notebooks/
│   └── experiments.ipynb     # Full analysis walkthrough
├── results/                  # Saved plots and data
├── requirements.txt
├── README.md
└── LICENSE
```

## Installation

```bash
git clone https://github.com/jrouhana/qaoa-maxcut-solver.git
cd qaoa-maxcut-solver
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Usage

### Quick Start

```python
from src.qaoa_solver import QAOAMaxCutSolver
from src.maxcut_utils import generate_random_graph

# Create a random graph with 6 nodes
G = generate_random_graph(n_nodes=6, edge_prob=0.5, seed=42)

# Run QAOA with depth p=3
solver = QAOAMaxCutSolver(G, p=3)
result = solver.solve()

print(f"Best cut value: {result['best_cut_value']}")
print(f"Best partition: {result['best_partition']}")
print(f"Approximation ratio: {result['approx_ratio']:.4f}")
```

### Running Benchmarks

```python
from src.benchmarking import run_benchmark_suite

# Benchmark across graph sizes and QAOA depths
results = run_benchmark_suite(
    node_range=range(3, 11),
    p_range=range(1, 6),
    n_trials=5
)
```

### Jupyter Notebook

The main analysis is in `notebooks/experiments.ipynb`. Open it with:

```bash
jupyter notebook notebooks/experiments.ipynb
```

## Results

### Approximation Ratio vs QAOA Depth

Higher `p` values consistently improve solution quality, with diminishing returns past p=3 for small graphs:

| Graph Size | p=1   | p=2   | p=3   | p=4   | p=5   |
|-----------|-------|-------|-------|-------|-------|
| 4 nodes   | 0.875 | 0.938 | 0.969 | 0.984 | 0.992 |
| 6 nodes   | 0.821 | 0.893 | 0.937 | 0.958 | 0.971 |
| 8 nodes   | 0.789 | 0.862 | 0.911 | 0.939 | 0.953 |
| 10 nodes  | 0.756 | 0.834 | 0.889 | 0.918 | 0.935 |

*Values are averages over 5 random Erdos-Renyi graphs per size.*

### Key Findings

- QAOA achieves >90% approximation ratios for p>=3 on graphs up to 8 qubits
- Performance degrades gracefully as graph size increases
- COBYLA optimizer converges within ~200 iterations for most cases
- Structured graphs (regular, cycle) tend to give better results than random graphs

## References

1. Farhi, E., Goldstone, J., & Gutmann, S. (2014). "A Quantum Approximate Optimization Algorithm." [arXiv:1411.4028](https://arxiv.org/abs/1411.4028)
2. Guerreschi, G. G., & Matsuura, A. Y. (2019). "QAOA for Max-Cut requires hundreds of qubits for quantum speed-up." [arXiv:1812.07589](https://arxiv.org/abs/1812.07589)
3. Qiskit Documentation: [https://qiskit.org/documentation/](https://qiskit.org/documentation/)

## License

MIT License — see [LICENSE](LICENSE) for details.
