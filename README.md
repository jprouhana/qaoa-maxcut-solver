# qaoa-maxcut-solver

QAOA for max-cut using qiskit. benchmarks different circuit depths on random graphs.

## setup

```
pip install -r requirements.txt
```

## usage

```python
from src.qaoa_solver import QAOAMaxCutSolver
from src.maxcut_utils import generate_random_graph

G = generate_random_graph(6, 0.5, seed=42)
solver = QAOAMaxCutSolver(G, p=3)
result = solver.solve()
```

the notebook in `notebooks/experiments.ipynb` has the full analysis with plots.
