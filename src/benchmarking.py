import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from pathlib import Path

from .qaoa_solver import QAOAMaxCutSolver
from .maxcut_utils import generate_random_graph, generate_regular_graph, generate_cycle_graph


def run_benchmark_suite(node_range=range(3, 11), p_range=range(1, 6),
                        n_trials=5, edge_prob=0.5, save_dir='results'):
    save_path = Path(save_dir)
    save_path.mkdir(exist_ok=True)

    results = {}

    for n in node_range:
        for p in p_range:
            ratios = []
            for trial in range(n_trials):
                seed = trial * 100 + n * 10 + p
                G = generate_random_graph(n, edge_prob, seed=seed)

                if not nx.is_connected(G):
                    continue

                solver = QAOAMaxCutSolver(G, p=p, seed=seed)
                result = solver.solve(maxiter=200)
                ratios.append(result['approx_ratio'])

                print(f"n={n}, p={p}, trial={trial}: "
                      f"ratio={result['approx_ratio']:.4f}")

            if ratios:
                results[(n, p)] = {
                    'mean': np.mean(ratios),
                    'std': np.std(ratios),
                    'all_ratios': ratios,
                }

    return results


def run_graph_type_comparison(n_nodes=6, p_range=range(1, 6), n_trials=3):
    graph_types = {
        'Erdos-Renyi (p=0.5)': lambda seed: generate_random_graph(n_nodes, 0.5, seed),
        'Regular (d=3)': lambda seed: generate_regular_graph(n_nodes, 3, seed),
        'Cycle': lambda seed: generate_cycle_graph(n_nodes),
    }

    results = {}
    for gtype, gen_func in graph_types.items():
        results[gtype] = {}
        for p in p_range:
            ratios = []
            for trial in range(n_trials):
                seed = trial * 50 + p
                G = gen_func(seed)
                solver = QAOAMaxCutSolver(G, p=p, seed=seed)
                result = solver.solve(maxiter=200)
                ratios.append(result['approx_ratio'])

            results[gtype][p] = {
                'mean': np.mean(ratios),
                'std': np.std(ratios),
            }

    return results


def plot_approx_ratio_vs_p(results, save_dir='results'):
    save_path = Path(save_dir)
    save_path.mkdir(exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 6))

    node_counts = sorted(set(n for n, p in results.keys()))
    p_values = sorted(set(p for n, p in results.keys()))

    for idx, n in enumerate(node_counts):
        means = []
        stds = []
        valid_p = []
        for p in p_values:
            if (n, p) in results:
                means.append(results[(n, p)]['mean'])
                stds.append(results[(n, p)]['std'])
                valid_p.append(p)

        if means:
            ax.errorbar(valid_p, means, yerr=stds, marker='o',
                       label=f'n={n}', capsize=3)

    ax.set_xlabel('QAOA Depth (p)')
    ax.set_ylabel('Approximation Ratio')
    ax.set_title('Approx Ratio vs Circuit Depth')
    ax.legend(title='Graph Size')
    ax.set_ylim(0.5, 1.05)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path / 'approx_ratio_vs_p.png', dpi=150)
    plt.close()
    print(f"saved plot to {save_path / 'approx_ratio_vs_p.png'}")


def plot_approx_ratio_vs_nodes(results, save_dir='results'):
    save_path = Path(save_dir)
    save_path.mkdir(exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 6))

    p_values = sorted(set(p for n, p in results.keys()))
    node_counts = sorted(set(n for n, p in results.keys()))

    for idx, p in enumerate(p_values):
        means = []
        stds = []
        valid_n = []
        for n in node_counts:
            if (n, p) in results:
                means.append(results[(n, p)]['mean'])
                stds.append(results[(n, p)]['std'])
                valid_n.append(n)

        if means:
            ax.errorbar(valid_n, means, yerr=stds, marker='s',
                       label=f'p={p}', capsize=3)

    ax.set_xlabel('Number of Nodes')
    ax.set_ylabel('Approximation Ratio')
    ax.set_title('QAOA Performance vs Graph Size')
    ax.legend(title='QAOA Depth')
    ax.set_ylim(0.5, 1.05)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path / 'approx_ratio_vs_nodes.png', dpi=150)
    plt.close()
    print(f"saved plot to {save_path / 'approx_ratio_vs_nodes.png'}")


def plot_convergence(cost_history, title='QAOA Convergence', save_dir='results'):
    save_path = Path(save_dir)
    save_path.mkdir(exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(range(len(cost_history)), [-c for c in cost_history])
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Expected Cut Value')
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path / 'convergence.png', dpi=150)
    plt.close()
    print(f"saved plot to {save_path / 'convergence.png'}")


def plot_graph_type_comparison(results, save_dir='results'):
    save_path = Path(save_dir)
    save_path.mkdir(exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 6))

    for gtype, p_data in results.items():
        p_vals = sorted(p_data.keys())
        means = [p_data[p]['mean'] for p in p_vals]
        stds = [p_data[p]['std'] for p in p_vals]

        ax.errorbar(p_vals, means, yerr=stds, marker='o',
                   label=gtype, capsize=3)

    ax.set_xlabel('QAOA Depth (p)')
    ax.set_ylabel('Approximation Ratio')
    ax.set_title('QAOA by Graph Type (n=6)')
    ax.legend()
    ax.set_ylim(0.6, 1.05)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path / 'graph_type_comparison.png', dpi=150)
    plt.close()
    print(f"saved plot to {save_path / 'graph_type_comparison.png'}")


if __name__ == '__main__':
    print("running benchmarks... this might take a while")

    results = run_benchmark_suite(
        node_range=range(3, 9),
        p_range=range(1, 6),
        n_trials=3
    )

    plot_approx_ratio_vs_p(results)
    plot_approx_ratio_vs_nodes(results)

    gtype_results = run_graph_type_comparison(n_nodes=6)
    plot_graph_type_comparison(gtype_results)

    print("\ndone! check results/ folder")
