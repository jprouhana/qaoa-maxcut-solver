"""
Benchmarking utilities for QAOA Max-Cut experiments.
Runs experiments across different graph sizes and QAOA depths,
then generates plots for analysis.
"""

import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from pathlib import Path

from .qaoa_solver import QAOAMaxCutSolver
from .maxcut_utils import generate_random_graph, generate_regular_graph, generate_cycle_graph


def run_benchmark_suite(node_range=range(3, 11), p_range=range(1, 6),
                        n_trials=5, edge_prob=0.5, save_dir='results'):
    """
    Run QAOA experiments across graph sizes and depths.

    For each (n_nodes, p) pair, runs n_trials with different random seeds
    and collects approximation ratios.
    """
    save_path = Path(save_dir)
    save_path.mkdir(exist_ok=True)

    results = {}

    for n in node_range:
        for p in p_range:
            ratios = []
            for trial in range(n_trials):
                seed = trial * 100 + n * 10 + p
                G = generate_random_graph(n, edge_prob, seed=seed)

                # skip disconnected graphs
                if not nx.is_connected(G):
                    continue

                solver = QAOAMaxCutSolver(G, p=p, seed=seed)
                result = solver.solve(maxiter=200)
                ratios.append(result['approx_ratio'])

                print(f"n={n}, p={p}, trial={trial}: "
                      f"approx_ratio={result['approx_ratio']:.4f}")

            if ratios:
                results[(n, p)] = {
                    'mean': np.mean(ratios),
                    'std': np.std(ratios),
                    'all_ratios': ratios,
                }

    return results


def run_graph_type_comparison(n_nodes=6, p_range=range(1, 6), n_trials=3):
    """
    Compare QAOA performance across different graph types.
    """
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
    """Plot approximation ratio vs QAOA depth for different graph sizes."""
    save_path = Path(save_dir)
    save_path.mkdir(exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 6))

    # group by node count
    node_counts = sorted(set(n for n, p in results.keys()))
    p_values = sorted(set(p for n, p in results.keys()))

    colors = plt.cm.viridis(np.linspace(0, 0.85, len(node_counts)))

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
                       label=f'n={n}', color=colors[idx], capsize=3,
                       linewidth=1.5)

    ax.set_xlabel('QAOA Depth (p)', fontsize=12)
    ax.set_ylabel('Approximation Ratio', fontsize=12)
    ax.set_title('QAOA Approximation Ratio vs Circuit Depth', fontsize=14)
    ax.legend(title='Graph Size')
    ax.set_ylim(0.5, 1.05)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path / 'approx_ratio_vs_p.png', dpi=150)
    plt.close()
    print(f"Saved: {save_path / 'approx_ratio_vs_p.png'}")


def plot_approx_ratio_vs_nodes(results, save_dir='results'):
    """Plot approximation ratio vs graph size for different depths."""
    save_path = Path(save_dir)
    save_path.mkdir(exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 6))

    p_values = sorted(set(p for n, p in results.keys()))
    node_counts = sorted(set(n for n, p in results.keys()))

    colors = plt.cm.plasma(np.linspace(0.1, 0.9, len(p_values)))

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
                       label=f'p={p}', color=colors[idx], capsize=3,
                       linewidth=1.5)

    ax.set_xlabel('Number of Qubits (Graph Nodes)', fontsize=12)
    ax.set_ylabel('Approximation Ratio', fontsize=12)
    ax.set_title('QAOA Performance vs Problem Size', fontsize=14)
    ax.legend(title='QAOA Depth')
    ax.set_ylim(0.5, 1.05)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path / 'approx_ratio_vs_nodes.png', dpi=150)
    plt.close()
    print(f"Saved: {save_path / 'approx_ratio_vs_nodes.png'}")


def plot_convergence(cost_history, title='QAOA Convergence', save_dir='results'):
    """Plot the optimization convergence curve."""
    save_path = Path(save_dir)
    save_path.mkdir(exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(range(len(cost_history)), [-c for c in cost_history],
            linewidth=1.5, color='#2196F3')
    ax.set_xlabel('Iteration', fontsize=12)
    ax.set_ylabel('Expected Cut Value', fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path / 'convergence.png', dpi=150)
    plt.close()
    print(f"Saved: {save_path / 'convergence.png'}")


def plot_graph_type_comparison(results, save_dir='results'):
    """Compare QAOA performance across different graph types."""
    save_path = Path(save_dir)
    save_path.mkdir(exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 6))

    colors = {'Erdos-Renyi (p=0.5)': '#FF6B6B', 'Regular (d=3)': '#4ECDC4',
              'Cycle': '#45B7D1'}
    markers = {'Erdos-Renyi (p=0.5)': 'o', 'Regular (d=3)': 's', 'Cycle': '^'}

    for gtype, p_data in results.items():
        p_vals = sorted(p_data.keys())
        means = [p_data[p]['mean'] for p in p_vals]
        stds = [p_data[p]['std'] for p in p_vals]

        ax.errorbar(p_vals, means, yerr=stds,
                   marker=markers.get(gtype, 'o'),
                   label=gtype, color=colors.get(gtype, 'gray'),
                   capsize=3, linewidth=1.5)

    ax.set_xlabel('QAOA Depth (p)', fontsize=12)
    ax.set_ylabel('Approximation Ratio', fontsize=12)
    ax.set_title('QAOA Performance by Graph Type (n=6)', fontsize=14)
    ax.legend()
    ax.set_ylim(0.6, 1.05)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path / 'graph_type_comparison.png', dpi=150)
    plt.close()
    print(f"Saved: {save_path / 'graph_type_comparison.png'}")


if __name__ == '__main__':
    print("Running QAOA benchmark suite...")
    print("This may take a while for larger graph sizes.\n")

    # run the main benchmark
    results = run_benchmark_suite(
        node_range=range(3, 9),
        p_range=range(1, 6),
        n_trials=3
    )

    # generate plots
    plot_approx_ratio_vs_p(results)
    plot_approx_ratio_vs_nodes(results)

    # run graph type comparison
    gtype_results = run_graph_type_comparison(n_nodes=6)
    plot_graph_type_comparison(gtype_results)

    print("\nDone! Check the results/ folder for plots.")
