#!/usr/bin/env python
"""
Compare actions from different policy types (e.g., 'zmq' vs 'agg') in the same plot.
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def compute_action_similarity(reference: np.ndarray, candidate: np.ndarray) -> float:
    """
    Compute a bounded similarity score in [0, 1] for two aligned action sequences.

    We use an NRMSE-based score because action trajectories are time-aligned and we care
    about absolute numerical closeness, not just shape correlation.
    """
    valid_steps = min(len(reference), len(candidate))
    if valid_steps == 0:
        return float("nan")

    ref = reference[:valid_steps]
    cand = candidate[:valid_steps]
    rmse = np.sqrt(np.mean((ref - cand) ** 2))

    # Normalize by the teleop range to make scores comparable across dimensions.
    scale = np.ptp(ref)
    if scale < 1e-8:
        scale = max(np.max(np.abs(ref)), 1.0)

    nrmse = rmse / scale
    return 1.0 / (1.0 + nrmse)


def compare_actions(data_dir: str, repo_id: str, episode: int, policy_types: list[str]):
    """
    Compare actions from different policy types in the same plot.

    Args:
        data_dir: Directory containing saved action files
        repo_id: Repository ID used during replay
        episode: Episode index to compare
        policy_types: List of policy types to compare (e.g., ['zmq', 'agg'])
    """
    repo_name = repo_id.replace("/", "_")
    data_path = Path(data_dir)

    # Load teleop actions once (they should be the same regardless of policy type)
    teleop_file = data_path / f"teleop_actions_{repo_name}_episode_{episode}_tel.npy"
    if not teleop_file.exists():
        # Try with the first policy type in the list
        teleop_file = data_path / f"teleop_actions_{repo_name}_episode_{episode}_{policy_types[0]}.npy"

    if not teleop_file.exists():
        print(f"Teleop actions file not found: {teleop_file}")
        return

    teleop_actions = np.load(teleop_file)

    action_dim = teleop_actions.shape[1]

    # Prepare plot dynamically based on action dimension
    ncols = 2 if action_dim > 1 else 1
    nrows = int(np.ceil(action_dim / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(7.5 * ncols, 3 * nrows))
    axes = np.atleast_1d(axes).flatten()

    # Plot teleop actions as reference
    for i in range(action_dim):
        axes[i].plot(teleop_actions[:, i], label="Teleop", linestyle="--", alpha=0.7, linewidth=1.5)

    # Load policy actions and precompute per-dimension similarities.
    colors = ["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple", "tab:brown", "tab:pink", "tab:gray"]
    policy_results = []
    for idx, policy_type in enumerate(policy_types):
        policy_file = data_path / f"policy_actions_{repo_name}_episode_{episode}_{policy_type}.npy"

        if not policy_file.exists():
            print(f"Policy actions file not found: {policy_file}")
            continue

        policy_actions = np.load(policy_file)
        similarities = [
            compute_action_similarity(teleop_actions[:, i], policy_actions[:, i])
            for i in range(min(action_dim, policy_actions.shape[1]))
        ]

        policy_results.append(
            {
                "policy_type": policy_type,
                "policy_actions": policy_actions,
                "color": colors[idx % len(colors)],
                "similarities": similarities,
            }
        )

    best_policy_by_dim = {}
    for i in range(action_dim):
        candidates = [
            (result["policy_type"], result["similarities"][i])
            for result in policy_results
            if i < len(result["similarities"]) and np.isfinite(result["similarities"][i])
        ]
        if candidates:
            best_policy_by_dim[i] = max(candidates, key=lambda item: item[1])[0]

    for result in policy_results:
        policy_type = result["policy_type"]
        policy_actions = result["policy_actions"]
        color = result["color"]
        for i in range(min(action_dim, policy_actions.shape[1])):
            similarity = result["similarities"][i]
            similarity_text = f"sim={similarity:.3f}" if np.isfinite(similarity) else "sim=n/a"
            best_marker = " * " if best_policy_by_dim.get(i) == policy_type else ""
            axes[i].plot(
                policy_actions[:, i],
                label=f"Policy ({policy_type}, {similarity_text}){best_marker}",
                alpha=0.7,
                color=color,
            )

    # Configure plots
    for i in range(action_dim):
        axes[i].set_title(f"Action Dimension {i}")
        axes[i].legend()
        axes[i].grid(True, alpha=0.3)

    # Hide extra subplots if grid has empty slots
    for i in range(action_dim, len(axes)):
        axes[i].set_visible(False)

    plt.suptitle(
        f"Action Comparison: {repo_name} Episode {episode}\n"
        f"(Teleop vs {', '.join(policy_types)} policies, similarity=1/(1+NRMSE))"
    )
    plt.tight_layout()
    plt.savefig(
        f"outputs/action_comparison_{repo_name}_episode_{episode}_{'_vs_'.join(policy_types)}.png",
        dpi=150,
        bbox_inches="tight",
    )
    print(f"saved to outputs/action_comparison_{repo_name}_episode_{episode}_{'_vs_'.join(policy_types)}.png")


def main():
    parser = argparse.ArgumentParser(description="Compare actions from different policy types")
    parser.add_argument("--data-dir", type=str, default="outputs/replay", help="Directory of npy files")
    parser.add_argument("--repo-id", type=str, required=True, help="Repository ID used during replay")
    parser.add_argument(
        "--episodes",
        type=str,
        required=True,
        help="Episode index(es) to compare, separated by commas (e.g., 0,1,2)",
    )
    parser.add_argument(
        "--policy-types", nargs="+", required=True, help="List of policy types to compare (e.g., zmq agg)"
    )

    args = parser.parse_args()

    try:
        episodes = [int(ep.strip()) for ep in args.episodes.split(",") if ep.strip()]
    except ValueError:
        parser.error("--episodes must be comma-separated integers, e.g. --episodes 0,1,2")

    if not episodes:
        parser.error("No valid episode index provided. Use --episodes with comma-separated integers.")

    for episode in episodes:
        compare_actions(args.data_dir, args.repo_id, episode, args.policy_types)


if __name__ == "__main__":
    main()
