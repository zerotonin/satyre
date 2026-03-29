#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════╗
║  five_strategy_comparison.py                             ║
║  « reproduces the foraging-efficiency comparison from    ║
║    Figure 5 D–E of the manuscript »                      ║
╚══════════════════════════════════════════════════════════╝

This example compares five locomotor strategies in a
toroidal arena under dark (tactile-only) conditions:

  1. Random walk (Gaussian steps)
  2. Lévy flight — OregonR velocity profile
  3. Lévy flight — dark-fly velocity profile
  4. Empirical OregonR locomotion (Markov replay)
  5. Empirical dark-fly locomotion (Markov replay)

To run, you need preprocessed .npy files for OregonR and
dark-fly strains (see ``satyre-preprocess``).

Usage:
    python examples/five_strategy_comparison.py \\
        --orl-dir data/preprocessed_orl \\
        --df-dir  data/preprocessed_df  \\
        --n-trials 50 --max-steps 1000
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from satyre import LevyWalker, MarkovWalker
from satyre.analysis import summarise_food_collected


def load_markov_data(directory: Path):
    """Load preprocessed transition matrices and velocities.

    Args:
        directory: Path to directory containing the three
            ``.npy`` files produced by ``satyre-preprocess``.

    Returns:
        Tuple of ``(transition_matrix, pm_index, velocities)``.
    """
    trans = np.load(directory / "cumsum_transition_matrix.npy")
    pm_idx = np.load(directory / "pm_index_matrix.npy")
    velos = np.load(directory / "velocity_array.npy")
    return trans, pm_idx, velos


def main():
    parser = argparse.ArgumentParser(
        description="Five-strategy foraging comparison."
    )
    parser.add_argument("--orl-dir", type=Path, required=True)
    parser.add_argument("--df-dir", type=Path, required=True)
    parser.add_argument("--n-trials", type=int, default=50)
    parser.add_argument("--max-steps", type=int, default=1_000)
    parser.add_argument("--food-mode", default="random")
    parser.add_argument("--total-food", type=int, default=1_000)
    args = parser.parse_args()

    food_found: dict[str, np.ndarray] = {}

    # ── 1. Random walk ──────────────────────────────────
    print("Running: Random walk …")
    rw = LevyWalker(
        mode="uniform",
        n_trials=args.n_trials,
        max_steps=args.max_steps,
        total_food=args.total_food,
        food_mode=args.food_mode,
    )
    food_found["Random walk"] = rw.simulate_multiple()["food_found"]

    # ── 2. Lévy flight (ORL velocities) ─────────────────
    print("Running: Lévy flight (ORL) …")
    lv_orl = LevyWalker(
        cauchy_alpha=1.5,
        mode="cauchy",
        n_trials=args.n_trials,
        max_steps=args.max_steps,
        total_food=args.total_food,
        food_mode=args.food_mode,
    )
    food_found["Lévy ORL"] = lv_orl.simulate_multiple()["food_found"]

    # ── 3. Lévy flight (dark-fly velocities) ────────────
    print("Running: Lévy flight (dark-fly) …")
    lv_df = LevyWalker(
        cauchy_alpha=1.5,
        mode="cauchy",
        n_trials=args.n_trials,
        max_steps=args.max_steps,
        total_food=args.total_food,
        food_mode=args.food_mode,
    )
    food_found["Lévy dark-fly"] = lv_df.simulate_multiple()["food_found"]

    # ── 4. OregonR Markov replay ────────────────────────
    print("Running: OregonR Markov replay …")
    t_orl, i_orl, v_orl = load_markov_data(args.orl_dir)
    mw_orl = MarkovWalker(
        t_orl, i_orl, v_orl,
        n_trials=args.n_trials,
        max_steps=args.max_steps,
        total_food=args.total_food,
        food_mode=args.food_mode,
    )
    food_found["OregonR"] = mw_orl.simulate_multiple()["food_found"]

    # ── 5. Dark-fly Markov replay ───────────────────────
    print("Running: Dark-fly Markov replay …")
    t_df, i_df, v_df = load_markov_data(args.df_dir)
    mw_df = MarkovWalker(
        t_df, i_df, v_df,
        n_trials=args.n_trials,
        max_steps=args.max_steps,
        total_food=args.total_food,
        food_mode=args.food_mode,
    )
    food_found["Dark-fly"] = mw_df.simulate_multiple()["food_found"]

    # ── Summary ─────────────────────────────────────────
    print("\n── Foraging efficiency (dark condition) ──")
    for name, ff in food_found.items():
        stats = summarise_food_collected({"food_found": ff})
        print(
            f"  {name:20s}  mean={stats['mean']:.2f}  "
            f"95% CI=[{stats['ci_lower']:.2f}, {stats['ci_upper']:.2f}]"
        )

    # ── Plot ────────────────────────────────────────────
    df = pd.DataFrame(food_found)
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.boxplot(data=df, notch=True, ax=ax)
    ax.set_ylabel("Food items collected", fontsize=14)
    ax.set_title("Foraging efficiency — dark condition", fontsize=16)
    ax.tick_params(labelsize=12)
    fig.tight_layout()
    fig.savefig("five_strategy_comparison.svg", dpi=300)
    fig.savefig("five_strategy_comparison.png", dpi=300)
    plt.show()
    print("\nFigure saved to five_strategy_comparison.svg / .png")


if __name__ == "__main__":
    main()
