"""Reproduce every figure and number quoted in the README.

    python scripts/make_figures.py
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from shuffling import (ExactChain, top_to_random_step, variant_step, tv_distance,
                       tv_exact, separation_exact, sst_tail, sst_mean, riffle_tv_exact)

OUT = Path(__file__).resolve().parents[1] / "figures"
OUT.mkdir(exist_ok=True)
C = {"blue": "#2a78d6", "orange": "#eb6834", "aqua": "#1baf7a", "yellow": "#eda100",
     "ink": "#0b0b0b", "ink2": "#52514e", "muted": "#898781", "grid": "#e6e5e1"}
RAMP = ["#86b6ef", "#5598e7", "#2a78d6", "#184f95"]  # sequential blue, light -> dark

plt.rcParams.update({
    "figure.dpi": 150, "savefig.bbox": "tight", "font.size": 10,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.edgecolor": C["muted"], "axes.labelcolor": C["ink2"], "xtick.color": C["muted"],
    "ytick.color": C["muted"], "axes.grid": True, "grid.color": C["grid"], "grid.linewidth": 0.6,
    "axes.titleweight": "bold", "axes.titlesize": 11, "axes.titlecolor": C["ink"],
    "lines.linewidth": 2, "legend.frameon": False,
})


def fig_top_to_random_52():
    N, n_max = 52, 450
    n = np.arange(n_max + 1)
    tv, sep, tail = tv_exact(N, n_max), separation_exact(N, n_max), sst_tail(N, n_max)
    fig, ax = plt.subplots(figsize=(7.2, 3.8))
    ax.plot(n, tail, color=C["orange"], label="Strong stationary time bound  P(T > n)")
    ax.plot(n, sep, color=C["aqua"], label="Exact separation distance  s(n)")
    ax.plot(n, tv, color=C["blue"], label="Exact total variation  d_TV(n)")
    ax.axvline(N * np.log(N), color=C["muted"], ls="--", lw=1)
    ax.text(N * np.log(N) + 4, 0.97, "N log N ≈ 205", color=C["ink2"], fontsize=9)
    ax.set(xlabel="number of top-to-random shuffles n", ylabel="distance to uniform",
           title="Top-to-random shuffle, N = 52: exact distances vs. the textbook bound", ylim=(0, 1.02))
    ax.legend(loc="center right", fontsize=8.5)
    fig.savefig(OUT / "top_to_random_52.png")


def fig_cutoff():
    fig, ax = plt.subplots(figsize=(7.2, 3.8))
    for N, col in zip([10, 52, 250, 1000], RAMP):
        n_max = int(2 * N * np.log(N))
        tv = tv_exact(N, n_max)
        x = np.arange(n_max + 1) / (N * np.log(N))
        ax.plot(x, tv, color=col, label=f"N = {N}")
    ax.set(xlabel="n / (N log N)", ylabel="d_TV(n)", xlim=(0, 2), ylim=(0, 1.02),
           title="Cutoff phenomenon: the TV curve converges to a step at n = N log N")
    ax.legend(loc="upper right")
    fig.savefig(OUT / "cutoff.png")


def fig_riffle_vs_insertion():
    N = 52
    m = np.arange(0, 13)
    tv_r = [1.0] + [riffle_tv_exact(N, k) for k in m[1:]]
    fig, ax = plt.subplots(figsize=(7.2, 3.6))
    ax.plot(m, tv_r, color=C["blue"], marker="o", ms=5)
    for k, v in zip(m[5:11], tv_r[5:11]):
        ax.annotate(f"{v:.3f}", (k, v), textcoords="offset points", xytext=(6, 4), fontsize=8, color=C["ink2"])
    ax.axvline(1.5 * np.log2(N), color=C["muted"], ls="--", lw=1)
    ax.text(1.5 * np.log2(N) + 0.15, 0.9, "(3/2) log₂ N ≈ 8.6", color=C["ink2"], fontsize=9)
    ax.set(xlabel="number of riffle shuffles m", ylabel="d_TV(m)", ylim=(0, 1.05),
           title="GSR riffle shuffle, N = 52 (exact, Bayer–Diaconis 1992)")
    fig.savefig(OUT / "riffle_52.png")


def fig_variant():
    fig, axes = plt.subplots(1, 3, figsize=(9.6, 3.2), sharey=True)
    for ax, N in zip(axes, [4, 6, 7]):
        n_max = 4 * N
        std = ExactChain(N)
        var = ExactChain(N)
        tv_std = [tv_distance(std.mu)]
        tv_var = [tv_distance(var.mu)]
        for t in range(1, n_max + 1):
            tv_std.append(tv_distance(std.advance(top_to_random_step, key=0)))
            tv_var.append(tv_distance(var.advance(variant_step(t), key=min(t, N) + 100)))
        ax.plot(range(n_max + 1), tv_std, color=C["blue"], label="top-to-random")
        ax.plot(range(n_max + 1), tv_var, color=C["orange"], label="variant (report §5)")
        ax.axvline(N, color=C["muted"], ls="--", lw=1)
        ax.set(title=f"N = {N}", xlabel="shuffles n")
    axes[0].set_ylabel("d_TV(n)  (exact, on all N! decks)")
    axes[0].legend(fontsize=8)
    fig.suptitle("The variant is an exact sampler: d_TV = 0 after exactly N steps", fontweight="bold", fontsize=11, y=1.04)
    fig.savefig(OUT / "variant.png")


def print_numbers():
    N, n_max = 52, 600
    tv, sep, tail = tv_exact(N, n_max), separation_exact(N, n_max), sst_tail(N, n_max)
    first = lambda a, e: int(np.argmax(a < e))
    print(f"E[T] = N H_N = {sst_mean(N):.1f}   N log N = {N*np.log(N):.1f}")
    print("eps   TV   separation   SST bound")
    for e in (0.5, 0.25, 0.05, 0.01):
        print(f"{e:<5} {first(tv, e):<4} {first(sep, e):<12} {first(tail, e)}")
    print("riffle:", [round(riffle_tv_exact(N, m), 3) for m in range(1, 13)])


if __name__ == "__main__":
    fig_top_to_random_52()
    fig_cutoff()
    fig_riffle_vs_insertion()
    fig_variant()
    print_numbers()
