#!/usr/bin/env python3
"""
Analysis & visualisation for **LSY‑Drone‑Racing** logs
=====================================================

Key figures (PNG + optional TikZ)
---------------------------------
1. Lap‑time distribution (box + violin)
2. Empirical CDF of lap‑times
3. Solver‑time histogram (log‑y)
4. Lap‑time ↔ avg‑speed scatter
5. Crash‑location hex‑heat‑map
6. Gate / obstacle layout **with speed‑coloured run paths** (crashes on top)
7. Mass‑deviation vs outcome (heavier / lighter)
8. Avg‑speed vs peak‑speed correlation (successful runs)
9. Representative speed / solver‑time series (fastest run)

*Extra logging already present in* **sim.py**:
• `pos_t` – per‑step XYZ positions  • compressed gate / obstacle traces  • `mass_deviation`
"""

from __future__ import annotations

from pathlib import Path
import json
import warnings
from typing import List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib import patches

# ───────────────────────── backend helpers ────────────────────────────
try:
    import tikzplotlib  # type: ignore

    _SAVE_AS_TIKZ = True
except ModuleNotFoundError:
    _SAVE_AS_TIKZ = False
    warnings.warn("tikzplotlib not available – PNG only", RuntimeWarning)

_DEF_DPI = 300


def _save(fig: plt.Figure, stem: str, outdir: Path) -> None:
    """Save *fig* as PNG (and TikZ if possible)."""
    outdir.mkdir(exist_ok=True)
    fig.savefig(outdir / f"{stem}.png", dpi=_DEF_DPI, bbox_inches="tight")
    if _SAVE_AS_TIKZ:
        tikzplotlib.save(outdir / f"{stem}.tex")
    plt.close(fig)


# ─────────────────────────── load run‑log ─────────────────────────────
RUNLOG = Path("run_logs.json")
OUTDIR = Path("figs_tikz")
if not RUNLOG.exists():
    raise FileNotFoundError(RUNLOG)

runs = pd.json_normalize(json.loads(RUNLOG.read_text()))

# ensure required columns exist -------------------------------------------------
need_cols = [
    "finished", "gates_passed", "lap_time", "solver_ms", "speed", "pos_t",
    "gate_pos_t", "gate_quat_t", "obs_pos_t", "crash_pos", "mass_deviation",
]
for c in need_cols:
    if c not in runs.columns:
        runs[c] = None

succ = runs["finished"].astype(bool)

n_total, n_success = len(runs), int(succ.sum())
print("────────────── run‑log summary ──────────────")
print(f"Total runs          : {n_total}")
print(f"Successful runs     : {n_success}")
print(f"Failed runs         : {n_total - n_success}")
if n_success:
    print(f"Average lap‑time    : {runs.loc[succ, 'lap_time'].mean():.3f} s")
print("──────────────────────────────────────────────")

# derived columns --------------------------------------------------------------
runs["avg_speed"]  = runs["speed"].map(lambda s: np.mean(s) if isinstance(s, list) and len(s) else np.nan)
runs["peak_speed"] = runs["speed"].map(lambda s: np.max(s)  if isinstance(s, list) and len(s) else np.nan)

# colour normaliser for speed‑coloured paths
_all_speeds = np.hstack(runs.loc[succ, "speed"].dropna().to_list()) if n_success else np.array([0, 1])
_vmin, _vmax = float(_all_speeds.min()), float(_all_speeds.max())

# ─────────────────────────────── figures ──────────────────────────────────────

## 1 ▸ Lap‑time distribution ---------------------------------------------------
if n_success:
    fig, ax = plt.subplots(figsize=(3.2, 2.8))
    bp = ax.boxplot(runs.loc[succ, "lap_time"], widths=0.35, showfliers=False, patch_artist=True)
    for patch in bp["boxes"]:
        patch.set_facecolor("#8ecae6")
    ax.violinplot(runs.loc[succ, "lap_time"], positions=[1.45], widths=0.55, showmeans=False)
    ax.set_ylabel("Lap‑time [s]")
    ax.set_xticks([])
    _save(fig, "01_box_violin_laptime", OUTDIR)

## 2 ▸ Empirical CDF -----------------------------------------------------------
if n_success:
    fig, ax = plt.subplots(figsize=(3.2, 2.8))
    t_sorted = np.sort(runs.loc[succ, "lap_time"].values)
    ax.step(t_sorted, np.arange(1, len(t_sorted)+1)/len(t_sorted), where="post")
    ax.set_xlabel("Lap‑time [s]")
    ax.set_ylabel("Cumulative probability")
    ax.grid(ls=":", alpha=0.7)
    _save(fig, "02_cdf_laptime", OUTDIR)

## 3 ▸ Solver‑time histogram ---------------------------------------------------
all_solver = np.concatenate(runs["solver_ms"].dropna().to_list()) if len(runs) else np.empty(0)
if all_solver.size:
    fig, ax = plt.subplots(figsize=(3.2, 2.6))
    ax.hist(all_solver, bins=60, color="#8d99ae", edgecolor="white", log=True)
    ax.axvline(20, ls="--", color="k", lw=1.2, label="20 ms budget")
    ax.text(0.97, 0.9, f">20 ms: {100*(all_solver>20).mean():.1f}%", transform=ax.transAxes,
            ha="right", va="top", fontsize=8,
            bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="k", alpha=0.6))
    ax.set_xlabel("Solver wall‑time per step [ms]")
    ax.set_ylabel("Count (log)")
    ax.legend()
    _save(fig, "03_hist_solver", OUTDIR)

## 4 ▸ Lap‑time vs avg‑speed scatter ------------------------------------------
if n_success:
    fig, ax = plt.subplots(figsize=(3.2, 2.6))
    sc = ax.scatter(runs.loc[succ, "avg_speed"], runs.loc[succ, "lap_time"],
                    c=runs.loc[succ, "avg_speed"], cmap="viridis", s=25)
    ax.set_xlabel("Average speed [m/s]")
    ax.set_ylabel("Lap‑time [s]")
    ax.set_title("Lap‑time vs avg‑speed")
    fig.colorbar(sc, ax=ax, label="Avg speed [m/s]")
    _save(fig, "04_scatter_laptime_vs_avgspeed", OUTDIR)

## 5 ▸ Crash‑location heat‑map -------------------------------------------------
crash_xy: List[List[float]] = [pos[:2] for pos in runs.loc[~succ, "crash_pos"].dropna() if len(pos)>=2]
if crash_xy:
    crash_arr = np.asarray(crash_xy)
    fig, ax = plt.subplots(figsize=(3.0, 3.0))
    hb = ax.hexbin(crash_arr[:,0], crash_arr[:,1], gridsize=60, cmap="inferno", mincnt=1)
    ax.set_aspect("equal")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    fig.colorbar(hb, ax=ax, pad=0.01).set_label("Crash count")
    ax.set_title("Crash location density")
    _save(fig, "05_heatmap_crashes", OUTDIR)

## 6 ▸ Gate / obstacle layout & speed‑paths -----------------------------------
if len(runs):
    half_gate, half_obs = 0.45/2, 0.10/2
    beige = "#F5DEB3"  # obstacle colour

    def yaw_from_quat(q: np.ndarray) -> float:
        x,y,z,w = q
        return np.arctan2(2*(w*z + x*y), 1 - 2*(y*y + z*z))

    fig, ax = plt.subplots(figsize=(4.2,3.6))
    ax.set_aspect("equal")
    cmap, norm = plt.get_cmap("viridis"), plt.Normalize(_vmin, _vmax)

    for _, row in runs.iterrows():
        if not row["pos_t"]:
            continue
        pos_arr, spd_arr = np.asarray(row["pos_t"]), np.asarray(row["speed"])
        segs = np.stack([pos_arr[:-1,:2], pos_arr[1:,:2]], axis=1)
        lc = LineCollection(segs, cmap=cmap, norm=norm, linewidth=1.2, alpha=0.6, zorder=1)
        lc.set_array(spd_arr[:-1])
        ax.add_collection(lc)

        # gates (last‑seen)
        if row["gate_pos_t"]:
            g_pos, g_quat = np.asarray(row["gate_pos_t"][-1]), np.asarray(row["gate_quat_t"][-1])
            for pos, quat in zip(g_pos, g_quat):
                dir_vec = np.array([np.cos(yaw_from_quat(quat)), np.sin(yaw_from_quat(quat))])
                p0, p1 = pos[:2] - half_gate*dir_vec, pos[:2] + half_gate*dir_vec
                ax.plot([p0[0], p1[0]],[p0[1], p1[1]], color="#023047", alpha=0.15, lw=2, zorder=2)

        # obstacles (last‑seen)
        if row["obs_pos_t"]:
            obs = np.asarray(row["obs_pos_t"][-1])
            for cx, cy in obs[~np.all(obs==0, axis=1), :2]:
                ax.add_patch(patches.Rectangle((cx-half_obs, cy-half_obs), 2*half_obs, 2*half_obs,
                                 linewidth=0, facecolor="#BE8C2F", alpha=0.28, zorder=2))

        # crash marker (highest z‑order)
        if not row["finished"] and isinstance(row["crash_pos"], list) and len(row["crash_pos"])>=2:
            ax.plot(row["crash_pos"][0], row["crash_pos"][1], "x", color="#c1121f", ms=6, mew=1.4, zorder=4)

    fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax, label="Speed [m/s]")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_title("Paths coloured by instantaneous speed")
    _save(fig, "06_paths_speed_gates_obs", OUTDIR)

## 7 ▸ Mass‑deviation vs outcome ----------------------------------------------
if runs["mass_deviation"].notna().any():
    fig, ax = plt.subplots(figsize=(3.2,2.6))
    data = [runs.loc[succ, "mass_deviation"].dropna(), runs.loc[~succ, "mass_deviation"].dropna()]
    ax.boxplot(data, labels=["Success","Crash"], patch_artist=True,
               boxprops=dict(facecolor="#8ecae6", alpha=0.6))
    ax.axhline(0, color="k", lw=0.8, ls=":")
    ax.set_ylabel("Mass deviation [kg]")
    ax.set_title("Drone mass deviation vs outcome")
    _save(fig, "07_mass_dev_vs_outcome", OUTDIR)

## 8 ▸ Avg‑speed vs peak‑speed (successful) -----------------------------------
if n_success:
    fig, ax = plt.subplots(figsize=(3.2,2.6))
    ax.scatter(runs.loc[succ, "avg_speed"], runs.loc[succ, "peak_speed"],
               s=25, c="#577590", alpha=0.7)
    ax.set_xlabel("Avg speed [m/s]")
    ax.set_ylabel("Peak speed [m/s]")
    ax.set_title("Avg vs peak speed (successful)")
    _save(fig, "08_scatter_avg_vs_peak", OUTDIR)

## 9 ▸ Speed & solver‑time series (fastest run) -------------------------------
if n_success:
    idx_fast = runs.loc[succ, "lap_time"].idxmin()
    t, vel, thr = np.asarray(runs.at[idx_fast, "t"]), np.asarray(runs.at[idx_fast, "speed"]), np.asarray(runs.at[idx_fast, "solver_ms"])
    fig, ax1 = plt.subplots(figsize=(4.2,2.8))
    ax1.plot(t, vel, label="‖v‖", lw=1.5, color="#0077b6")
    ax1.set_xlabel("Time [s]")
    ax1.set_ylabel("Speed [m/s]")
    ax1.grid(ls=":", alpha=0.5)

    ax2 = ax1.twinx()
    ax2.plot(t, thr, "r--", lw=1.0, alpha=0.7, label="solve ms")
    ax2.set_ylabel("Solver time [ms]")

    if len(vel):
        p_idx = int(np.argmax(vel))
        ax1.annotate(f"peak {vel[p_idx]:.2f} m/s", xy=(t[p_idx], vel[p_idx]),
                     xytext=(t[p_idx]+0.2, vel[p_idx]+0.2),
                     arrowprops=dict(arrowstyle="->", lw=0.8), fontsize=7)

    fig.legend(loc="upper center", ncol=2, frameon=False)
    _save(fig, "09_timeseries_speed_solver", OUTDIR)

print("✓ All figures written to", OUTDIR.resolve())
