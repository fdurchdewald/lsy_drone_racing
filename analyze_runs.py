#!/usr/bin/env python3
"""
Analyse `run_logs.json` and create figures for the paper.
Figures are saved as TikZ (.tex) if tikzplotlib is present,
otherwise as high-res PNG.

Required log keys (per episode):
    finished, gates_passed, lap_time, solver_ms, speed,
    gate_pos_t (Nx4x3) – for the heat-map.
"""

from pathlib import Path
import json, warnings, itertools
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# ---------------------------------------------------------------------------
#  backend: TikZ if available, else PNG
# ---------------------------------------------------------------------------
try:
    import tikzplotlib                # patched build (0.10.1.post13+)
    _SAVE_AS_TIKZ = True
except ModuleNotFoundError:
    _SAVE_AS_TIKZ = False
    warnings.warn("tikzplotlib not available – still saving PNG files", RuntimeWarning)

def _save(fig, stem: str, outdir: Path, show_preview: bool = True):
    outdir.mkdir(exist_ok=True)
    png_path = outdir / f"{stem}.png"
    fig.savefig(png_path, dpi=300, bbox_inches="tight")

    if _SAVE_AS_TIKZ:
        tikz_path = outdir / f"{stem}.tex"
        tikzplotlib.save(tikz_path)

    if show_preview:
        plt.figure(fig.number)
        plt.pause(0.5)
        plt.close(fig)
    else:
        plt.close(fig)




# ---------------------------------------------------------------------------
#  load run-log
# ---------------------------------------------------------------------------
RUNLOG = Path("run_logs.json")
OUTDIR = Path("figs_tikz")
if not RUNLOG.exists():
    raise FileNotFoundError(RUNLOG)

runs = pd.json_normalize(json.loads(RUNLOG.read_text()))
succ = runs["finished"].astype(bool)

# ---------------------------------------------------------------------------
#  console summary
# ---------------------------------------------------------------------------
n_total     = len(runs)
n_success   = succ.sum()
avg_time    = runs.loc[succ, "lap_time"].mean()
print(f"Total runs          : {n_total}")
print(f"Successful runs     : {n_success}")
print(f"Failed runs         : {n_total - n_success}")
print(f"Success percentage  : {100*n_success/n_total:.1f} %")
print(f"Average time (succ) : {avg_time:.2f} s")

# ---------------------------------------------------------------------------
#  FIG 1 • lap-time distribution (box + violin)
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(3, 2.8))
bp = ax.boxplot(runs.loc[succ, "lap_time"],
                showfliers=False, widths=0.3, patch_artist=True)
for patch in bp["boxes"]:
    patch.set_facecolor("#8ecae6")
ax.violinplot(runs.loc[succ, "lap_time"],
              positions=[1.45], widths=0.5, showmeans=False)
ax.set_ylabel("Lap-time [s]")
ax.set_xticks([]); ax.set_xlim(0.5, 2)
_save(fig, "01_box_violin_laptime", OUTDIR)

# ---------------------------------------------------------------------------
#  FIG 2 • empirical CDF of lap times
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(3, 2.8))
times = np.sort(runs.loc[succ, "lap_time"].values)
ax.step(times, np.arange(1, len(times)+1)/len(times), where="post")
ax.set_xlabel("Lap-time [s]"); ax.set_ylabel("Cumulative probability")
ax.grid(True, ls=":")
_save(fig, "02_cdf_laptime", OUTDIR)

# ---------------------------------------------------------------------------
#  FIG 3 • survival curve per gate
# ---------------------------------------------------------------------------
N_G = int(runs["gates_passed"].max())
survival = [(runs["gates_passed"] > k).mean() for k in range(N_G)]
fig, ax = plt.subplots(figsize=(3, 2.8))
ax.step(range(1, N_G+1), survival, where="post", marker="o")
ax.set_xlabel("Gate index"); ax.set_ylabel("$P(\\text{survive})$")
ax.set_ylim(0, 1.05); ax.grid(ls=":")
_save(fig, "03_survival_per_gate", OUTDIR)

# ---------------------------------------------------------------------------
#  FIG 4 • scatter: lap-time vs. gates passed
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(3, 2.8))
colors = succ.map({True: "#55a630", False: "#c1121f"})
ax.scatter(runs["gates_passed"], runs["lap_time"], c=colors, alpha=.7, s=15)
ax.set_xlabel("Gates passed"); ax.set_ylabel("Lap-time [s]")
ax.set_xlim(-0.2, N_G+0.2); ax.grid(ls=":")
_save(fig, "04_scatter_time_vs_gates", OUTDIR)

# ---------------------------------------------------------------------------
#  FIG 5 • solver wall-time histogram
# ---------------------------------------------------------------------------
all_solver = np.hstack(runs["solver_ms"].to_list())
fig, ax = plt.subplots(figsize=(3, 2.8))
ax.hist(all_solver, bins=40)
ax.axvline(20, ls="--", color="k", label="20 ms real-time budget")
ax.set_xlabel("Solver wall-time per step [ms]")
ax.set_ylabel("Count"); ax.legend()
_save(fig, "05_hist_solver", OUTDIR)

# ---------------------------------------------------------------------------
#  FIG 6 • 2-D trajectory heat-map (xy density)
# ---------------------------------------------------------------------------
# stack all xy-tracks (skip very long/crashed runs to keep size reasonable)
tracks_xy = []
for tr in runs.loc[succ, "gate_pos_t"]:
    tr_arr = np.asarray(tr)[:, :, :2]         # N × 4 × 2
    tracks_xy.append(tr_arr.reshape(-1, 2))
xy_all = np.vstack(tracks_xy)
fig, ax = plt.subplots(figsize=(3, 3))
hb = ax.hexbin(xy_all[:,0], xy_all[:,1], gridsize=80, cmap="viridis",
               mincnt=5, linewidths=0)
ax.set_aspect("equal")
ax.set_xlabel("x [m]"); ax.set_ylabel("y [m]")
cbar = fig.colorbar(hb, ax=ax, pad=0.01)
cbar.set_label("Sample count")
_save(fig, "06_heatmap_xy_lines", OUTDIR)

# ---------------------------------------------------------------------------
#  FIG 7 • representative speed & throttle profile
# ---------------------------------------------------------------------------
if n_success:
    idx_fastest = runs.loc[succ, "lap_time"].idxmin()
    t = np.asarray(runs.at[idx_fastest, "t"])
    v = np.asarray(runs.at[idx_fastest, "speed"])
    thr = np.asarray(runs.at[idx_fastest, "solver_ms"])      # as proxy for control effort
    fig, ax1 = plt.subplots(figsize=(3.6, 2.6))
    ax1.plot(t, v, label="$||v||$", lw=1.2)
    ax1.set_xlabel("Time [s]"); ax1.set_ylabel("Speed [m/s]")
    ax2 = ax1.twinx()
    ax2.plot(t, thr, "r--", alpha=.6, label="solve ms")
    ax2.set_ylabel("Solver time [ms]")
    for gi in range(1, N_G):
        ax1.axvline(gi, ls=":", color="k", lw=0.6)
    ax1.legend(loc="upper left"); ax2.legend(loc="upper right")
    _save(fig, "07_timeseries_speed_solver", OUTDIR)

print("✓ Figures written to", OUTDIR.resolve())
