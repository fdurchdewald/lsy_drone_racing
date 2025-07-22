"""
Simulate the LSY benchmark and write rich logs (one JSON entry per episode).

Creates/updates:
    run_logs.json   – appended after every episode
    sim.log         – INFO-level execution log
"""
from __future__ import annotations
import json, logging
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import TYPE_CHECKING, List

import fire, gymnasium, numpy as np
from gymnasium.wrappers.jax_to_numpy import JaxToNumpy
from lsy_drone_racing.utils.utils import *

if TYPE_CHECKING:
    from lsy_drone_racing.control.controller import Controller
    from lsy_drone_racing.envs.drone_race import DroneRaceEnv
    from ml_collections import ConfigDict

# ─────────────────────────  helper  ──────────────────────────
def _to_py(obj):
    """Convert NumPy scalars/arrays inside nested containers to Python types."""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.generic,)):
        return obj.item()
    if isinstance(obj, list):
        return [_to_py(x) for x in obj]
    if isinstance(obj, dict):
        return {k: _to_py(v) for k, v in obj.items()}
    return obj


def _dump(log_list, path: Path):
    path.write_text(json.dumps(_to_py(log_list), indent=2))


# ────────────────────────  dataclass  ─────────────────────────
@dataclass
class RunLog:
    lap_time: float | None
    finished: bool
    gates_passed: int
    seed: int
    # initial geometry (nominal)
    gate_pos0: list[list[float]]
    gate_quat0: list[list[float]]
    obs_pos0:  list[list[float]]
    # per-step traces
    t: List[float]
    speed: List[float]
    min_gate_dist: List[float]
    min_obs_dist: List[float]
    solver_ms: List[float]
    gate_pos_t:  List[List[List[float]]]   # N × 4 × 3
    gate_quat_t: List[List[List[float]]]   # N × 4 × 4
    obs_pos_t:   List[List[List[float]]]   # N × 4 × 3
    # crash snapshot
    crash_gate: int | None = None
    crash_pos:  list[float] | None = None
    crash_gate_dist: float | None = None
    crash_obs_dist:  float | None = None


# ─────────────────────  main simulate()  ──────────────────────
logger = logging.getLogger(__name__)
def simulate(
    config: str = "level2.toml",
    controller: str | None = None,
    n_runs: int = 1000,
    gui: bool | None = False,
    visualize: bool = False,
) -> list[float]:

    cfg: ConfigDict = load_config(Path(__file__).parents[1] / "config" / config)
    cfg.sim.gui = gui if gui is not None else cfg.sim.gui

    ctl_cls = load_controller(
        Path(__file__).parents[1] / "lsy_drone_racing/control" / (controller or cfg.controller.file)
    )

    env: DroneRaceEnv = gymnasium.make(
        cfg.env.id,
        freq=cfg.env.freq,
        sim_config=cfg.sim,
        sensor_range=cfg.env.sensor_range,
        control_mode=cfg.env.control_mode,
        track=cfg.env.track,
        disturbances=cfg.env.get("disturbances"),
        randomizations=cfg.env.get("randomizations"),
        seed=cfg.env.seed,
    )
    env = JaxToNumpy(env)

    logs: list[RunLog] = []
    lap_times: list[float] = []
    out = Path("run_logs.json")

    for ep in range(n_runs):
        obs, info = env.reset()
        run = RunLog(
            lap_time=None, finished=False, gates_passed=0, seed=int(info.get("seed", -1)),
            gate_pos0=obs["gates_pos"].tolist(),
            gate_quat0=obs["gates_quat"].tolist(),
            obs_pos0=obs["obstacles_pos"].tolist(),
            t=[], speed=[], min_gate_dist=[], min_obs_dist=[], solver_ms=[],
            gate_pos_t=[], gate_quat_t=[], obs_pos_t=[]
        )

        ctl: Controller = ctl_cls(obs, info, cfg)
        step = 0
        while True:
            t_now = step / cfg.env.freq
            act = ctl.compute_control(obs, info)
            obs, reward, term, trunc, info = env.step(act)

            # traces
            run.t.append(t_now)
            run.speed.append(float(np.linalg.norm(obs["vel"])))
            run.min_gate_dist.append(float(np.linalg.norm(obs["gates_pos"] - obs["pos"], axis=1).min()))
            valid_obs = obs["obstacles_pos"][~np.all(obs["obstacles_pos"] == 0, axis=1)]
            if valid_obs.size:
                radii = np.asarray(cfg.env.get("obstacle_radius", [0.1] * len(valid_obs)))
                surf = np.linalg.norm(valid_obs - obs["pos"], axis=1) - radii[: len(valid_obs)]
                run.min_obs_dist.append(float(surf.min()))
            else:
                run.min_obs_dist.append(np.inf)
            run.solver_ms.append(float(ctl.get_last_solve_ms()))
            run.gate_pos_t.append(obs["gates_pos"].tolist())
            run.gate_quat_t.append(obs["gates_quat"].tolist())
            run.obs_pos_t.append(obs["obstacles_pos"].tolist())

            if cfg.sim.gui and visualize:
                _render(env, ctl)

            if term or trunc or ctl.step_callback(act, obs, reward, term, trunc, info):
                break
            step += 1

        gates_passed = obs["target_gate"] if obs["target_gate"] >= 0 \
                       else len(cfg.env.track.gates)
        run.gates_passed = int(gates_passed)
        run.finished     = gates_passed == len(cfg.env.track.gates)
        run.lap_time     = t_now if run.finished else None
        if not run.finished:
            run.crash_gate = int(obs["target_gate"])
            run.crash_pos  = obs["pos"].tolist()
            run.crash_gate_dist = run.min_gate_dist[-1]
            run.crash_obs_dist  = run.min_obs_dist[-1]

        logs.append(run)
        lap_times.append(run.lap_time)

        # flush
        _dump([asdict(r) for r in logs], out)
        logger.info("Episode %d/%d logged (file size %.1f kB)",
                    ep+1, n_runs, out.stat().st_size/1024)

    env.close()
    print("✓ run_logs.json at", out.resolve())
    return lap_times


# ───────────────────────  optional drawing  ───────────────────
def _render(env, ctl):
    draw_line(env, ctl.get_trajectory(),            [0,1,0,1], 2, 2)
    if len(tr := ctl.get_drone_trajectory()) > 1:
        draw_line(env, tr,                          [1,0,0,1], 2, 2)
    draw_line(env, ctl.get_planned_trajectory(),    [0,0,1,1], 2, 2)
    draw_tunnel_regions_from_corners(env, ctl.get_tunnel_regions())
    for p in ctl.get_waypoints():
        draw_point(env, p, [1,0.5,0,1])
    env.render()


# ─────────────────────────  entry point  ───────────────────────
if __name__ == "__main__":
    logging.basicConfig(filename="sim.log",
                        filemode="w",
                        level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    logging.getLogger().addHandler(logging.StreamHandler())
    fire.Fire(simulate, serialize=lambda _: None)
