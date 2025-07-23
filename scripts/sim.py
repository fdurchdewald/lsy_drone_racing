"""Simulate the LSY benchmark and write rich logs (one JSON entry per episode).

Creates/updates:
    run_logs.json   – appended after every episode
    sim.log         – INFO-level execution log
"""
from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, List

import fire
import gymnasium
import numpy as np
from gymnasium.wrappers.jax_to_numpy import JaxToNumpy

from lsy_drone_racing.utils.utils import (
    draw_line,
    draw_point,
    draw_tunnel_bounds,
    load_config,
    load_controller,
)

if TYPE_CHECKING:
    from ml_collections import ConfigDict

    from lsy_drone_racing.control.controller import Controller
    from lsy_drone_racing.envs.drone_race import DroneRaceEnv

# ─────────────────────────  helper  ──────────────────────────
def _to_py(obj: object) -> object:
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


def _dump(log_list: list, path: Path):
    path.write_text(json.dumps(_to_py(log_list), indent=2))


# ────────────────────────  dataclass  ─────────────────────────
@dataclass
class RunLog:
    """Log data for a single simulation run.

    Attributes:
        lap_time (float | None): Time to complete the lap, or None if not finished.
        finished (bool): Whether the drone finished the race.
        gates_passed (int): Number of gates passed.
        seed (int): Random seed used for the episode.
        gate_pos0 (list[list[float]]): Initial positions of gates.
        gate_quat0 (list[list[float]]): Initial orientations of gates.
        obs_pos0 (list[list[float]]): Initial positions of obstacles.
        t (List[float]): Time stamps for each step.
        speed (List[float]): Drone speed at each step.
        min_gate_dist (List[float]): Minimum distance to gates at each step.
        min_obs_dist (List[float]): Minimum distance to obstacles at each step.
        solver_ms (List[float]): Controller solver time per step (ms).
        gate_pos_t (List[List[List[float]]]): Gate positions over time.
        gate_quat_t (List[List[List[float]]]): Gate orientations over time.
        obs_pos_t (List[List[List[float]]]): Obstacle positions over time.
        crash_gate (int | None): Gate index at crash, if any.
        crash_pos (list[float] | None): Position at crash, if any.
        crash_gate_dist (float | None): Distance to gate at crash, if any.
        crash_obs_dist (float | None): Distance to obstacle at crash, if any.
        drone_mass_default (float): Default drone mass.
        drone_mass (float): Actual drone mass.
        mass_deviation (float): Difference between actual and default mass.
        pos_t (List[List[float]]): Drone positions over time.
    """
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
    drone_mass_default: float = 0.0
    drone_mass: float = 0.0
    mass_deviation: float = 0.0
    pos_t: List[List[float]] = field(default_factory=list)


# ─────────────────────  main simulate()  ──────────────────────
logger = logging.getLogger(__name__)
def simulate(
    config: str = "level2.toml",
    controller: str | None = None,
    n_runs: int = 1000,
    gui: bool | None = False,
    visualize: bool = False,
) -> list[float]:
    """Simulate the LSY drone racing benchmark for a given number of episodes.

    Parameters
    ----------
    config : str
        Path to the configuration file.
    controller : str | None
        Path to the controller file or None to use default from config.
    n_runs : int
        Number of simulation episodes to run.
    gui : bool | None
        Whether to enable GUI visualization.
    visualize : bool
        Whether to visualize trajectories and waypoints.

    Returns:
    -------
    list[float]
        List of lap times for each episode (None if not finished).
    """
    cfg: ConfigDict = load_config(Path(__file__).parents[1] / "config" / config)
    cfg.sim.gui = gui if gui is not None else cfg.sim.gui

    controller_cls = load_controller(
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
            gate_pos_t=[], gate_quat_t=[], obs_pos_t=[], pos_t=[]
        )

        default_mass  = env.unwrapped.drone_mass
        current_mass  = env.unwrapped.sim.data.params.mass
        default_mass = float(np.asarray(env.unwrapped.drone_mass).ravel()[0])
        current_mass_arr = np.asarray(env.unwrapped.sim.data.params.mass)
        current_mass     = float(current_mass_arr.ravel()[0])
        run.drone_mass_default = float(default_mass)
        run.drone_mass         = float(current_mass)
        run.mass_deviation     = float(current_mass - default_mass)

        controller: Controller = controller_cls(obs, info, cfg)
        step = 0
        i = 0
        fps = 60 
        while True:
            t_now = step / cfg.env.freq
            act = controller.compute_control(obs, info)
            obs, reward, term, trunc, info = env.step(act)

            # traces
            run.t.append(t_now)
            run.pos_t.append(obs["pos"].tolist())
            run.speed.append(float(np.linalg.norm(obs["vel"])))
            run.min_gate_dist.append(float(np.linalg.norm(obs["gates_pos"] - obs["pos"], axis=1).min()))
            valid_obs = obs["obstacles_pos"][~np.all(obs["obstacles_pos"] == 0, axis=1)]
            if valid_obs.size:
                radii = np.asarray(cfg.env.get("obstacle_radius", [0.1] * len(valid_obs)))
                surf = np.linalg.norm(valid_obs - obs["pos"], axis=1) - radii[: len(valid_obs)]
                run.min_obs_dist.append(float(surf.min()))
            else:
                run.min_obs_dist.append(np.inf)
            run.solver_ms.append(float(controller.get_last_solve_ms()))
            if not run.gate_pos_t or not np.allclose(obs["gates_pos"], run.gate_pos_t[-1]):
                run.gate_pos_t.append(obs["gates_pos"].tolist())
                run.gate_quat_t.append(obs["gates_quat"].tolist())

            if not run.obs_pos_t or not np.allclose(obs["obstacles_pos"], run.obs_pos_t[-1]):
                run.obs_pos_t.append(obs["obstacles_pos"].tolist())


            if cfg.sim.gui:
                if visualize:
                    # draw given trajectory
                    draw_line(env, controller.get_trajectory(), rgba=np.array([0.0, 1.0, 0.0, 1.0]),
                            min_size=2.0, max_size=2.0)
                    # draw waypoints
                    for point in controller.get_waypoints():
                        draw_point(env, point, rgba=np.array([1.0, 0.5, 0.0, 1.0]))
                    # draw drone trajectory
                    if len(controller.get_drone_trajectory()) >= 2:  
                        draw_line(env, controller.get_drone_trajectory(),
                                rgba=np.array([1.0, 0.0, 0.0, 1.0]),
                                min_size=2.0, max_size=2.0)
                    # draw planned trajectory
                    draw_line(env, controller.get_planned_trajectory(),
                            rgba=np.array([1.0, 1.0, 0.0, 1.0]),
                            min_size=3.0, max_size=3.0)
                    # draw tunnel bounds
                    draw_tunnel_bounds(env, controller.get_tunnel_regions())
                if ((i * fps) % cfg.env.freq) < fps:
                    env.render()
                i += 1

            if term or trunc or controller.step_callback(act, obs, reward, term, trunc, info):
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
        log_episode_stats(obs, info, cfg, run.lap_time)

    env.close()
    print("✓ run_logs.json at", out.resolve())
    return lap_times


def log_episode_stats(obs: dict, info: dict, config: ConfigDict, curr_time: float):
    """Log the statistics of a single episode."""
    gates_passed = obs["target_gate"]
    if gates_passed == -1:  # The drone has passed the final gate
        gates_passed = len(config.env.track.gates)
    finished = gates_passed == len(config.env.track.gates)
    logger.info(
        f"Flight time (s): {curr_time}\nFinished: {finished}\nGates passed: {gates_passed}\n"
    )




if __name__ == "__main__":
    logging.basicConfig()
    logging.getLogger("lsy_drone_racing").setLevel(logging.INFO)
    logger.setLevel(logging.INFO)
    fire.Fire(simulate, serialize=lambda _: None)