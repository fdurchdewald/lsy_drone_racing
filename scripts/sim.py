"""Simulate the competition as in the IROS 2022 Safe Robot Learning competition.

Run as:

    $ python scripts/sim.py --config level0.toml

Look for instructions in `README.md` and in the official documentation.
"""

from __future__ import annotations

import logging
from pathlib import Path
import mujoco
from typing import TYPE_CHECKING

import fire
import gymnasium
from gymnasium.wrappers.jax_to_numpy import JaxToNumpy
import numpy as np

from lsy_drone_racing.utils.utils import draw_line, load_config, load_controller, draw_tunnel, draw_tube_splines, draw_tube_dynamic

if TYPE_CHECKING:
    from ml_collections import ConfigDict
    from lsy_drone_racing.control.controller import Controller
    from lsy_drone_racing.envs.drone_race import DroneRaceEnv


logger = logging.getLogger(__name__)


def simulate(
    config: str = "level0.toml",
    controller: str | None = None,
    n_runs: int = 1,
    gui: bool | None = True,
) -> list[float]:
    """Evaluate the drone controller over multiple episodes.

    Args:
        config: The path to the configuration file. Assumes the file is in `config/`.
        controller: The name of the controller file in `lsy_drone_racing/control/` or None. If None,
            the controller specified in the config file is used.
        n_runs: The number of episodes.
        gui: Enable/disable the simulation GUI.

    Returns:
        A list of episode times.
    """
    # Load configuration
    config = load_config(Path(__file__).parents[1] / "config" / config)
    if gui is None:
        gui = config.sim.gui
    else:
        config.sim.gui = gui

    # Load the controller class
    control_path = Path(__file__).parents[1] / "lsy_drone_racing/control"
    controller_path = control_path / (controller or config.controller.file)
    controller_cls = load_controller(controller_path)

    # Create the racing environment
    env: DroneRaceEnv = gymnasium.make(
        config.env.id,
        freq=config.env.freq,
        sim_config=config.sim,
        sensor_range=config.env.sensor_range,
        control_mode=config.env.control_mode,
        track=config.env.track,
        disturbances=config.env.get("disturbances"),
        randomizations=config.env.get("randomizations"),
        seed=config.env.seed,
    )
    env = JaxToNumpy(env)

    ep_times: list[float] = []
    for _ in range(n_runs):
        obs, info = env.reset()
        controller: Controller = controller_cls(obs, info, config)

        # --- Prepare a permanent sample of the spline trajectory ---
        # num_samples = 200
        # t_lin = np.linspace(0, 1, num_samples)
        path_points = controller.get_trajectory()

        # --- Prepare storage for the actually flown path ---
        flown_positions: list[np.ndarray] = []

        i = 0
        fps = 60

        while True:
            curr_time = i / config.env.freq

            # Compute and apply control
            action = controller.compute_control(obs, info)
            obs, reward, terminated, truncated, info = env.step(action)
            controller_finished = controller.step_callback(
                action, obs, reward, terminated, truncated, info
            )

            # Record current drone position (ground truth state)
            flown_positions.append(obs["pos"])

            # Draw both the planned path and the flown path every frame
            if config.sim.gui:

                draw_tube_dynamic(env, controller.tunnel_cache,
                                        n_circle=12, thickness=2.0)


                # planned path: grün, Stärke 2
                draw_line(env, path_points,
                        rgba=np.array([0.0, 1.0, 0.0, 1.0]),
                        min_size=2.0, max_size=2.0)

                # tatsächlich geflogener Pfad: rot, Stärke 1.5
                if len(flown_positions) >= 2:
                    fp = np.vstack(flown_positions)
                    draw_line(env, fp,
                              rgba=np.array([1.0, 0.0, 0.0, 1.0]),
                              min_size=1.5, max_size=1.5)
                    
                # viewer = env.unwrapped.sim.viewer          # py-mujoco Viewer
                # viewer.scn.ngeom = 0                       # löscht alle Geoms
                # viewer.markers.clear()   
                env.render()

            if terminated or truncated or controller_finished:
                break

            i += 1

        # Episode bookkeeping
        controller.episode_callback()
        log_episode_stats(obs, info, config, curr_time)
        controller.episode_reset()
        ep_times.append(curr_time if obs["target_gate"] == -1 else None)

    env.close()
    return ep_times


def log_episode_stats(obs: dict, info: dict, config: ConfigDict, curr_time: float):
    """Log the statistics of a single episode."""
    gates_passed = obs["target_gate"]
    if gates_passed == -1:  # The drone has passed the final gate
        gates_passed = len(config.env.track.gates)
    finished = gates_passed == len(config.env.track.gates)
    logger.info(
        f"Flight time (s): {curr_time}\n"
        f"Finished: {finished}\n"
        f"Gates passed: {gates_passed}\n"
    )


if __name__ == "__main__":
    logging.basicConfig()
    logging.getLogger("lsy_drone_racing").setLevel(logging.INFO)
    logger.setLevel(logging.INFO)
    fire.Fire(simulate, serialize=lambda _: None)