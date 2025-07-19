from __future__ import annotations

import logging
from pathlib import Path
import mujoco
from typing import TYPE_CHECKING

import fire
import gymnasium
from gymnasium.wrappers.jax_to_numpy import JaxToNumpy
import numpy as np

import time
from lsy_drone_racing.utils.utils import *


if TYPE_CHECKING:
    from ml_collections import ConfigDict
    from lsy_drone_racing.control.controller import Controller
    from lsy_drone_racing.envs.drone_race import DroneRaceEnv


logger = logging.getLogger(__name__)


def simulate(
    config: str = "level2.toml",
    controller: str | None = None,
    n_runs: int = 1,
    gui: bool | None = True,
    visualize: bool = True,
    PARAM_DICT: dict[str, float] | None = None,
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
        seed=int(time.time()),
    )
    env = JaxToNumpy(env)

    ep_times: list[float] = []

    list_times = []
    list_gates_passed = []
    for _ in range(n_runs):
        obs, info = env.reset()
        controller: Controller = controller_cls(obs, info, config)

        # --- Prepare a permanent sample of the spline trajectory ---
        # num_samples = 200
        # t_lin = np.linspace(0, 1, num_samples)

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

            if terminated or truncated or controller_finished:
                dist_z = controller.get_distz()
                break

            i += 1

        # Episode bookkeeping
        controller.episode_callback()
        log_episode_stats(obs, info, config, curr_time)
        controller.episode_reset()
        ep_times.append(curr_time if obs["target_gate"] == -1 else None)
        gates_passed = obs["target_gate"]
        if gates_passed == -1:  # The drone has passed the final gate
            gates_passed = 4

        list_gates_passed.append(int(gates_passed))
    env.close()
    return ep_times, list_gates_passed, dist_z


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