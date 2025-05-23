#!/usr/bin/env python3
from pathlib import Path

import gymnasium
from lsy_drone_racing.utils import load_config
from lsy_drone_racing.envs.drone_race import DroneRaceEnv

def main():
    # 1) Config laden (Level2) als Path
    cfg_path = Path(__file__).parents[2] / "config" / "level2.toml"
    config = load_config(cfg_path)

    # 2) Env headless bauen
    #    setze gui=False direkt in sim_config ab
    sim_cfg = config.sim
    sim_cfg.gui = False
    env: DroneRaceEnv = gymnasium.make(
        config.env.id,
        freq=config.env.freq,
        sim_config=sim_cfg,
        sensor_range=config.env.sensor_range,
        control_mode=config.env.control_mode,
        track=config.env.track,
        disturbances=config.env.disturbances,
        randomizations=config.env.randomizations,
        seed=config.env.seed,
    )

    # 3) Reset und obs printen
    obs, _info = env.reset()
    print("=== Observation dict ===")
    for k, v in obs.items():
        if hasattr(v, "shape") and v.shape != ():
            sample = v.flatten()[:3]
        else:
            sample = v.item() if hasattr(v, "item") else v
        print(f"{k:15s}  shape={getattr(v, 'shape', None)}  sample={sample}")

    # 4) Cleanup
    env.close()

if __name__ == "__main__":
    main()
