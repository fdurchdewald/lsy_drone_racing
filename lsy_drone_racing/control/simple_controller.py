from __future__ import annotations  # Python 3.10 type hints

from typing import TYPE_CHECKING

import numpy as np
from scipy.interpolate import CubicSpline

from lsy_drone_racing.control import Controller

if TYPE_CHECKING:
    from numpy.typing import NDArray


class SimpleController(Controller):
    def __init__(self, obs: dict[str, NDArray[np.floating]], info: dict, config: dict):
        print("SimpleController")
        super().__init__(obs, info, config)
        self._tick = 0
        self._freq = config.env.freq
        self._finished = False
        self.mv = 0.4;
        self.hold = 0.16;
        self.step = 0;
    

    def compute_control(
        self, obs: dict[str, NDArray[np.floating]], info: dict | None = None
    ) -> NDArray[np.floating]:
        print(obs["pos"])
        position = obs["pos"]
        print(obs["target_gate"])

        if self.step == 0:
            if position[2] < 0.47:
                vol = np.array([0, 0, self.mv], dtype=np.float32)
            else:
                self.step += 1
        if self.step == 1:
            if position[0] > 0.226:
                vol = np.array([-self.mv, 0, self.hold], dtype=np.float32)
            else:
                self.step += 1
        if self.step == 2:
            if position[1] > -0.55:
                vol = np.array([0, -self.mv, self.hold], dtype=np.float32)
            else:
                self.step += 1
        if self.step == 3:
            if position[0] < 0.95:
                vol = np.array([self.mv, 0, self.hold], dtype=np.float32)
            else: 
                self.step += 1
        if self.step == 4:
            if position[2] < 1.06:
                vol = np.array([0, 0, self.mv], dtype=np.float32)
            else:
                self.step += 1
        if self.step == 5:                
            if position[1] > -1.74:
                vol = np.array([0, -self.mv, self.hold], dtype=np.float32)
            else:
                self.step += 1
        if self.step == 6:
            if position[0] > -0.5:
                vol = np.array([-self.mv, 0, self.hold], dtype=np.float32)
            else:
                self.step += 1
        if self.step == 7:
            if position[1] < 0.16:
                vol = np.array([0, self.mv, self.hold], dtype=np.float32)
            else:
                vol = np.array([0, self.mv, self.hold], dtype=np.float32)
                self._finished = True




        # Rückgabeformat: [x, y, z, vx, vy, vz, ax, ay, az, yaw, rrate, prate, yrate]
        return np.concatenate((position, vol, np.zeros(7)), dtype=np.float32)

    def step_callback(
        self,
        action: NDArray[np.floating],
        obs: dict[str, NDArray[np.floating]],
        reward: float,
        terminated: bool,
        truncated: bool,
        info: dict,
    ) -> bool:
        return False

