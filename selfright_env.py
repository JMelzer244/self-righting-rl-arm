import numpy as np
import mujoco
import gymnasium as gym
from gymnasium import spaces

XML_PATH        = "scene.xml"
BASE_BODY       = "part_1"
JOINT_NAMES     = ["roll1", "roll2"]
START_HEIGHT    = 0.12
CONTROL_HZ      = 50
STAND_POSE      = np.array([0.0, 0.0])
SERVO_SPEED     = 4.0       # rad/s, MG996R ~4-6
SERVO_TORQUE_NM = 1.0       # N*m, mjcf has no limit so we cap it here

MAX_STEPS    = 300
SUCCESS_HOLD = 25
POSE_TOL     = 0.10


def _rand_quat(rng):
    u1, u2, u3 = rng.random(3)
    return np.array([
        np.sqrt(u1) * np.cos(2 * np.pi * u3),
        np.sqrt(1 - u1) * np.sin(2 * np.pi * u2),
        np.sqrt(1 - u1) * np.cos(2 * np.pi * u2),
        np.sqrt(u1) * np.sin(2 * np.pi * u3),
    ])


class SelfRightEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self):
        super().__init__()
        self.model = mujoco.MjModel.from_xml_path(XML_PATH)
        self.data = mujoco.MjData(self.model)
        self.rng = np.random.default_rng()

        self.model.actuator_forcelimited[:] = 1
        self.model.actuator_forcerange[:, 0] = -SERVO_TORQUE_NM
        self.model.actuator_forcerange[:, 1] =  SERVO_TORQUE_NM

        self.bid = self.model.body(BASE_BODY).id
        base_jnt = self.model.body(BASE_BODY).jntadr[0]
        self.base_qadr = self.model.jnt_qposadr[base_jnt]
        self.base_dof = self.model.jnt_dofadr[base_jnt]

        self.jadr = [self.model.joint(n).qposadr[0] for n in JOINT_NAMES]
        self.jdof = [self.model.joint(n).dofadr[0] for n in JOINT_NAMES]
        self.jrange = np.array([self.model.joint(n).range for n in JOINT_NAMES])  # (2,2)

        self.n_sub = max(1, round((1.0 / CONTROL_HZ) / self.model.opt.timestep))
        self.ctrl_dt = self.n_sub * self.model.opt.timestep
        self.max_dservo = SERVO_SPEED * self.ctrl_dt

        self.observation_space = spaces.Box(-np.inf, np.inf, (8,), np.float32)
        self.action_space = spaces.Box(-1.0, 1.0, (2,), np.float32)

        self.servo_target = np.zeros(2)
        self.cmd = np.zeros(2)
        self.last_action = np.zeros(2)
        self.steps = 0
        self.hold = 0

    def _R(self):
        return self.data.xmat[self.bid].reshape(3, 3)

    def _norm(self, ang):
        lo, hi = self.jrange[:, 0], self.jrange[:, 1]
        return 2.0 * (ang - lo) / (hi - lo) - 1.0

    def _obs(self):
        R = self._R()
        grav_body = -R[2, :]
        angvel = self.data.qvel[self.base_dof + 3:self.base_dof + 6]
        return np.concatenate([grav_body, angvel, self.cmd]).astype(np.float32)

    def _uprightness(self):
        return self._R()[2, 2]

    def reset(self, *, seed=None, options=None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        mujoco.mj_resetData(self.model, self.data)

        q = self.base_qadr
        self.data.qpos[q + 0:q + 2] = 0.0
        self.data.qpos[q + 2] = START_HEIGHT
        self.data.qpos[q + 3:q + 7] = _rand_quat(self.rng)
        mujoco.mj_forward(self.model, self.data)

        self.servo_target = self.data.qpos[self.jadr].copy()
        self.cmd = self._norm(self.servo_target)
        self.last_action[:] = 0.0
        self.steps = 0
        self.hold = 0
        return self._obs(), {}

    def step(self, action):
        action = np.clip(action, -1.0, 1.0)

        lo, hi = self.jrange[:, 0], self.jrange[:, 1]
        desired = lo + (action + 1.0) * 0.5 * (hi - lo)
        self.servo_target = np.clip(
            desired,
            self.servo_target - self.max_dservo,
            self.servo_target + self.max_dservo,
        )
        self.data.ctrl[:2] = self.servo_target
        self.cmd = self._norm(self.servo_target)

        for _ in range(self.n_sub):
            mujoco.mj_step(self.model, self.data)

        up = self._uprightness()
        gate = max(up, 0.0)

        smooth = np.sum((action - self.last_action) ** 2)
        pose_err = np.sum((self.data.qpos[self.jadr] - STAND_POSE) ** 2)
        jvel = np.sum(self.data.qvel[self.jdof] ** 2)

        reward = (
            up
            - 0.15 * smooth
            - 0.005 * np.sum(action ** 2)
            - gate * 0.30 * pose_err
            - gate * 0.02 * jvel
        )
        self.last_action = action.copy()

        base_av = np.linalg.norm(self.data.qvel[self.base_dof + 3:self.base_dof + 6])
        ok = (up > 0.95) and (pose_err < POSE_TOL) and (base_av < 0.5)
        self.hold = self.hold + 1 if ok else 0
        terminated = self.hold >= SUCCESS_HOLD
        if terminated:
            reward += 50.0

        self.steps += 1
        truncated = self.steps >= MAX_STEPS
        return self._obs(), float(reward), terminated, truncated, {}


if __name__ == "__main__":
    m = mujoco.MjModel.from_xml_path(XML_PATH)
    total = 0.0
    for i in range(m.nbody):
        total += m.body(i).mass[0]
        print(f"{i}: {m.body(i).name:12s} {m.body(i).mass[0]:.4f} kg")
    print(f"total: {total*1000:.1f} g")
    for i in range(m.nu):
        print(f"{m.actuator(i).name}: limited={m.actuator_forcelimited[i]} range={m.actuator_forcerange[i]}")
    print("free joint:", any(m.jnt_type[i] == 0 for i in range(m.njnt)))
