import numpy as np
from collections import deque
from selfright_env import SelfRightEnv

MASS_RANGE      = (0.8, 1.2)
FRICTION_RANGE  = (0.6, 1.4)
SERVO_SPEED_RNG = (0.7, 1.3)
FORCE_NM        = (0.7, 1.1)
LATENCY_STEPS   = (0, 2)
ACCEL_NOISE     = 0.05
GYRO_NOISE      = 0.10


class SelfRightEnvDR(SelfRightEnv):
    def __init__(self):
        super().__init__()
        self.nom_mass     = self.model.body_mass.copy()
        self.nom_inertia  = self.model.body_inertia.copy()
        self.nom_friction = self.model.geom_friction.copy()
        self.nom_servo    = self.max_dservo
        self.model.actuator_forcelimited[:] = 1
        self._delay = 0
        self._act_buf = deque(maxlen=1)

    def reset(self, *, seed=None, options=None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        r = self.rng

        ms = r.uniform(*MASS_RANGE, size=self.nom_mass.shape)
        self.model.body_mass[:]    = self.nom_mass * ms
        self.model.body_inertia[:] = self.nom_inertia * ms[:, None]

        self.model.geom_friction[:, 0] = self.nom_friction[:, 0] * r.uniform(*FRICTION_RANGE)
        self.max_dservo = self.nom_servo * r.uniform(*SERVO_SPEED_RNG)

        F = r.uniform(*FORCE_NM)
        self.model.actuator_forcerange[:, 0] = -F
        self.model.actuator_forcerange[:, 1] =  F

        self._delay = int(r.integers(LATENCY_STEPS[0], LATENCY_STEPS[1] + 1))
        self._act_buf = deque([np.zeros(2)] * self._delay, maxlen=self._delay + 1)

        return super().reset(seed=seed, options=options)

    def _obs(self):
        o = super()._obs()
        o[0:3] += self.rng.normal(0, ACCEL_NOISE, 3)   # accel / gravity dir
        o[3:6] += self.rng.normal(0, GYRO_NOISE, 3)    # gyro
        return o.astype(np.float32)

    def step(self, action):
        action = np.asarray(action, dtype=np.float32)
        if self._delay == 0:
            return super().step(action)
        self._act_buf.append(action)
        return super().step(self._act_buf[0])
