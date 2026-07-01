from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from selfright_env import SelfRightEnv

env = make_vec_env(SelfRightEnv, n_envs=4)

model = PPO(
    "MlpPolicy", env,
    verbose=1,
    n_steps=2048,
    batch_size=256,
    gamma=0.99,
    gae_lambda=0.95,
    ent_coef=0.0,
    learning_rate=3e-4,
)

model.learn(total_timesteps=500_000)
model.save("selfright")
print("saved selfright.zip")
