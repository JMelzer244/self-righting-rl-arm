from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from domain_rand import SelfRightEnvDR

env = make_vec_env(SelfRightEnvDR, n_envs=8)
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
model.learn(total_timesteps=2_000_000)
model.save("selfright_dr")
print("saved selfright_dr.zip")
