import imageio
import mujoco
from stable_baselines3 import PPO
from selfright_env import SelfRightEnv

env = SelfRightEnv()
model = PPO.load("selfright_dr")

renderer = mujoco.Renderer(env.model, height=480, width=640)
frames = []

obs, _ = env.reset()
for _ in range(900):
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, _ = env.step(action)
    renderer.update_scene(env.data, camera=-1)
    frames.append(renderer.render())
    if terminated or truncated:
        obs, _ = env.reset()

imageio.mimsave("rollout.mp4", frames, fps=30)
print("saved rollout.mp4")
