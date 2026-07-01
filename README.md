# self-righting-rl-arm

A 2-DOF self-righting robotic arm trained with PPO in MuJoCo, deployed to physical hardware through a full sim-to-real pipeline running on an ESP32-S3.

The policy learns entirely in simulation, then runs on-device as a hand-coded MLP — no runtime dependencies, no host connection. Domain randomization over mass, friction, latency, and sensor noise bridges the sim-to-real gap.

## Demo

**Simulation (MuJoCo)** — 2-DOF arm self-righting from a fallen pose:

https://github.com/user-attachments/assets/8202d09b-0d75-4aae-b954-ff0e8793351b

*(physical hardware clip coming soon)*

## How it works

The pipeline goes from CAD straight through to firmware running on the microcontroller:

1. **CAD → model** — the arm is designed in Onshape and exported to MuJoCo (MJCF) via `onshape-to-robot`, with per-part mass overrides so the sim mass matches the real printed-and-populated hardware.
2. **Environment** — a custom Gymnasium environment (`selfright_env.py`) defines the self-righting task: the arm starts in a random fallen pose and is rewarded for returning upright.
3. **Training** — PPO (Stable-Baselines3) trains the policy. `train.py` runs the base task; `train_dr.py` adds domain randomization for robustness.
4. **Domain randomization** — `domain_rand.py` randomizes mass, friction, actuator latency, and sensor noise each episode so the policy doesn't overfit to the idealized sim.
5. **Export** — `export_policy.py` converts the trained network into `policy_weights.h`, a plain C header holding the MLP weights.
6. **Deployment** — the ESP32-S3 runs the policy directly from `policy_weights.h` as a hand-coded forward pass, reading the IMU and driving the servos in a closed loop.

## Hardware

- **MCU:** ESP32-S3
- **Actuators:** MG996R servos
- **IMU:** MPU6050 (I2C)
- **Power:** 2S LiPo → UBEC → 6V servo rail

## Repo contents

| File | Purpose |
|---|---|
| `selfright_env.py` | Gymnasium environment for the self-righting task |
| `train.py` / `train_dr.py` | PPO training (base / with domain randomization) |
| `domain_rand.py` | Domain randomization wrapper |
| `export_policy.py` | Exports trained policy to `policy_weights.h` |
| `policy_weights.h` | Trained MLP weights for on-device inference |
| `robot.xml` / `scene.xml` | MuJoCo model and scene |
| `assets/` | STL meshes referenced by the model |
| `config.json` | `onshape-to-robot` export config |
| `view.py` | Load and inspect the model interactively |
| `watch.py` | Run a policy rollout and record video |

## Running it

```bash
# set up environment
conda create -n selfright python=3.11
conda activate selfright
pip install mujoco stable-baselines3 gymnasium onshape-to-robot

# train
python train_dr.py

# watch a rollout
python watch.py

# export for deployment
python export_policy.py
```

To re-export the model from Onshape, add your API keys to a `.env` file (`ONSHAPE_ACCESS_KEY` / `ONSHAPE_SECRET_KEY`) and run `onshape-to-robot .`

## License

MIT
