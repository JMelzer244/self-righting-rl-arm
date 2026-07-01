# self-righting-rl-arm

A 2-DOF arm that learns to right itself after being knocked over. The policy is trained with PPO in MuJoCo and deployed to an ESP32-S3 that drives the servos directly.

To close the sim-to-real gap, training uses domain randomization over mass, friction, actuator latency, and sensor noise so the policy generalizes to the real hardware instead of overfitting to the simulator.

## Demo

Sim rollout — the arm recovering from a fallen start:

https://github.com/user-attachments/assets/8202d09b-0d75-4aae-b954-ff0e8793351b

## How it works

- The arm is modeled in Onshape and exported to a MuJoCo model via `onshape-to-robot`, with per-part mass overrides so the simulated mass matches the printed hardware.
- `selfright_env.py` defines the Gymnasium environment: the arm starts in a random fallen pose and is rewarded for returning upright.
- `train.py` and `train_dr.py` run PPO (Stable-Baselines3); the `_dr` variant adds domain randomization.
- `domain_rand.py` randomizes the physics parameters each episode.
- `export_policy.py` converts the trained network to `policy_weights.h`, a C header holding the MLP weights.
- The ESP32-S3 runs the policy from that header, reading the IMU and driving the servos in a closed loop with no host connection.

## Hardware

- ESP32-S3
- 2 MG996R servos
- MPU6050 IMU (I2C)
- 2S LiPo → UBEC → 6V servo rail

## Running it

```bash
conda activate selfright
python train_dr.py        # train the policy
python watch.py           # record a rollout
python export_policy.py   # generate policy_weights.h
```

To re-export the model from Onshape, add your API keys to a `.env` file (`ONSHAPE_ACCESS_KEY` / `ONSHAPE_SECRET_KEY`) and run `onshape-to-robot .`
