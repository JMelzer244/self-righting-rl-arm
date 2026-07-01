import re
import numpy as np
from stable_baselines3 import PPO

MODEL = "selfright_dr"

model = PPO.load(MODEL, device="cpu")
sd = model.policy.state_dict()

hidden = {}
for k in sd:
    m = re.match(r"mlp_extractor\.policy_net\.(\d+)\.weight", k)
    if m:
        i = m.group(1)
        hidden[int(i)] = (sd[f"mlp_extractor.policy_net.{i}.weight"].numpy(),
                          sd[f"mlp_extractor.policy_net.{i}.bias"].numpy())
layers = [hidden[i] for i in sorted(hidden)]
layers.append((sd["action_net.weight"].numpy(), sd["action_net.bias"].numpy()))

dims = [W.shape for W, _ in layers]
print("layer weight shapes:", dims)
assert dims == [(64, 8), (64, 64), (2, 64)], f"unexpected arch {dims}"


def forward(o):
    x = np.asarray(o, dtype=np.float64)
    for i, (W, b) in enumerate(layers):
        x = W @ x + b
        if i < len(layers) - 1:
            x = np.tanh(x)
    return np.clip(x, -1.0, 1.0)


rng = np.random.default_rng(0)
worst = 0.0
for _ in range(1000):
    o = rng.normal(size=8).astype(np.float32)
    a_sb, _ = model.predict(o, deterministic=True)
    worst = max(worst, float(np.max(np.abs(a_sb - forward(o)))))
print(f"max action mismatch vs SB3: {worst:.2e}")
assert worst < 1e-4, "extracted weights don't match the policy"
print("weights verified")


def carr(name, A):
    A = np.asarray(A, dtype=np.float32)
    if A.ndim == 1:
        body = ", ".join(f"{x:.8e}f" for x in A)
        return f"const float {name}[{A.shape[0]}] = {{{body}}};\n"
    rows = []
    for r in A:
        rows.append("  {" + ", ".join(f"{x:.8e}f" for x in r) + "}")
    return f"const float {name}[{A.shape[0]}][{A.shape[1]}] = {{\n" + ",\n".join(rows) + "\n};\n"


(W0, b0), (W1, b1), (W2, b2) = layers
with open("policy_weights.h", "w") as f:
    f.write("#pragma once\n\n")
    f.write("#define N_OBS 8\n#define N_H1 64\n#define N_H2 64\n#define N_ACT 2\n\n")
    f.write(carr("L0_W", W0)); f.write(carr("L0_B", b0)); f.write("\n")
    f.write(carr("L1_W", W1)); f.write(carr("L1_B", b1)); f.write("\n")
    f.write(carr("L2_W", W2)); f.write(carr("L2_B", b2))
print("wrote policy_weights.h")
