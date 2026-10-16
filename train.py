import os
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
import gymnasium as gym
import minigrid

ENV_ID = "MiniGrid-DoorKey-6x6-v0"
EPOCHS = 200
REPEAT = 1        # how many times each correction sample is counted
SEED = 0
EVAL_EPISODES = 200

torch.manual_seed(SEED)
np.random.seed(SEED)


def load(path):
    d = np.load(path)
    return d["obs"], d["direction"], d["action"]


# ---------- data ----------
o, di, a = load("demos.npz")
print("demos:", len(a), "samples")
if os.path.exists("corrections.npz"):
    co, cd, ca = load("corrections.npz")
    print("corrections:", len(ca), "samples (x%d)" % REPEAT)
    o = np.concatenate([o] + [co] * REPEAT)
    di = np.concatenate([di] + [cd] * REPEAT)
    a = np.concatenate([a] + [ca] * REPEAT)
print("total:", len(a), "samples")

X = torch.tensor(o, dtype=torch.float32).permute(0, 3, 1, 2) / 10.0
D = torch.tensor(di, dtype=torch.long)
Y = torch.tensor(a, dtype=torch.long)


# ---------- model ----------
class Policy(nn.Module):
    def __init__(self, n_actions=7):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1), nn.ReLU(),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(),
            nn.Flatten())
        self.dir_emb = nn.Embedding(4, 8)
        self.head = nn.Sequential(
            nn.Linear(32 * 7 * 7 + 8, 128), nn.ReLU(),
            nn.Linear(128, n_actions))

    def forward(self, x, d):
        return self.head(torch.cat([self.conv(x), self.dir_emb(d)], dim=1))


# ---------- train (always from scratch) ----------
dl = DataLoader(TensorDataset(X, D, Y), batch_size=64, shuffle=True)
model = Policy()
opt = torch.optim.Adam(model.parameters(), lr=1e-3)

for epoch in range(1, EPOCHS + 1):
    total = 0.0
    for x, d, y in dl:
        loss = F.cross_entropy(model(x, d), y)
        opt.zero_grad(); loss.backward(); opt.step()
        total += loss.item() * len(y)
    if epoch % 25 == 0:
        print(f"epoch {epoch:3d} | loss {total / len(Y):.3f}")

model.eval()
torch.save(model.state_dict(), "policy.pt")
print("saved policy.pt")


# ---------- evaluate ----------
env = gym.make(ENV_ID)

def act(obs, sample):
    x = torch.tensor(obs["image"], dtype=torch.float32).permute(2, 0, 1).unsqueeze(0) / 10.0
    d = torch.tensor([int(obs["direction"])])
    with torch.no_grad():
        logits = model(x, d)
    if sample:
        return torch.distributions.Categorical(logits=logits).sample().item()
    return logits.argmax(1).item()

def evaluate(n, sample):
    wins = 0
    for i in range(n):
        obs, _ = env.reset(seed=1000 + i)      # same layouts every run
        done, r = False, 0
        while not done:
            obs, r, term, trunc, _ = env.step(act(obs, sample))
            done = term or trunc
        wins += r > 0
    return wins / n

print(f"success rate (argmax):   {evaluate(EVAL_EPISODES, False):.3f}")
print(f"success rate (sampling): {evaluate(EVAL_EPISODES, True):.3f}")