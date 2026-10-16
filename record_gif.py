import os
import torch, torch.nn as nn
from PIL import Image
import gymnasium as gym
import minigrid

ENV_ID = "MiniGrid-DoorKey-6x6-v0"
SEEDS = [1000, 1001, 1002]     # same layouts used in the evaluation
OUT = "assets/demo.gif"
SIZE = 320
FRAME_MS = 200


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


model = Policy()
model.load_state_dict(torch.load("policy.pt"))
model.eval()

env = gym.make(ENV_ID, render_mode="rgb_array")


def act(obs):
    x = torch.tensor(obs["image"], dtype=torch.float32).permute(2, 0, 1).unsqueeze(0) / 10.0
    d = torch.tensor([int(obs["direction"])])
    with torch.no_grad():
        return model(x, d).argmax(1).item()


frames, durations = [], []
for seed in SEEDS:
    obs, _ = env.reset(seed=seed)
    done, r, steps = False, 0, 0
    frames.append(Image.fromarray(env.render())); durations.append(FRAME_MS)
    while not done:
        obs, r, term, trunc, _ = env.step(act(obs))
        done = term or trunc
        steps += 1
        frames.append(Image.fromarray(env.render())); durations.append(FRAME_MS)
    durations[-1] = 1000          # hold the last frame of each episode
    print(f"seed {seed}: {'SUCCESS' if r > 0 else 'FAILED'} in {steps} steps")

frames = [f.resize((SIZE, SIZE), Image.NEAREST) for f in frames]
os.makedirs("assets", exist_ok=True)
frames[0].save(OUT, save_all=True, append_images=frames[1:],
               duration=durations, loop=0)
print("saved", OUT, f"({os.path.getsize(OUT) / 1024:.0f} KB)")