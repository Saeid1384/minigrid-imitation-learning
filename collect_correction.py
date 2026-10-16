import os
import numpy as np
import torch, torch.nn as nn
import tkinter as tk
from PIL import Image, ImageTk
import gymnasium as gym
import minigrid

ENV_ID = "MiniGrid-DoorKey-6x6-v0"
POLICY = "policy.pt"
OUT = "corrections.npz"
SIZE = 480
DELAY = 150  # ms between model steps
FACING = {0: "RIGHT", 1: "DOWN", 2: "LEFT", 3: "UP"}


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
model.load_state_dict(torch.load(POLICY))
model.eval()

env = gym.make(ENV_ID, render_mode="rgb_array")


def model_action(obs):
    x = torch.tensor(obs["image"], dtype=torch.float32).permute(2, 0, 1).unsqueeze(0) / 10.0
    d = torch.tensor([int(obs["direction"])])
    with torch.no_grad():
        return model(x, d).argmax(1).item()


class App:
    def __init__(self, root):
        self.root = root
        root.title("DoorKey Corrections")

        self.all_o, self.all_d, self.all_a = [], [], []
        if os.path.exists(OUT):
            d = np.load(OUT)
            self.all_o = list(d["obs"])
            self.all_d = list(d["direction"])
            self.all_a = list(d["action"])
        self.saved_eps = 0

        self.canvas = tk.Canvas(root, width=SIZE, height=SIZE, bg="black")
        self.canvas.grid(row=0, column=0, padx=10, pady=10)

        panel = tk.Frame(root)
        panel.grid(row=0, column=1, padx=10, pady=10, sticky="n")

        tk.Label(panel, justify="left", font=("Arial", 10),
                 text="The model plays by itself. When it gets stuck\n"
                      "or makes a mistake, press Take Over and fix it.\n"
                      "Only YOUR actions are recorded.").grid(
            row=0, column=0, columnspan=3, pady=(0, 8))

        self.mode_lbl = tk.Label(panel, text="", font=("Arial", 13, "bold"))
        self.mode_lbl.grid(row=1, column=0, columnspan=3, pady=(0, 4))

        self.facing = tk.Label(panel, text="", font=("Arial", 12, "bold"), fg="#b00020")
        self.facing.grid(row=2, column=0, columnspan=3, pady=(0, 6))

        self.status = tk.Label(panel, text="", justify="left", font=("Arial", 11))
        self.status.grid(row=3, column=0, columnspan=3, pady=(0, 8))

        self.toggle_btn = tk.Button(panel, width=22, height=2, command=self.toggle)
        self.toggle_btn.grid(row=4, column=0, columnspan=2, padx=3, pady=3)
        self.takeover_btn = tk.Button(panel, text="Take Over", width=15, height=2,
                                      bg="#ffcdd2", command=self.take_over)
        self.takeover_btn.grid(row=4, column=2, padx=3, pady=3)

        self.act_buttons = []

        def mk(text, a, r, c, bg=None):
            b = tk.Button(panel, text=text, width=15, height=2,
                          command=lambda: self.act(a))
            if bg:
                b.config(bg=bg)
            b.grid(row=r, column=c, padx=3, pady=3)
            self.act_buttons.append(b)

        mk("↺  Rotate Left", 0, 5, 0, "#bbdefb")
        mk("↑  Forward", 2, 5, 1)
        mk("Rotate Right  ↻", 1, 5, 2, "#ffe0b2")
        mk("Pick Up Key", 3, 6, 0)
        mk("Open Door", 5, 6, 1)

        self.save_btn = tk.Button(panel, text="Save Corrections", width=46,
                                  bg="#c8f7c5", command=self.save_episode)
        self.save_btn.grid(row=7, column=0, columnspan=3, pady=(15, 3))

        tk.Button(panel, text="Next Episode (discard current)", width=46,
                  command=self.new_episode).grid(row=8, column=0, columnspan=3, pady=3)

        tk.Button(panel, text="Finish and Close", width=46,
                  command=root.destroy).grid(row=9, column=0, columnspan=3, pady=(15, 3))

        self.new_episode()
        root.after(DELAY, self.tick)

    def new_episode(self):
        self.obs, _ = env.reset()
        self.ep_o, self.ep_d, self.ep_a = [], [], []
        self.model_steps = 0
        self.done = False
        self.reward = 0
        self.mode = "model"          # "model" | "paused" | "human"
        self.refresh()

    def tick(self):
        if self.mode == "model" and not self.done:
            a = model_action(self.obs)
            self.obs, r, term, trunc, _ = env.step(a)
            self.done = term or trunc
            self.reward = r
            self.model_steps += 1
            self.refresh()
        self.root.after(DELAY, self.tick)

    def toggle(self):
        if self.done:
            return
        self.mode = "paused" if self.mode == "model" else "model"
        self.refresh()

    def take_over(self):
        if self.done:
            return
        self.mode = "human"
        self.refresh()

    def act(self, a):
        if self.done or self.mode != "human":
            return
        self.ep_o.append(self.obs["image"].copy())
        self.ep_d.append(int(self.obs["direction"]))
        self.ep_a.append(a)
        self.obs, r, term, trunc, _ = env.step(a)
        self.done = term or trunc
        self.reward = r
        self.refresh()

    def save_episode(self):
        if not self.done or not self.ep_a:
            return
        self.all_o += self.ep_o
        self.all_d += self.ep_d
        self.all_a += self.ep_a
        self.saved_eps += 1
        np.savez(OUT,
                 obs=np.array(self.all_o, dtype=np.uint8),
                 direction=np.array(self.all_d, dtype=np.int64),
                 action=np.array(self.all_a, dtype=np.int64))
        self.new_episode()

    def refresh(self):
        img = Image.fromarray(env.render()).resize((SIZE, SIZE), Image.NEAREST)
        self.photo = ImageTk.PhotoImage(img)
        self.canvas.create_image(0, 0, anchor="nw", image=self.photo)

        self.facing.config(text=f"Facing: {FACING[int(self.obs['direction'])]}")

        human = self.mode == "human" and not self.done
        for b in self.act_buttons:
            b.config(state="normal" if human else "disabled")
        self.toggle_btn.config(
            text="Pause Model" if self.mode == "model" else "Run Model",
            state="disabled" if (self.done or self.mode == "human") else "normal")
        self.takeover_btn.config(
            state="disabled" if (self.done or self.mode == "human") else "normal")
        self.save_btn.config(state="normal" if (self.done and self.ep_a) else "disabled")

        if self.done:
            mode_txt = "Finished: SUCCESS" if self.reward > 0 else "Finished: FAILED (time out)"
        else:
            mode_txt = {"model": "Mode: MODEL running",
                        "paused": "Mode: PAUSED",
                        "human": "Mode: HUMAN (recording)"}[self.mode]
        self.mode_lbl.config(text=mode_txt)
        self.status.config(text=(
            f"Model steps: {self.model_steps}\n"
            f"Your recorded steps: {len(self.ep_a)}\n"
            f"Episodes saved this session: {self.saved_eps}\n"
            f"Total correction samples in file: {len(self.all_a)}"
        ))


root = tk.Tk()
App(root)
root.mainloop()