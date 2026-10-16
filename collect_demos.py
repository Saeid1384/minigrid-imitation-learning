import os
import numpy as np
import tkinter as tk
from PIL import Image, ImageTk
import gymnasium as gym
import minigrid

ENV_ID = "MiniGrid-DoorKey-6x6-v0"
OUT = "demos.npz"
SIZE = 480
FACING = {0: "RIGHT", 1: "DOWN", 2: "LEFT", 3: "UP"}

env = gym.make(ENV_ID, render_mode="rgb_array")


class App:
    def __init__(self, root):
        root.title("DoorKey Demo Collector")

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
                 text="Goal: pick up the key, open the door,\nthen reach the green square.\n\n"
                      "Rotate buttons turn the agent in place,\n"
                      "relative to where the red triangle points.").grid(
            row=0, column=0, columnspan=3, pady=(0, 8))

        self.facing = tk.Label(panel, text="", font=("Arial", 14, "bold"), fg="#b00020")
        self.facing.grid(row=1, column=0, columnspan=3, pady=(0, 8))

        self.status = tk.Label(panel, text="", justify="left", font=("Arial", 11))
        self.status.grid(row=2, column=0, columnspan=3, pady=(0, 10))

        self.act_buttons = []

        def mk(text, a, r, c, bg=None):
            b = tk.Button(panel, text=text, width=15, height=2,
                          command=lambda: self.act(a))
            if bg:
                b.config(bg=bg)
            b.grid(row=r, column=c, padx=3, pady=3)
            self.act_buttons.append(b)

        mk("↺  Rotate Left", 0, 3, 0, "#bbdefb")
        mk("↑  Forward", 2, 3, 1)
        mk("Rotate Right  ↻", 1, 3, 2, "#ffe0b2")
        mk("Pick Up Key", 3, 4, 0)
        mk("Open Door", 5, 4, 1)

        tk.Button(panel, text="Reset Episode", width=46,
                  command=self.new_episode).grid(row=5, column=0, columnspan=3, pady=(15, 3))

        self.save_btn = tk.Button(panel, text="Save This Episode", width=46,
                                  bg="#c8f7c5", command=self.save_episode)
        self.save_btn.grid(row=6, column=0, columnspan=3, pady=3)

        self.discard_btn = tk.Button(panel, text="Discard This Episode", width=46,
                                     bg="#f7c5c5", command=self.new_episode)
        self.discard_btn.grid(row=7, column=0, columnspan=3, pady=3)

        tk.Button(panel, text="Finish and Close", width=46,
                  command=root.destroy).grid(row=8, column=0, columnspan=3, pady=(15, 3))

        self.new_episode()

    def new_episode(self):
        self.obs, _ = env.reset()
        self.ep_o, self.ep_d, self.ep_a = [], [], []
        self.done = False
        self.reward = 0
        self.refresh()

    def act(self, a):
        if self.done:
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

        for b in self.act_buttons:
            b.config(state="disabled" if self.done else "normal")
        st = "normal" if self.done else "disabled"
        self.save_btn.config(state=st)
        self.discard_btn.config(state=st)

        if self.done:
            msg = "Finished: SUCCESS" if self.reward > 0 else "Finished: FAILED (time out)"
        else:
            msg = "Playing..."
        self.status.config(text=(
            f"{msg}\n"
            f"Steps this episode: {len(self.ep_a)}\n"
            f"Episodes saved this session: {self.saved_eps}\n"
            f"Total samples in file: {len(self.all_a)}"
        ))


root = tk.Tk()
App(root)
root.mainloop()