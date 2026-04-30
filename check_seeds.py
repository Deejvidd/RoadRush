"""
Run: python check_seeds.py
     python check_seeds.py --model snapshots/snapshot_ep1500.json
Shows per-seed greedy score so you can identify and swap out bad seeds.
"""

import numpy as np, json, os, argparse
from train3_grid import DQN, HeadlessGame, EVAL_SEEDS, N_IN, N_H

ap = argparse.ArgumentParser()
ap.add_argument("--model", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "model.json"))
args = ap.parse_args()

d = json.load(open(args.model))
stored_nh = d.get("n_h", None)
if stored_nh and stored_nh != N_H:
    print(f"WARNING: model n_h={stored_nh} but current N_H={N_H}")

net = DQN()
net.set_weights(d.get("weights", d))

game = HeadlessGame()
print(f"\n  Evaluating {len(EVAL_SEEDS)} seeds...\n")
scores = []
for seed in EVAL_SEEDS:
    s = game.reset(seed=seed)
    done = False
    while not done:
        a = int(np.argmax(net.q(np.asarray(s, dtype=np.float32))))
        s, _, done, sc = game.step(a)
    scores.append(sc)
    tag = " ← WEAK" if sc < 400 else (" ★ MAX" if sc == 2500 else "")
    print(f"  seed={seed:6d}  score={sc:5d}{tag}")

print(f"\n  mean={np.mean(scores):.0f}  min={min(scores)}  max={max(scores)}")
print(f"\n  Weak seeds (score < 400): {[EVAL_SEEDS[i] for i,s in enumerate(scores) if s < 400]}")
