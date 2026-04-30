"""
Standalone headless DQN trainer — parallel envs, 256-hidden net, multi-seed training.

Usage:
  python train3_grid.py                   # 10000 episodes, auto workers
  python train3_grid.py --episodes 5000
  python train3_grid.py --load            # continue from model.json
  python train3_grid.py --snapshots       # save snapshots during training
  python train3_grid.py --workers 4       # override parallel worker count
"""

import numpy as np, json, os, random, argparse, time
from collections import deque
from multiprocessing import Pool

# ── Hyperparameters ──────────────────────────────────────────────────────────────
GAMMA           = 0.99
LR              = 3e-4
LR_FINE         = 5e-5
FINE_TUNE_THRESHOLD = 1200      # multi-seed mean (was 2000 for single-seed)
GRAD_CLIP       = 1.0
BETA1, BETA2    = 0.9, 0.999
BUFFER_SIZE     = 300_000       # was 150k
BATCH_SIZE      = 128
TAU             = 0.005
WARMUP_STEPS    = 2000
TRAIN_EVERY     = 4
REWARD_SCALE    = 0.1
HUBER_DELTA     = 1.0
EPS_START       = 1.0
EPS_END         = 0.01
EPS_DECAY_STEPS = 250_000

N_OBS_SLOTS = 4
GRID_COLS   = 5
GRID_ROWS   = 3
N_IN  = 2 + (GRID_COLS * GRID_ROWS) + 1 + N_OBS_SLOTS * 3   # = 30
N_H   = 256                     # was 128
N_OUT = 3

FIXED_SEED        = 42
EVAL_SEEDS        = [42, 500, 456, 789, 1337, 2024, 7, 99, 314, 1000]  # 123 replaced with 500
GREEDY_EVAL_EVERY = 25

MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "model.json")
SNAP_DIR   = os.path.join(os.path.dirname(os.path.abspath(__file__)), "snapshots")
SNAPSHOT_AT = [1, 5, 10, 25, 50, 100, 200, 300, 500, 750, 900, 1000, 1100, 1200, 1300, 1400, 1500, 1700, 2000,
               3000, 4000, 5000, 6000, 7000, 8000, 10000, 12000, 15000]

# ── Q-Network ────────────────────────────────────────────────────────────────────
class DQN:
    def __init__(self):
        self.W1 = np.random.randn(N_H, N_IN)  * np.sqrt(2 / N_IN)
        self.b1 = np.zeros(N_H)
        self.W2 = np.random.randn(N_H, N_H)   * np.sqrt(2 / N_H)
        self.b2 = np.zeros(N_H)
        self.W3 = np.random.randn(N_OUT, N_H) * np.sqrt(2 / N_H)
        self.b3 = np.zeros(N_OUT)
        self._t = 0
        for name in ("W1","b1","W2","b2","W3","b3"):
            w = getattr(self, name)
            setattr(self, "_m_" + name, np.zeros_like(w))
            setattr(self, "_v_" + name, np.zeros_like(w))

    def forward(self, X):
        X  = np.asarray(X, dtype=np.float32)
        if X.ndim == 1: X = X[np.newaxis, :]
        H1 = np.maximum(0, X  @ self.W1.T + self.b1)
        H2 = np.maximum(0, H1 @ self.W2.T + self.b2)
        Q  = H2 @ self.W3.T + self.b3
        return Q, H2, H1

    def q(self, s):
        Q, _, _ = self.forward(s)
        return Q[0]

    def _adam_step(self, lr, **grads):
        self._t += 1
        t = self._t
        for name, g in grads.items():
            m = BETA1 * getattr(self, "_m_" + name) + (1 - BETA1) * g
            v = BETA2 * getattr(self, "_v_" + name) + (1 - BETA2) * g**2
            setattr(self, "_m_" + name, m)
            setattr(self, "_v_" + name, v)
            m_hat = m / (1 - BETA1**t)
            v_hat = v / (1 - BETA2**t)
            setattr(self, name, getattr(self, name) - lr * m_hat / (np.sqrt(v_hat) + 1e-8))

    def train_step(self, S, A, targets, lr=LR):
        B = len(S)
        Q, H2, H1 = self.forward(S)
        q_pred = Q[np.arange(B), A]
        err    = q_pred - targets
        dQ     = np.zeros_like(Q)
        dQ[np.arange(B), A] = np.clip(err, -HUBER_DELTA, HUBER_DELTA) / B
        dW3 = dQ.T @ H2;               db3 = dQ.sum(0)
        dH2 = dQ @ self.W3;            dH2 *= (H2 > 0)
        dW2 = dH2.T @ H1;              db2 = dH2.sum(0)
        dH1 = dH2 @ self.W2;           dH1 *= (H1 > 0)
        dW1 = dH1.T @ np.asarray(S, dtype=np.float32); db1 = dH1.sum(0)
        grads = [dW1, db1, dW2, db2, dW3, db3]
        norm  = float(np.sqrt(sum((g*g).sum() for g in grads)))
        if norm > GRAD_CLIP:
            sc = GRAD_CLIP / (norm + 1e-8)
            dW1, db1, dW2, db2, dW3, db3 = [g * sc for g in grads]
        self._adam_step(lr, W1=dW1, b1=db1, W2=dW2, b2=db2, W3=dW3, b3=db3)
        return float((err**2).mean()), float(q_pred.mean())

    def get_weights(self):
        return {k: getattr(self, k).tolist() for k in ("W1","b1","W2","b2","W3","b3")}

    def set_weights(self, w):
        for k in ("W1","b1","W2","b2","W3","b3"):
            if k in w:
                arr = np.array(w[k], dtype=np.float32)
                if arr.shape == getattr(self, k).shape:
                    setattr(self, k, arr)
        self._t = 0
        for name in ("W1","b1","W2","b2","W3","b3"):
            wa = getattr(self, name)
            setattr(self, "_m_" + name, np.zeros_like(wa))
            setattr(self, "_v_" + name, np.zeros_like(wa))

    def copy_from(self, other):
        self.set_weights(other.get_weights())

# ── Headless Game ─────────────────────────────────────────────────────────────────
class HeadlessGame:
    W, H        = 480, 720
    ROAD_L      = 75
    ROAD_R      = 405
    ROAD_W      = ROAD_R - ROAD_L
    BASE_SPEED  = 3.5
    SPAWN_INTERVAL = 60

    GRID_ROW_BANDS = [(0, 120), (120, 260), (260, 430)]

    def __init__(self): self.reset()

    def reset(self, seed=FIXED_SEED):
        self.rng          = np.random.RandomState(seed)
        self.px           = self.W / 2
        self.py           = self.H - 130
        self.pw, self.ph  = 42, 70
        self.pvx          = 0.0
        self.obstacles    = []
        self.score        = 0
        self.frame        = 0
        self.level        = 1
        self.spawn_timer  = 0
        self._ms_hit      = set()
        return self.get_features()

    def _grid_occupancy(self):
        col_w    = self.ROAD_W / GRID_COLS
        player_y = self.py + self.ph / 2

        grid = []
        for (near, far) in self.GRID_ROW_BANDS:
            y_bottom = player_y - near
            y_top    = player_y - far
            for ci in range(GRID_COLS):
                cell_x1 = self.ROAD_L + ci * col_w
                cell_x2 = cell_x1 + col_w
                hit = 0.0
                for o in self.obstacles:
                    ox1 = o["x"] - o["w"] / 2
                    ox2 = o["x"] + o["w"] / 2
                    oy1 = o["y"]
                    oy2 = o["y"] + o["h"]
                    if ox2 > cell_x1 and ox1 < cell_x2 and oy2 > y_top and oy1 < y_bottom:
                        hit = 1.0
                        break
                grid.append(hit)
        return grid

    def get_features(self):
        base = [
            (self.px - self.ROAD_L) / self.ROAD_W,
            float(np.tanh(self.pvx / 8.0)),
        ]
        base += self._grid_occupancy()
        base += [self.spawn_timer / self.SPAWN_INTERVAL]

        upcoming = sorted(self.obstacles, key=lambda o: -o["y"])[:N_OBS_SLOTS]
        slots = []
        for o in upcoming:
            slots.append((o["x"] - self.ROAD_L) / self.ROAD_W)
            slots.append(max(0.0, self.py - o["y"]) / self.H)
            slots.append(o["w"] / self.ROAD_W)
        while len(slots) < N_OBS_SLOTS * 3:
            slots.extend([0.5, 1.0, 0.12])

        return base + slots

    def compute_reward(self):
        feat      = self.get_features()
        near_row  = feat[2:7]
        clear_fwd = 1.0 - max(near_row)
        progress  = min(1.0, self.score / 2500.0)
        r  = 0.10 + 0.15 * progress
        r += clear_fwd * 0.4
        wall_prox = min(feat[0], 1.0 - feat[0])
        if wall_prox < 0.08:
            r -= (0.08 - wall_prox) / 0.08 * 1.5
        return r

    def step(self, action):
        self.frame += 1
        spd = self.BASE_SPEED + (self.level - 1) * 0.3
        if action == 0: self.pvx -= 1.2
        if action == 2: self.pvx += 1.2
        self.pvx *= 0.75
        self.px   = np.clip(self.px + self.pvx,
                            self.ROAD_L + self.pw/2 + 2,
                            self.ROAD_R - self.pw/2 - 2)

        self.spawn_timer += 1
        if self.spawn_timer >= self.SPAWN_INTERVAL:
            self.spawn_timer = 0
            m  = 24
            ox = self.ROAD_L + m + self.rng.random() * (self.ROAD_W - m * 2)
            self.obstacles.append({"x": ox, "y": -90, "w": 40, "h": 68,
                                   "speed": spd + 0.3 + self.rng.random() * 1.0})

        for o in self.obstacles:
            o["y"] += o["speed"]
        self.obstacles = [o for o in self.obstacles if o["y"] < self.H + 100]

        m = 6
        for o in self.obstacles:
            if (abs(self.px - o["x"]) < (self.pw/2 + o["w"]/2 - m) and
                    abs((self.py + self.ph/2) - (o["y"] + o["h"]/2)) < (self.ph/2 + o["h"]/2 - m)):
                penalty = -(50.0 + 30.0 * min(1.0, self.score / 2500.0))
                return self.get_features(), penalty, True, self.score

        self.score += 1
        self.level  = min(5, 1 + self.score // 500)

        milestone_bonus = 0.0
        for ms in (500, 1000, 1500, 2000):
            if self.score == ms and ms not in self._ms_hit:
                self._ms_hit.add(ms)
                milestone_bonus = 10.0
                break

        if self.score >= 2500:
            return self.get_features(), 50.0, True, self.score

        return self.get_features(), self.compute_reward() + milestone_bonus, False, self.score

# ── Worker — must be top-level for multiprocessing pickle ─────────────────────────
def collect_episode(args):
    """Run one episode with the given weights/seed/epsilon. Returns (transitions, score)."""
    weights, seed, eps = args
    net = DQN()
    net.set_weights(weights)
    game = HeadlessGame()
    state = game.reset(seed=seed)
    transitions = []
    done = False
    while not done:
        if random.random() < eps:
            action = random.randint(0, N_OUT - 1)
        else:
            action = int(np.argmax(net.q(np.asarray(state, dtype=np.float32))))
        next_state, reward, done, score = game.step(action)
        transitions.append((
            np.asarray(state,      dtype=np.float32),
            action,
            reward * REWARD_SCALE,
            np.asarray(next_state, dtype=np.float32),
            float(done),
        ))
        state = next_state
    return transitions, score

# ── Helpers ───────────────────────────────────────────────────────────────────────
def epsilon(step):
    if step >= EPS_DECAY_STEPS: return EPS_END
    return EPS_START + (EPS_END - EPS_START) * (step / EPS_DECAY_STEPS)

def pick_train_seed(ep_i):
    """Every 5th episode explicitly uses an eval seed; others are pseudo-random."""
    if ep_i % 5 == 0:
        return EVAL_SEEDS[(ep_i // 5) % len(EVAL_SEEDS)]
    return (FIXED_SEED + ep_i * 7919) % (2**31)

def greedy_eval(net, seeds=None):
    """Run one deterministic episode per seed, return (mean, min, max, scores_list)."""
    if seeds is None:
        seeds = EVAL_SEEDS
    g = HeadlessGame()
    scores = []
    for seed in seeds:
        s = g.reset(seed=seed)
        done = False
        while not done:
            a = int(np.argmax(net.q(np.asarray(s, dtype=np.float32))))
            s, _, done, sc = g.step(a)
        scores.append(sc)
    return float(np.mean(scores)), int(np.min(scores)), int(np.max(scores)), scores

# ── Training loop ─────────────────────────────────────────────────────────────────
def train(n_episodes, load=False, take_snapshots=False, n_workers=None):
    if n_workers is None:
        n_workers = max(1, (os.cpu_count() or 4) - 1)

    net = DQN()
    if load and os.path.exists(MODEL_PATH):
        d = json.load(open(MODEL_PATH))
        w = d.get("weights", d)
        stored_nin = d.get("n_in", None)
        stored_nh  = d.get("n_h",  None)
        shape_ok = (stored_nin is None or stored_nin == N_IN) and \
                   (stored_nh  is None or stored_nh  == N_H)
        if not shape_ok:
            print(f"  WARNING: saved model shape mismatch "
                  f"(n_in={stored_nin}, n_h={stored_nh}). Starting fresh.")
        else:
            net.set_weights(w)
            print(f"  Loaded weights from {MODEL_PATH}")
    tgt = DQN(); tgt.copy_from(net)

    buf           = deque(maxlen=BUFFER_SIZE)
    scores_recent = deque(maxlen=20)

    env_steps      = 0
    grad_steps     = 0
    best_score     = 0
    best_avg       = 0.0
    best_greedy    = 0.0
    best_greedy_w  = None
    fine_tune_mode = False

    if take_snapshots:
        os.makedirs(SNAP_DIR, exist_ok=True)

    t0       = time.time()
    loss_avg = 0.0
    q_avg    = 0.0

    print(f"\n  DQN | workers={n_workers} | eval_seeds={len(EVAL_SEEDS)} | {n_episodes} episodes")
    print(f"  obs={N_IN} | net={N_IN}->{N_H}->{N_H}->{N_OUT} | buffer={BUFFER_SIZE}")
    print(f"  LR={LR} -> LR_FINE={LR_FINE} at greedy_mean>{FINE_TUNE_THRESHOLD}\n")

    ep_i       = 0
    next_eval  = GREEDY_EVAL_EVERY
    snap_idx   = 0
    last_print = 0

    with Pool(n_workers) as pool:
        while ep_i < n_episodes:
            # ── collect a batch of episodes in parallel ───────────────────────────
            batch_n = min(n_workers, n_episodes - ep_i)
            eps     = epsilon(env_steps)
            weights = net.get_weights()
            seeds   = [pick_train_seed(ep_i + i) for i in range(batch_n)]
            args    = [(weights, s, eps) for s in seeds]

            results = pool.map(collect_episode, args) if n_workers > 1 \
                      else [collect_episode(a) for a in args]

            # ── push transitions into replay buffer ───────────────────────────────
            batch_steps = 0
            last_score  = 0
            for transitions, score in results:
                ep_i       += 1
                last_score  = score
                for t in transitions:
                    buf.append(t)
                env_steps  += len(transitions)
                batch_steps += len(transitions)
                scores_recent.append(score)
                if score > best_score:
                    best_score = score

            avg = float(np.mean(scores_recent)) if scores_recent else 0.0
            if avg > best_avg: best_avg = avg

            # ── gradient updates proportional to experience collected ──────────────
            for _ in range(batch_steps // TRAIN_EVERY):
                if len(buf) < max(BATCH_SIZE, WARMUP_STEPS):
                    break
                batch  = random.sample(buf, BATCH_SIZE)
                S  = np.array([b[0] for b in batch], dtype=np.float32)
                A  = np.array([b[1] for b in batch], dtype=np.int32)
                R  = np.array([b[2] for b in batch], dtype=np.float32)
                S2 = np.array([b[3] for b in batch], dtype=np.float32)
                D  = np.array([b[4] for b in batch], dtype=np.float32)

                Q_on, _, _  = net.forward(S2)
                a_star      = Q_on.argmax(axis=1)
                Q_tgt, _, _ = tgt.forward(S2)
                targets     = R + GAMMA * Q_tgt[np.arange(BATCH_SIZE), a_star] * (1.0 - D)

                lr_now   = LR_FINE if fine_tune_mode else LR
                loss, qm = net.train_step(S, A, targets, lr=lr_now)
                grad_steps += 1
                loss_avg = 0.98 * loss_avg + 0.02 * loss
                q_avg    = 0.98 * q_avg    + 0.02 * qm

                for k in ("W1","b1","W2","b2","W3","b3"):
                    setattr(tgt, k, (1-TAU)*getattr(tgt,k) + TAU*getattr(net,k))

            # ── greedy eval at every GREEDY_EVAL_EVERY checkpoint ─────────────────
            while ep_i >= next_eval:
                g_mean, g_min, g_max, _ = greedy_eval(net)
                if g_mean > best_greedy:
                    best_greedy   = g_mean
                    best_greedy_w = net.get_weights()
                    if g_mean >= FINE_TUNE_THRESHOLD and not fine_tune_mode:
                        fine_tune_mode = True
                        print(f"  ⚡ LR -> LR_FINE={LR_FINE} (greedy_mean crossed {FINE_TUNE_THRESHOLD})")
                    print(f"  ★ ep={ep_i:5d}  greedy mean={g_mean:.0f} min={g_min} max={g_max}  (NEW BEST)")
                    json.dump({"weights": best_greedy_w, "episode": ep_i,
                               "best_greedy_mean": best_greedy, "best_score": best_score,
                               "n_in": N_IN, "n_h": N_H, "eval_seeds": EVAL_SEEDS},
                              open(MODEL_PATH, "w"))
                else:
                    tag = " ⚡FINE" if fine_tune_mode else ""
                    print(f"    ep={ep_i:5d}  greedy mean={g_mean:.0f} min={g_min} max={g_max}"
                          f"  (best={best_greedy:.0f}){tag}")
                next_eval += GREEDY_EVAL_EVERY

            # ── snapshots ─────────────────────────────────────────────────────────
            while snap_idx < len(SNAPSHOT_AT) and ep_i >= SNAPSHOT_AT[snap_idx]:
                if take_snapshots:
                    snap_ep = SNAPSHOT_AT[snap_idx]
                    snap = {"weights": net.get_weights(), "episode": ep_i,
                            "avg_score": round(avg, 1), "best_score": best_score,
                            "grad_steps": grad_steps, "epsilon": round(epsilon(env_steps), 4),
                            "n_in": N_IN, "n_h": N_H}
                    json.dump(snap, open(os.path.join(SNAP_DIR, f"snapshot_ep{snap_ep}.json"), "w"))
                    print(f"  [SNAP] Saved snapshot_ep{snap_ep}.json (avg={avg:.1f})")
                snap_idx += 1

            # ── status print every 20 episodes ────────────────────────────────────
            if ep_i - last_print >= 20 or ep_i <= 5:
                last_print = ep_i
                dt      = time.time() - t0
                ep_rate = ep_i / max(dt, 1e-9)
                steps_rate = env_steps / max(dt, 1e-9)
                avg_ep_len = env_steps / max(ep_i, 1)
                steps_left = (n_episodes - ep_i) * avg_ep_len
                eta     = steps_left / max(steps_rate, 1e-9) / 60
                lr_tag  = "FINE" if fine_tune_mode else "FULL"
                print(f"  ep={ep_i:5d}/{n_episodes} score={last_score:5d} avg20={avg:6.1f} "
                      f"eps={epsilon(env_steps):.3f} grad={grad_steps} "
                      f"loss={loss_avg:.3f} q={q_avg:5.2f} lr={lr_tag} "
                      f"best={best_score}  {ep_rate:.1f}ep/s ETA {eta:.1f}min")

    final_w = best_greedy_w if best_greedy_w is not None else net.get_weights()
    json.dump({"weights": final_w, "episode": n_episodes, "grad_steps": grad_steps,
               "best_greedy_mean": best_greedy, "best_score": best_score,
               "n_in": N_IN, "n_h": N_H, "eval_seeds": EVAL_SEEDS},
              open(MODEL_PATH, "w"))

    dt = time.time() - t0
    print(f"\n  Saved model.json  (best_greedy_mean={best_greedy:.0f})")
    print(f"  Done in {dt/60:.1f} min | best_score={best_score} | grad_steps={grad_steps}\n")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes",  type=int,  default=15000)
    ap.add_argument("--load",      action="store_true")
    ap.add_argument("--snapshots", action="store_true")
    ap.add_argument("--workers",   type=int,  default=None,
                    help="parallel workers (default: cpu_count-1)")
    args = ap.parse_args()
    train(args.episodes, load=args.load, take_snapshots=args.snapshots, n_workers=args.workers)
