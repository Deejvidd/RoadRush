"""
Road Rush — Imitation Learning + DQN Reinforcement Learning Server
===================================================================
Install:  pip install flask numpy
Run:      python ai_server_rl.py  →  opens http://127.0.0.1:8080

Algorithm: DQN (Deep Q-Network) — pure numpy, from scratch
  - Q-network: 11 → 64 → 64 → 3 (Q-values for left/none/right)
  - Target network (soft Polyak update, τ=TAU each gradient step)
  - Double DQN target (online picks action, target evaluates)
  - Huber loss + reward scaling for stability
  - Experience replay buffer (uniform sample)
  - Epsilon-greedy exploration with linear decay
  - Adam optimizer, MSE loss, gradient clipping
  - Single fixed seed for all rollouts (training + inference)

Workflow:
  1. Optional: RECORD → play the game, demos collected
  2. Optional: PRETRAIN → net learns to copy your moves (behavioral cloning)
  3. RL ON → DQN improves the policy from reward
"""

from flask import Flask, request, jsonify, send_file, Response
import numpy as np, json, os, threading, webbrowser, random, datetime, sys
from collections import deque

if hasattr(sys.stdout, 'reconfigure'):
    try: sys.stdout.reconfigure(encoding='utf-8')
    except Exception: pass

app = Flask(__name__)

# ── Hyperparameters ─────────────────────────────────────────────────────────────
GAMMA         = 0.99
LR            = 5e-4
LR_IL         = 0.002
GRAD_CLIP     = 1.0
BETA1, BETA2  = 0.9, 0.999
BUFFER_SIZE   = 50000
BATCH_SIZE    = 64
TAU           = 0.005     # soft target update (Polyak averaging)
WARMUP_STEPS  = 1000      # collect transitions before training
TRAIN_EVERY   = 4         # gradient step cadence (every N env steps)
REWARD_SCALE  = 0.1       # divide all rewards by 10 (stabilizes targets)
HUBER_DELTA   = 1.0       # Huber loss transition point
EPS_START     = 1.0
EPS_END       = 0.05
EPS_DECAY_STEPS = 80000
IL_EPOCHS     = 300
N_IN, N_H, N_OUT = 30, 256, 3
N_OBS_SLOTS    = 4
GRID_COLS      = 5
GRID_ROWS      = 3
GRID_ROW_BANDS = [(0, 120), (120, 260), (260, 430)]
FIXED_SEED    = 42

# ── Global state ────────────────────────────────────────────────────────────────
il_demos   = []
step_count = 0
episode    = 0
ep_scores  = []
il_trained = False

dqn_state = {
    'grad_steps': 0,
    'env_steps': 0,
    'epsilon': EPS_START,
    'loss_avg': 0.0,
    'q_avg': 0.0,
}

replay = deque(maxlen=BUFFER_SIZE)
pending = {'state': None, 'action': None}   # waiting for reward from next call

# ── DQN Network ─────────────────────────────────────────────────────────────────
class DQN:
    """
    Fully connected Q-network (11 → 64 → 64 → 3) trained with Adam + MSE.
    No critic head, no softmax — just raw Q-values.
    """
    def __init__(self):
        self.W1 = np.random.randn(N_H, N_IN)  * np.sqrt(2 / N_IN)
        self.b1 = np.zeros(N_H)
        self.W2 = np.random.randn(N_H, N_H)   * np.sqrt(2 / N_H)
        self.b2 = np.zeros(N_H)
        self.W3 = np.random.randn(N_OUT, N_H) * np.sqrt(2 / N_H)
        self.b3 = np.zeros(N_OUT)
        self._t = 0
        for name in ('W1','b1','W2','b2','W3','b3'):
            w = getattr(self, name)
            setattr(self, '_m_' + name, np.zeros_like(w))
            setattr(self, '_v_' + name, np.zeros_like(w))

    def forward(self, X):
        X = np.asarray(X, dtype=np.float32)
        if X.ndim == 1: X = X[np.newaxis, :]
        H1 = np.maximum(0, X  @ self.W1.T + self.b1)
        H2 = np.maximum(0, H1 @ self.W2.T + self.b2)
        Q  = H2 @ self.W3.T + self.b3
        return Q, H2, H1

    def q(self, state):
        Q, _, _ = self.forward(state)
        return Q[0]

    def _adam_step(self, lr, **grads):
        self._t += 1
        t = self._t
        for name, g in grads.items():
            m = BETA1 * getattr(self, '_m_' + name) + (1 - BETA1) * g
            v = BETA2 * getattr(self, '_v_' + name) + (1 - BETA2) * g**2
            setattr(self, '_m_' + name, m)
            setattr(self, '_v_' + name, v)
            m_hat = m / (1 - BETA1**t)
            v_hat = v / (1 - BETA2**t)
            setattr(self, name, getattr(self, name) - lr * m_hat / (np.sqrt(v_hat) + 1e-8))

    def train_step(self, states, actions, targets, lr=LR):
        """One gradient step on a minibatch. targets[i] = r + γ·max_a' Q_target(s',a') · (1-done)."""
        B = len(states)
        Q, H2, H1 = self.forward(states)
        q_pred = Q[np.arange(B), actions]
        err = q_pred - targets
        # Huber loss gradient: err clipped to [-delta, +delta]
        dQ = np.zeros_like(Q)
        dQ[np.arange(B), actions] = np.clip(err, -HUBER_DELTA, HUBER_DELTA) / B
        dW3 = dQ.T @ H2
        db3 = dQ.sum(0)
        dH2 = dQ @ self.W3
        dH2 *= (H2 > 0)
        dW2 = dH2.T @ H1
        db2 = dH2.sum(0)
        dH1 = dH2 @ self.W2
        dH1 *= (H1 > 0)
        dW1 = dH1.T @ np.asarray(states, dtype=np.float32)
        db1 = dH1.sum(0)
        grads = [dW1, db1, dW2, db2, dW3, db3]
        norm = float(np.sqrt(sum((g*g).sum() for g in grads)))
        if norm > GRAD_CLIP:
            s = GRAD_CLIP / (norm + 1e-8)
            dW1, db1, dW2, db2, dW3, db3 = [g * s for g in grads]
        self._adam_step(lr, W1=dW1, b1=db1, W2=dW2, b2=db2, W3=dW3, b3=db3)
        return float((err**2).mean()), float(q_pred.mean())

    def il_step(self, states, labels, lr=LR_IL):
        """Behavioral cloning via cross-entropy over softmax(Q)."""
        X = np.asarray(states, dtype=np.float32)
        Y = np.asarray(labels, dtype=np.int32)
        B = len(Y)
        H1 = np.maximum(0, X  @ self.W1.T + self.b1)
        H2 = np.maximum(0, H1 @ self.W2.T + self.b2)
        L  = H2 @ self.W3.T + self.b3
        ex = np.exp(L - L.max(axis=1, keepdims=True))
        P  = ex / ex.sum(axis=1, keepdims=True)
        loss = -np.log(P[np.arange(B), Y] + 1e-9).mean()
        dL = P.copy()
        dL[np.arange(B), Y] -= 1
        dL /= B
        dW3 = dL.T @ H2; db3 = dL.sum(0)
        dH2 = dL @ self.W3; dH2 *= (H2 > 0)
        dW2 = dH2.T @ H1;  db2 = dH2.sum(0)
        dH1 = dH2 @ self.W2; dH1 *= (H1 > 0)
        dW1 = dH1.T @ X;   db1 = dH1.sum(0)
        self.W3 -= lr * dW3; self.b3 -= lr * db3
        self.W2 -= lr * dW2; self.b2 -= lr * db2
        self.W1 -= lr * dW1; self.b1 -= lr * db1
        return float(loss)

    def accuracy(self, X, Y):
        Q, _, _ = self.forward(X)
        return float((Q.argmax(axis=1) == np.asarray(Y)).mean())

    def get_weights(self):
        return {k: getattr(self, k).tolist() for k in ('W1','b1','W2','b2','W3','b3')}

    def set_weights(self, w):
        for k in ('W1','b1','W2','b2','W3','b3'):
            if k in w:
                arr = np.array(w[k], dtype=np.float32)
                if arr.shape == getattr(self, k).shape:
                    setattr(self, k, arr)
        # Reset Adam moments
        self._t = 0
        for name in ('W1','b1','W2','b2','W3','b3'):
            w_arr = getattr(self, name)
            setattr(self, '_m_' + name, np.zeros_like(w_arr))
            setattr(self, '_v_' + name, np.zeros_like(w_arr))


net        = DQN()
target_net = DQN()
target_net.set_weights(net.get_weights())


def _sync_target():
    """Hard copy online weights → target. Used on model load."""
    target_net.set_weights(net.get_weights())


def _soft_update_target():
    """Polyak averaging: target ← (1-τ)·target + τ·online. Called every gradient step."""
    for k in ('W1','b1','W2','b2','W3','b3'):
        setattr(target_net, k,
                (1 - TAU) * getattr(target_net, k) + TAU * getattr(net, k))


def _epsilon(step):
    if step >= EPS_DECAY_STEPS: return EPS_END
    frac = step / EPS_DECAY_STEPS
    return EPS_START + (EPS_END - EPS_START) * frac


def _train_from_replay():
    """Sample a minibatch and do one gradient step."""
    if len(replay) < max(BATCH_SIZE, WARMUP_STEPS):
        return
    batch = random.sample(replay, BATCH_SIZE)
    states  = np.array([b[0] for b in batch], dtype=np.float32)
    actions = np.array([b[1] for b in batch], dtype=np.int32)
    rewards = np.array([b[2] for b in batch], dtype=np.float32)
    next_s  = np.array([b[3] for b in batch], dtype=np.float32)
    dones   = np.array([b[4] for b in batch], dtype=np.float32)
    # Double DQN: online picks action, target evaluates it
    Q_online_next, _, _ = net.forward(next_s)
    a_star = Q_online_next.argmax(axis=1)
    Q_target_next, _, _ = target_net.forward(next_s)
    targets = rewards + GAMMA * Q_target_next[np.arange(BATCH_SIZE), a_star] * (1.0 - dones)
    loss, q_avg = net.train_step(states, actions, targets, lr=LR)
    dqn_state['grad_steps'] += 1
    dqn_state['loss_avg'] = 0.98 * dqn_state['loss_avg'] + 0.02 * loss
    dqn_state['q_avg']    = 0.98 * dqn_state['q_avg']    + 0.02 * q_avg
    _soft_update_target()


# ── Static file serving ──────────────────────────────────────────────────────────
STATIC_DIR      = os.path.dirname(os.path.abspath(__file__))
IL_SESSIONS_DIR = os.path.join(STATIC_DIR, 'il_sessions')

@app.route('/')
def index():
    p = os.path.join(STATIC_DIR, 'road_rush.html')
    return send_file(p) if os.path.exists(p) else ("road_rush.html not found", 404)

@app.route('/<path:filename>')
def static_files(filename):
    allowed = {'style.css', 'game.js'}
    if filename in allowed:
        return send_file(os.path.join(STATIC_DIR, filename))
    if filename.startswith('snapshots/') and filename.endswith('.json'):
        fpath = os.path.join(STATIC_DIR, filename)
        if os.path.exists(fpath): return send_file(fpath)
    if filename.startswith('il_sessions/') and filename.endswith('.json'):
        fpath = os.path.join(STATIC_DIR, filename)
        if os.path.exists(fpath): return send_file(fpath)
    return ("Not found", 404)

# ── Status ───────────────────────────────────────────────────────────────────────
@app.route('/status', methods=['GET'])
def status():
    return jsonify({
        "mode": "IL+DQN",
        "il_demos": len(il_demos),
        "il_trained": il_trained,
        "episode": episode,
        "steps": step_count,
        "entropy": round(dqn_state['epsilon'], 4),  # browser reads 'entropy' field; we report epsilon
        "epsilon": round(dqn_state['epsilon'], 4),
        "replay_size": len(replay),
        "grad_steps": dqn_state['grad_steps'],
        "loss": round(dqn_state['loss_avg'], 4),
        "q_avg": round(dqn_state['q_avg'], 3),
        "avg_score": round(float(np.mean(ep_scores[-20:])), 1) if ep_scores else 0
    })

# ── Imitation Learning ───────────────────────────────────────────────────────────
@app.route('/record_batch', methods=['POST'])
def record_batch():
    batch = request.get_json()
    il_demos.extend(batch)
    return jsonify({"recorded": len(il_demos)})

@app.route('/pretrain_stream')
def pretrain_stream():
    def generate():
        global il_trained
        if len(il_demos) < 50:
            yield f"data: {json.dumps({'error': f'Need >=50 demos, have {len(il_demos)}'})}\n\n"
            return
        X = np.array([d['x']     for d in il_demos], dtype=np.float32)
        Y = np.array([d['label'] for d in il_demos], dtype=np.int32)
        counts = [int((Y==i).sum()) for i in range(3)]
        print(f"\n  [IL] demos={len(il_demos)}  L={counts[0]} N={counts[1]} R={counts[2]}")
        max_c = max(counts)
        Xb, Yb = [], []
        for cls in range(3):
            idx = np.where(Y == cls)[0]
            if len(idx) == 0: continue
            rep = np.resize(idx, max_c)
            Xb.append(X[rep]); Yb.append(np.full(max_c, cls, dtype=np.int32))
        Xb = np.vstack(Xb); Yb = np.concatenate(Yb)
        perm = np.random.permutation(len(Xb))
        Xb, Yb = Xb[perm], Yb[perm]
        print(f"  [IL] training {IL_EPOCHS} epochs on {len(Xb)} balanced samples...")
        bs = 128
        best_acc = 0.0
        patience = 0
        final_acc = 0.0
        for ep_i in range(IL_EPOCHS):
            idx = np.random.permutation(len(Xb))
            for s in range(0, len(Xb), bs):
                b = idx[s:s+bs]
                net.il_step(Xb[b], Yb[b])
            acc = float(net.accuracy(X, Y))
            final_acc = acc
            progress = int((ep_i + 1) / IL_EPOCHS * 100)
            yield f"data: {json.dumps({'progress': progress, 'accuracy': round(acc, 3)})}\n\n"
            if acc > 0.93:
                print(f"  [IL] early stop at epoch {ep_i} — acc={acc:.1%}")
                break
            if acc > best_acc + 0.005:
                best_acc = acc; patience = 0
            else:
                patience += 1
                if patience >= 15:
                    print(f"  [IL] early stop (plateau) at epoch {ep_i}"); break
        il_trained = True
        _sync_target()  # copy IL weights into target so DQN bootstrapping starts sane
        print(f"  [IL] done -- acc={final_acc:.1%}\n")
        session_file = None
        try:
            os.makedirs(IL_SESSIONS_DIR, exist_ok=True)
            ts = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
            session_data = {
                "timestamp": ts, "accuracy": round(final_acc, 4),
                "sample_count": len(il_demos), "episode": episode,
                "weights": net.get_weights(), "il_trained": True
            }
            session_file = f'il_session_{ts}.json'
            json.dump(session_data, open(os.path.join(IL_SESSIONS_DIR, session_file), 'w'))
            print(f"  [IL] Session saved: {session_file}")
        except Exception as e:
            print(f"  [IL] Warning: could not save session: {e}")
        yield f"data: {json.dumps({'done': True, 'ok': True, 'accuracy': round(final_acc, 3), 'samples': len(il_demos), 'session_file': session_file})}\n\n"
    return Response(generate(), mimetype='text/event-stream',
                    headers={'Cache-Control': 'no-cache', 'X-Accel-Buffering': 'no'})

@app.route('/il_reset', methods=['POST'])
def il_reset():
    global il_demos, il_trained
    il_demos = []; il_trained = False
    return jsonify({"ok": True})

# ── DQN step ────────────────────────────────────────────────────────────────────
@app.route('/step', methods=['POST'])
def step_route():
    global step_count
    data   = request.get_json()
    state  = np.array(data['state'], dtype=np.float32)
    reward = float(data.get('reward', 0.0))
    done   = bool(data.get('done', False))

    # Close previous pending transition
    if pending['state'] is not None:
        replay.append((pending['state'].copy(), int(pending['action']),
                       reward * REWARD_SCALE, state.copy(), float(done)))
        dqn_state['env_steps'] += 1
        step_count += 1
        if dqn_state['env_steps'] % TRAIN_EVERY == 0:
            _train_from_replay()

    if done:
        pending['state'] = None
        return jsonify({
            'action': 1, 'entropy': round(dqn_state['epsilon'], 4),
            'epsilon': round(dqn_state['epsilon'], 4),
            'replay_size': len(replay)
        })

    # ε-greedy
    eps = _epsilon(dqn_state['env_steps'])
    dqn_state['epsilon'] = eps
    q_vals = net.q(state)
    if random.random() < eps:
        action = random.randint(0, N_OUT - 1)
    else:
        action = int(np.argmax(q_vals))

    pending['state']  = state
    pending['action'] = action

    return jsonify({
        'action': action,
        'entropy': round(eps, 4),     # browser HUD reads 'entropy'; we show ε
        'epsilon': round(eps, 4),
        'q': [round(float(x), 3) for x in q_vals],
        'replay_size': len(replay)
    })

@app.route('/episode_end', methods=['POST'])
def episode_end():
    global episode
    data  = request.get_json()
    score = data.get('score', 0)
    ep_scores.append(score)
    avg   = float(np.mean(ep_scores[-20:])) if ep_scores else 0
    episode += 1
    best_tag = " * NEW BEST" if ep_scores and score == max(ep_scores) else ""
    print(f"  ep={episode:4d}  score={score:5d}  avg20={avg:6.1f}  eps={dqn_state['epsilon']:.3f}  "
          f"loss={dqn_state['loss_avg']:.3f}  q={dqn_state['q_avg']:.2f}  grad_steps={dqn_state['grad_steps']}{best_tag}")
    return jsonify({"ok": True})

# ── Save / Load ──────────────────────────────────────────────────────────────────
@app.route('/save', methods=['POST'])
def save():
    data = {"weights": net.get_weights(), "il_trained": il_trained,
            "episode": episode, "grad_steps": dqn_state['grad_steps'], "n_in": N_IN}
    json.dump(data, open('model.json', 'w'))
    print(f"  Saved (ep={episode}, grad_steps={dqn_state['grad_steps']}, n_in={N_IN})")
    return jsonify({"saved": True})

@app.route('/load', methods=['POST'])
def load():
    global il_trained, episode
    try:
        d = json.load(open('model.json'))
        stored_nin = d.get('n_in', None)
        stored_nh  = d.get('n_h',  None)
        if (stored_nin is not None and stored_nin != N_IN) or \
           (stored_nh  is not None and stored_nh  != N_H):
            msg = f"Model shape mismatch (n_in={stored_nin}, n_h={stored_nh}). Load aborted."
            print(f"  WARNING: {msg}")
            return jsonify({"error": msg}), 400
        w = d.get('weights', d)
        net.set_weights(w)
        _sync_target()
        il_trained = d.get('il_trained', False)
        episode    = d.get('episode', episode)
        dqn_state['grad_steps'] = d.get('grad_steps', 0)
        print(f"  Loaded model.json (ep={episode}, grad_steps={dqn_state['grad_steps']}, n_in={N_IN})")
        return jsonify({"loaded": True, "il_trained": il_trained})
    except FileNotFoundError:
        return jsonify({"error": "model.json not found"}), 404

# ── Snapshots ────────────────────────────────────────────────────────────────────
@app.route('/load_snapshot', methods=['POST'])
def load_snapshot():
    data  = request.get_json()
    fname = data.get('file', '')
    snap_path = os.path.join(STATIC_DIR, 'snapshots', fname)
    try:
        d = json.load(open(snap_path))
        net.set_weights(d['weights'])
        _sync_target()
        print(f"  [SNAP] Loaded {fname} (ep={d.get('episode')})")
        return jsonify({
            "loaded": True, "episode": d.get('episode'),
            "avg_score": d.get('avg_score'), "best_score": d.get('best_score')
        })
    except FileNotFoundError:
        return jsonify({"error": f"Snapshot not found: {fname}"}), 404

@app.route('/list_snapshots', methods=['GET'])
def list_snapshots():
    snap_dir = os.path.join(STATIC_DIR, 'snapshots')
    if not os.path.isdir(snap_dir):
        return jsonify({"snapshots": []})
    snaps = []
    for f in sorted(os.listdir(snap_dir)):
        if f.endswith('.json'):
            try:
                d = json.load(open(os.path.join(snap_dir, f)))
                snaps.append({"file": f, "episode": d.get("episode", 0),
                              "avg_score": d.get("avg_score", 0),
                              "best_score": d.get("best_score", 0)})
            except: pass
    return jsonify({"snapshots": snaps})

# ── IL Sessions ──────────────────────────────────────────────────────────────────
@app.route('/list_il_sessions', methods=['GET'])
def list_il_sessions():
    if not os.path.isdir(IL_SESSIONS_DIR):
        return jsonify({"sessions": []})
    sessions = []
    for f in sorted(os.listdir(IL_SESSIONS_DIR), reverse=True):
        if f.endswith('.json'):
            try:
                d = json.load(open(os.path.join(IL_SESSIONS_DIR, f)))
                sessions.append({
                    "file": f, "timestamp": d.get("timestamp", ""),
                    "accuracy": d.get("accuracy", 0),
                    "sample_count": d.get("sample_count", 0),
                    "episode": d.get("episode", 0)
                })
            except: pass
    return jsonify({"sessions": sessions})

@app.route('/load_il_session', methods=['POST'])
def load_il_session():
    global il_trained
    data  = request.get_json()
    fname = data.get('file', '')
    path  = os.path.join(IL_SESSIONS_DIR, fname)
    try:
        d = json.load(open(path))
        net.set_weights(d['weights'])
        _sync_target()
        il_trained = True
        print(f"  [IL] Loaded session {fname} (acc={d.get('accuracy', 0):.1%})")
        return jsonify({"loaded": True, "accuracy": d.get("accuracy"),
                        "timestamp": d.get("timestamp"),
                        "sample_count": d.get("sample_count")})
    except FileNotFoundError:
        return jsonify({"error": f"IL session not found: {fname}"}), 404
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# ── Weights (for local browser inference) ────────────────────────────────────────
@app.route('/get_weights', methods=['GET'])
def get_weights():
    return jsonify({"weights": net.get_weights(), "n_in": N_IN, "n_h": N_H, "n_out": N_OUT})

# ── Params (read-only) ───────────────────────────────────────────────────────────
@app.route('/get_params', methods=['GET'])
def get_params():
    return jsonify({
        "algo": "DQN", "gamma": GAMMA, "lr": LR,
        "buffer_size": BUFFER_SIZE, "batch_size": BATCH_SIZE,
        "tau": TAU, "warmup": WARMUP_STEPS,
        "reward_scale": REWARD_SCALE, "huber_delta": HUBER_DELTA,
        "double_dqn": True,
        "train_every": TRAIN_EVERY,
        "eps_start": EPS_START, "eps_end": EPS_END, "eps_decay_steps": EPS_DECAY_STEPS,
        "fixed_seed": FIXED_SEED,
        "grad_steps": dqn_state['grad_steps'],
        "epsilon": round(dqn_state['epsilon'], 4),
    })

# ══════════════════════════════════════════════════════════════════════════════
# HEADLESS GAME SIMULATOR — 5×3 occupancy grid, physics matches train3_grid.py
# ══════════════════════════════════════════════════════════════════════════════
class HeadlessGame:
    W, H        = 480, 720
    ROAD_L      = 75
    ROAD_R      = 405
    ROAD_W      = ROAD_R - ROAD_L
    BASE_SPEED  = 3.5
    SPAWN_INTERVAL = 60
    SPAWN_COUNT    = 1

    def __init__(self): self.reset()

    def reset(self, seed=FIXED_SEED):
        self.rng         = np.random.RandomState(seed)
        self.px          = self.W / 2
        self.py          = self.H - 130
        self.pw, self.ph = 42, 70
        self.pvx         = 0.0
        self.obstacles   = []
        self.score       = 0
        self.frame       = 0
        self.level       = 1
        self.spawn_timer = 0
        self._ms_hit     = set()
        return self.get_features()

    def _grid_occupancy(self):
        col_w    = self.ROAD_W / GRID_COLS
        player_y = self.py + self.ph / 2
        grid = []
        for (near, far) in GRID_ROW_BANDS:
            y_bottom = player_y - near
            y_top    = player_y - far
            for ci in range(GRID_COLS):
                cell_x1 = self.ROAD_L + ci * col_w
                cell_x2 = cell_x1 + col_w
                hit = 0.0
                for o in self.obstacles:
                    ox1 = o['x'] - o['w'] / 2
                    ox2 = o['x'] + o['w'] / 2
                    oy1 = o['y']
                    oy2 = o['y'] + o['h']
                    if ox2 > cell_x1 and ox1 < cell_x2 and oy2 > y_top and oy1 < y_bottom:
                        hit = 1.0
                        break
                grid.append(hit)
        return grid   # 15 values

    def get_features(self):
        """
        30-dimensional observation (must match train3_grid.py exactly):
          [0]    player X normalised on road
          [1]    tanh(velocity / 8)
          [2-16] 5x3 occupancy grid (15 binary cells, row-major, closest first)
          [17]   spawn_timer / SPAWN_INTERVAL
          [18-29] 4 nearest obstacle slots (norm_x, norm_y_dist, norm_width)
                  padded with (0.5, 1.0, 0.12) when empty
        """
        base = [
            (self.px - self.ROAD_L) / self.ROAD_W,
            float(np.tanh(self.pvx / 8.0)),
        ]
        base += self._grid_occupancy()
        base += [self.spawn_timer / self.SPAWN_INTERVAL]
        upcoming = sorted(self.obstacles, key=lambda o: -o['y'])[:N_OBS_SLOTS]
        slots = []
        for o in upcoming:
            slots.append((o['x'] - self.ROAD_L) / self.ROAD_W)
            slots.append(max(0.0, self.py - o['y']) / self.H)
            slots.append(o['w'] / self.ROAD_W)
        while len(slots) < N_OBS_SLOTS * 3:
            slots.extend([0.5, 1.0, 0.12])
        return base + slots  # length = 30

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
        spd = self.BASE_SPEED
        if action == 0: self.pvx -= 1.2
        if action == 2: self.pvx += 1.2
        self.pvx *= 0.75
        self.px = np.clip(self.px + self.pvx,
                          self.ROAD_L + self.pw/2 + 2,
                          self.ROAD_R - self.pw/2 - 2)
        self.spawn_timer += 1
        if self.spawn_timer >= self.SPAWN_INTERVAL:
            self.spawn_timer = 0
            m  = 24
            ox = self.ROAD_L + m + self.rng.random() * (self.ROAD_W - m * 2)
            self.obstacles.append({'x': ox, 'y': -90, 'w': 40, 'h': 68,
                                   'speed': spd + 0.3 + self.rng.random() * 1.0})
        for o in self.obstacles: o['y'] += o['speed']
        self.obstacles = [o for o in self.obstacles if o['y'] < self.H + 100]
        m = 6
        for o in self.obstacles:
            if (abs(self.px - o['x']) < (self.pw/2 + o['w']/2 - m) and
                    abs((self.py + self.ph/2) - (o['y'] + o['h']/2)) < (self.ph/2 + o['h']/2 - m)):
                penalty = -(50.0 + 30.0 * min(1.0, self.score / 2500.0))
                return self.get_features(), penalty, True, self.score
        self.score += 1
        self.level = min(5, 1 + self.score // 500)
        milestone_bonus = 0.0
        for ms in (500, 1000, 1500, 2000):
            if self.score == ms and ms not in self._ms_hit:
                self._ms_hit.add(ms)
                milestone_bonus = 10.0
                break
        if self.score >= 2500:
            return self.get_features(), 50.0, True, self.score
        return self.get_features(), self.compute_reward() + milestone_bonus, False, self.score

# ── Fast headless DQN training ───────────────────────────────────────────────────
fast_train_status = {"running": False, "episode": 0, "total": 0,
                     "scores": [], "snapshots": [], "best_score": 0, "best_avg": 0,
                     "grad_steps": 0}


def _run_fast_training(n_episodes, snapshot_at):
    """Runs headless DQN training in a background thread."""
    fast_train_status.update({
        "running": True, "episode": 0, "total": n_episodes,
        "scores": [], "snapshots": [], "best_score": 0, "best_avg": 0, "grad_steps": 0
    })
    snap_dir = os.path.join(STATIC_DIR, 'snapshots')
    os.makedirs(snap_dir, exist_ok=True)

    local_net    = DQN(); local_net.set_weights(net.get_weights())
    local_target = DQN(); local_target.set_weights(local_net.get_weights())
    buf = deque(maxlen=BUFFER_SIZE)
    env_steps = 0
    grad_steps = 0

    game = HeadlessGame()

    for ep_i in range(1, n_episodes + 1):
        # Vary the seed each episode so the model learns a general policy,
        # not a memorised fixed obstacle sequence.
        ep_seed = (FIXED_SEED + ep_i) % (2**31)
        state = game.reset(seed=ep_seed)
        done  = False
        while not done:
            eps = EPS_START + (EPS_END - EPS_START) * min(1.0, env_steps / EPS_DECAY_STEPS)
            q_vals = local_net.q(np.asarray(state, dtype=np.float32))
            if random.random() < eps:
                action = random.randint(0, N_OUT - 1)
            else:
                action = int(np.argmax(q_vals))
            next_state, reward, done, score = game.step(action)
            buf.append((np.asarray(state, dtype=np.float32), action, reward * REWARD_SCALE,
                        np.asarray(next_state, dtype=np.float32), float(done)))
            env_steps += 1
            state = next_state
            if env_steps % TRAIN_EVERY == 0 and len(buf) >= max(BATCH_SIZE, WARMUP_STEPS):
                batch = random.sample(buf, BATCH_SIZE)
                S  = np.array([b[0] for b in batch], dtype=np.float32)
                A  = np.array([b[1] for b in batch], dtype=np.int32)
                R  = np.array([b[2] for b in batch], dtype=np.float32)
                S2 = np.array([b[3] for b in batch], dtype=np.float32)
                D  = np.array([b[4] for b in batch], dtype=np.float32)
                # Double DQN: online picks action, target evaluates
                Q_online_next, _, _ = local_net.forward(S2)
                a_star = Q_online_next.argmax(axis=1)
                Q_target_next, _, _ = local_target.forward(S2)
                targets = R + GAMMA * Q_target_next[np.arange(BATCH_SIZE), a_star] * (1.0 - D)
                local_net.train_step(S, A, targets, lr=LR)
                grad_steps += 1
                # Soft target update
                for kk in ('W1','b1','W2','b2','W3','b3'):
                    setattr(local_target, kk,
                            (1 - TAU) * getattr(local_target, kk) + TAU * getattr(local_net, kk))

        fast_train_status["scores"].append(score)
        fast_train_status["episode"] = ep_i
        fast_train_status["grad_steps"] = grad_steps
        if score > fast_train_status["best_score"]:
            fast_train_status["best_score"] = score
        avg = float(np.mean(fast_train_status["scores"][-20:]))
        if avg > fast_train_status["best_avg"]:
            fast_train_status["best_avg"] = round(avg, 1)

        if ep_i % 50 == 0 or ep_i <= 5:
            print(f"  [FAST] ep={ep_i:5d}/{n_episodes}  score={score:5d}  avg20={avg:6.1f}  "
                  f"eps={eps:.3f}  grad_steps={grad_steps}  replay={len(buf)}")

        if ep_i in snapshot_at:
            w = local_net.get_weights()
            snap_data = {"weights": w, "episode": ep_i,
                         "avg_score": round(avg, 1),
                         "best_score": fast_train_status["best_score"],
                         "grad_steps": grad_steps, "epsilon": round(eps, 4)}
            json.dump(snap_data, open(os.path.join(snap_dir, f'snapshot_ep{ep_i}.json'), 'w'))
            fast_train_status["snapshots"].append({
                "episode": ep_i, "file": f"snapshot_ep{ep_i}.json",
                "avg_score": round(avg, 1),
                "best_score": fast_train_status["best_score"]
            })
            print(f"  [FAST] Snapshot saved at ep {ep_i}  avg={avg:.1f}")

    net.set_weights(local_net.get_weights())
    _sync_target()
    json.dump({"weights": local_net.get_weights(), "il_trained": il_trained,
               "episode": episode + n_episodes, "grad_steps": grad_steps, "n_in": N_IN},
              open('model.json', 'w'))
    fast_train_status["running"] = False
    print(f"\n  [FAST] Done! {n_episodes} eps, best_score={fast_train_status['best_score']}, "
          f"best_avg20={fast_train_status['best_avg']:.1f}, grad_steps={grad_steps}\n")


@app.route('/train_fast', methods=['POST'])
def train_fast():
    if fast_train_status["running"]:
        return jsonify({"error": "Training already running"}), 409
    data = request.get_json()
    n_episodes  = int(data.get('episodes', 500))
    snapshot_at = data.get('snapshot_at', [1, 5, 10, 25, 50, 100, 200, 300, 500, 750, 1000])
    snapshot_at = [s for s in snapshot_at if s <= n_episodes]
    if n_episodes not in snapshot_at: snapshot_at.append(n_episodes)
    print(f"\n  [FAST] Starting {n_episodes} headless DQN episodes (fixed seed={FIXED_SEED})...")
    print(f"  [FAST] Snapshots at: {snapshot_at}")
    t = threading.Thread(target=_run_fast_training, args=(n_episodes, set(snapshot_at)), daemon=True)
    t.start()
    return jsonify({"ok": True, "episodes": n_episodes, "snapshot_at": snapshot_at})

@app.route('/train_fast_status', methods=['GET'])
def train_fast_status_route():
    return jsonify(fast_train_status)

# ──────────────────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    print("=" * 60)
    print("  Road Rush -- IL + DQN (pure numpy) Server")
    print("  http://127.0.0.1:8080")
    print()
    print("  Algorithm: DQN (Deep Q-Network) — 5×3 grid obs")
    print(f"  Network:   {N_IN} -> {N_H} -> {N_H} -> {N_OUT}  (obs: 2 pos + 15 grid + 1 spawn + 12 slots)")
    print(f"  Buffer={BUFFER_SIZE}  batch={BATCH_SIZE}  tau={TAU}  "
          f"reward_scale={REWARD_SCALE}  Double-DQN")
    print(f"  Epsilon: {EPS_START} -> {EPS_END} over {EPS_DECAY_STEPS} steps")
    print(f"  Fixed seed: {FIXED_SEED}  (uniform physics across all levels)")
    print("=" * 60)
    threading.Timer(1.5, lambda: webbrowser.open('http://127.0.0.1:8080')).start()
    app.run(host='0.0.0.0', debug=False, port=int(os.environ.get('PORT', 8080)))
