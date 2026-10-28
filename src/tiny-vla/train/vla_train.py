#!/usr/bin/env python3
"""TinyVLA v2: language-conditioned visual navigation on a 16x16 grid.

Observation
  - wall map (16x16)
  - three goal channels (red/green/blue, 16x16 each)
  - robot position
Language
  - bag-of-words instruction, e.g. "go to the red flag"
Architecture
  - language encoder -> colour attention (3-way softmax)
  - selected goal map = sum_c p(c) * goal_channel[c]
  - target position = argmax(selected goal map)
  - policy MLP over [wall, selected goal, local wall, robot pos, target pos,
    relative target, language]
Action
  - 4 discrete moves (up/down/left/right)

Behaviour cloning from a BFS expert, optional DAgger.  Exports a flat binary
for the Rust inference binary and a self-test vector file.
"""
import argparse
import json
import os
import struct
import time
from collections import deque

import numpy as np

N = 16
R = 3
LOCAL = 2 * R + 1
N_ACTIONS = 4
WALL_DIM = N * N
GOAL_DIM = N * N
LOCAL_DIM = LOCAL * LOCAL
POS_DIM = 2
LANG_DIM = 16
COLOR_DIM = 3
IN_DIM = WALL_DIM + WALL_DIM + LOCAL_DIM + LOCAL_DIM + POS_DIM + LANG_DIM + 4 + 4 + 4

VOCAB = [
    "go", "navigate", "head", "move", "reach", "find",
    "to", "for", "the",
    "red", "green", "blue",
    "goal", "flag", "target", "marker", "zone",
]
COLORS = ["red", "green", "blue"]
NOUNS = ["goal", "flag", "target", "marker", "zone"]
VERBS = ["go", "navigate", "head", "move", "reach", "find"]
PREPS = ["to", "for"]

DX = [0, 0, -1, 1]   # up, down, left, right
DY = [-1, 1, 0, 0]

H1 = 256
H2 = 128
COLOR_LOSS_WEIGHT = 0.5
GAIN_SCALE = 4.0
RESIDUAL_SCALE = 0.01
TIE_BIAS = np.array([0.0, -1e-4, -2e-4, -3e-4], dtype=np.float32)


# --------------------------------------------------------------------------
# map generation / expert
# --------------------------------------------------------------------------

def components(open_mask):
    seen = np.zeros_like(open_mask, dtype=bool)
    out = []
    ys, xs = np.nonzero(open_mask)
    for sy, sx in zip(ys.tolist(), xs.tolist()):
        if seen[sy, sx]:
            continue
        stack = [(sx, sy)]
        seen[sy, sx] = True
        comp = []
        while stack:
            x, y = stack.pop()
            comp.append((x, y))
            for d in range(4):
                nx, ny = x + DX[d], y + DY[d]
                if 0 <= nx < N and 0 <= ny < N and open_mask[ny, nx] and not seen[ny, nx]:
                    seen[ny, nx] = True
                    stack.append((nx, ny))
        out.append(comp)
    return out


def bfs_expert(passable, tx, ty):
    dist = np.full((N, N), -1, dtype=np.int16)
    first = np.full((N, N), -1, dtype=np.int8)
    if not passable[ty, tx]:
        return dist, first
    q = deque()
    dist[ty, tx] = 0
    q.append((tx, ty))
    back = {0: 1, 1: 0, 2: 3, 3: 2}
    while q:
        x, y = q.popleft()
        for d in range(4):
            nx, ny = x + DX[d], y + DY[d]
            if 0 <= nx < N and 0 <= ny < N and passable[ny, nx] and dist[ny, nx] < 0:
                dist[ny, nx] = dist[y, x] + 1
                first[ny, nx] = back[d]
                q.append((nx, ny))
    return dist, first


def make_map(rng, max_attempts=200):
    for _ in range(max_attempts):
        wall = np.zeros((N, N), dtype=np.uint8)
        for _ in range(int(rng.integers(4, 9))):
            x0 = int(rng.integers(0, N))
            y0 = int(rng.integers(0, N))
            w = int(rng.integers(1, 5))
            h = int(rng.integers(1, 5))
            wall[y0:min(N, y0 + h), x0:min(N, x0 + w)] = 1
        comps = sorted(components(wall == 0), key=len, reverse=True)
        if not comps or len(comps[0]) < 90:
            continue
        cells = comps[0]
        open_mask = np.zeros((N, N), dtype=bool)
        for x, y in cells:
            open_mask[y, x] = True
        wall = (~open_mask).astype(np.uint8)

        goals = []
        for _ in range(800):
            x, y = cells[int(rng.integers(0, len(cells)))]
            if any((x, y) == (gx, gy) for gx, gy in goals):
                continue
            if all(abs(x - gx) + abs(y - gy) >= 3 for gx, gy in goals):
                goals.append((x, y))
            if len(goals) == 3:
                break
        if len(goals) != 3:
            continue

        target = int(rng.integers(0, 3))
        tx, ty = goals[target]
        passable = open_mask.copy()
        for c, (gx, gy) in enumerate(goals):
            if c != target:
                passable[gy, gx] = False
        dist, first = bfs_expert(passable, tx, ty)
        dist_by_color, first_by_color = [], []
        for c in range(3):
            pc = open_mask.copy()
            for c2, (gx, gy) in enumerate(goals):
                if c2 != c:
                    pc[gy, gx] = False
            d_c, f_c = bfs_expert(pc, goals[c][0], goals[c][1])
            dist_by_color.append(d_c)
            first_by_color.append(f_c)

        candidates = [
            (x, y) for (x, y) in cells
            if passable[y, x] and dist[y, x] >= 6 and (x, y) not in goals
        ]
        if not candidates:
            continue
        rx, ry = candidates[int(rng.integers(0, len(candidates)))]
        if first[ry, rx] < 0:
            continue
        return {
            "wall": wall, "goals": goals, "target": target,
            "robot": (rx, ry), "passable": passable,
            "dist": dist, "first": first, "cells": cells,
            "dist_by_color": dist_by_color, "first_by_color": first_by_color,
        }
    raise RuntimeError("could not generate a valid map")


def make_instruction(rng, target):
    verb = VERBS[int(rng.integers(0, len(VERBS)))]
    prep = PREPS[int(rng.integers(0, len(PREPS)))]
    noun = NOUNS[int(rng.integers(0, len(NOUNS)))]
    return [verb, prep, "the", COLORS[target], noun]


def bow_vector(tokens):
    v = np.zeros(len(VOCAB), dtype=np.uint8)
    for t in tokens:
        v[VOCAB.index(t)] = 1
    return v


def goal_channels(m):
    g = np.zeros((N, N, 3), dtype=np.uint8)
    for c, (gx, gy) in enumerate(m["goals"]):
        g[gy, gx, c] = 1
    return g


def local_map(m, rx, ry, pad):
    padded = np.full((N + 2 * R, N + 2 * R), pad, dtype=m.dtype)
    padded[R:R + N, R:R + N] = m
    return padded[ry:ry + LOCAL, rx:rx + LOCAL].reshape(-1)


def local_wall(wall, rx, ry):
    return local_map(wall, int(round(rx)), int(round(ry)), 1)


# --------------------------------------------------------------------------
# dataset
# --------------------------------------------------------------------------

def gen_dataset(n_maps, seed, verbose=True):
    rng = np.random.default_rng(seed)
    walls, goals, dists, poss, bows, labels, map_ids, targets = [], [], [], [], [], [], [], []
    maps = []
    t0 = time.time()
    for mi in range(n_maps):
        m = make_map(rng)
        tokens = make_instruction(rng, m["target"])
        b = bow_vector(tokens)
        g = goal_channels(m)
        maps.append(m)

        states = []
        x, y = m["robot"]
        tx, ty = m["goals"][m["target"]]
        for _ in range(40):
            states.append((x, y))
            if (x, y) == (tx, ty):
                break
            a = int(m["first"][y, x])
            if a < 0:
                break
            x, y = x + DX[a], y + DY[a]
        for _ in range(4):
            x, y = m["cells"][int(rng.integers(0, len(m["cells"])))]
            if m["passable"][y, x] and m["dist"][y, x] >= 0:
                states.append((x, y))

        for (sx, sy) in states:
            a = int(m["first"][sy, sx])
            if a < 0:
                continue
            walls.append(m["wall"].reshape(-1))
            goals.append(g.reshape(-1))
            dists.append(m["dist"].reshape(-1).astype(np.int16))
            poss.append(np.array([sx / (N - 1), sy / (N - 1)], dtype=np.float32))
            bows.append(b)
            labels.append(a)
            map_ids.append(mi)
            targets.append(m["target"])
        if verbose and (mi + 1) % 2000 == 0:
            print(f"  maps {mi+1}/{n_maps}  samples {len(labels)}  "
                  f"{time.time()-t0:.1f}s", flush=True)
    data = {
        "wall": np.stack(walls).astype(np.uint8),
        "goal": np.stack(goals).astype(np.uint8),
        "dist": np.stack(dists).astype(np.int16),
        "pos": np.stack(poss).astype(np.float32),
        "bow": np.stack(bows).astype(np.uint8),
        "label": np.array(labels, dtype=np.uint8),
        "map_id": np.array(map_ids, dtype=np.int32),
        "target": np.array(targets, dtype=np.uint8),
    }
    return data, maps


# --------------------------------------------------------------------------
# model
# --------------------------------------------------------------------------

def relu(x):
    return np.maximum(x, 0.0)


def softmax(x):
    z = x - x.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


class Model:
    def __init__(self, seed=0):
        rng = np.random.default_rng(seed)
        self.p = {
            "Wl": rng.normal(0, np.sqrt(2.0 / len(VOCAB)), (len(VOCAB), LANG_DIM)).astype(np.float32),
            "bl": np.zeros(LANG_DIM, np.float32),
            "Wc": rng.normal(0, np.sqrt(2.0 / len(VOCAB)), (len(VOCAB), COLOR_DIM)).astype(np.float32),
            "bc": np.zeros(COLOR_DIM, np.float32),
            "W1": rng.normal(0, np.sqrt(2.0 / IN_DIM), (IN_DIM, H1)).astype(np.float32),
            "b1": np.zeros(H1, np.float32),
            "W2": rng.normal(0, np.sqrt(2.0 / H1), (H1, H2)).astype(np.float32),
            "b2": np.zeros(H2, np.float32),
            "W3": rng.normal(0, 0.01, (H2, N_ACTIONS)).astype(np.float32),
            "b3": np.zeros(N_ACTIONS, np.float32),
        }
        self.m = {k: np.zeros_like(v) for k, v in self.p.items()}
        self.v = {k: np.zeros_like(v) for k, v in self.p.items()}
        self.t = 0

    def _build_input(self, wall, goal, dist, pos, bow, lang):
        """wall/goal: (B,256) f32; dist (B,256) int16; pos (B,2); bow (B,V)."""
        B = wall.shape[0]
        color_logits = bow @ self.p["Wc"] + self.p["bc"]
        cp = softmax(color_logits)                      # B,3
        dnorm = np.where(dist >= 0, dist.astype(np.float32) / 4.0, 16.0)
        local_w = []
        local_d = []
        nb_d = np.full((B, 4), 16.0, dtype=np.float32)
        nb_w = np.ones((B, 4), dtype=np.float32)
        gain = np.full((B, 4), -16.0, dtype=np.float32)
        for i in range(B):
            rx = int(round(pos[i, 0] * (N - 1)))
            ry = int(round(pos[i, 1] * (N - 1)))
            w = wall[i].reshape(N, N)
            d = dnorm[i].reshape(N, N)
            raw = dist[i].reshape(N, N)
            local_w.append(local_map(w.astype(np.uint8), rx, ry, 0))
            local_d.append(local_map(d, rx, ry, 16.0))
            cur = d[ry, rx]
            for a in range(4):
                nx, ny = rx + DX[a], ry + DY[a]
                if 0 <= nx < N and 0 <= ny < N and w[ny, nx] == 0:
                    nb_w[i, a] = 0.0
                    if raw[ny, nx] >= 0:
                        nb_d[i, a] = d[ny, nx]
                        gain[i, a] = cur - d[ny, nx]
        local = np.stack(local_w).astype(np.float32)
        local_dist = np.stack(local_d).astype(np.float32)
        x = np.concatenate([wall, dnorm, local, local_dist, pos, lang,
                            nb_d, nb_w, gain], axis=1)
        return x, cp, color_logits, gain

    def forward(self, wall, goal, dist, pos, bow):
        lp = bow @ self.p["Wl"] + self.p["bl"]
        lang = relu(lp)
        x, cp, color_logits, gain = self._build_input(wall, goal, dist, pos, bow, lang)
        h1p = x @ self.p["W1"] + self.p["b1"]
        h1 = relu(h1p)
        h2p = h1 @ self.p["W2"] + self.p["b2"]
        h2 = relu(h2p)
        logits = (GAIN_SCALE * gain + RESIDUAL_SCALE * (h2 @ self.p["W3"] + self.p["b3"])
                  + TIE_BIAS)
        cache = (bow, lp, lang, x, h1p, h1, h2p, h2)
        return logits, color_logits, cp, cache

    def predict(self, wall, goal, dist, pos, bow):
        logits, color_logits, cp, _ = self.forward(wall, goal, dist, pos, bow)
        return logits.argmax(axis=1), color_logits.argmax(axis=1)

    def loss_and_grads(self, wall, goal, dist, pos, bow, y, color_y=None):
        logits, color_logits, cp, (bow, lp, lang, x, h1p, h1, h2p, h2) = self.forward(
            wall, goal, dist, pos, bow)
        B = logits.shape[0]
        shifted = logits - logits.max(axis=1, keepdims=True)
        exp = np.exp(shifted)
        p = exp / exp.sum(axis=1, keepdims=True)
        policy_loss = -np.log(p[np.arange(B), y] + 1e-12).mean()

        dlogits = p
        dlogits[np.arange(B), y] -= 1.0
        dlogits /= B

        g = {}
        dres = RESIDUAL_SCALE * dlogits
        g["W3"] = h2.T @ dres
        g["b3"] = dres.sum(axis=0)
        dh2 = dres @ self.p["W3"].T
        dh2[h2 <= 0] = 0
        g["W2"] = h1.T @ dh2
        g["b2"] = dh2.sum(axis=0)
        dh1 = dh2 @ self.p["W2"].T
        dh1[h1 <= 0] = 0
        g["W1"] = x.T @ dh1
        g["b1"] = dh1.sum(axis=0)
        dx = dh1 @ self.p["W1"].T
        dlang = dx[:, WALL_DIM + WALL_DIM + LOCAL_DIM + LOCAL_DIM + POS_DIM:WALL_DIM + WALL_DIM + LOCAL_DIM + LOCAL_DIM + POS_DIM + LANG_DIM]

        color_loss = 0.0
        if color_y is not None:
            cp_safe = np.clip(cp, 1e-9, 1.0)
            color_loss = -np.log(cp_safe[np.arange(B), color_y] + 1e-12).mean()
            dcl = cp.copy()
            dcl[np.arange(B), color_y] -= 1.0
            dcl /= B
            dcl *= COLOR_LOSS_WEIGHT
            g["Wc"] = bow.T @ dcl
            g["bc"] = dcl.sum(axis=0)
        else:
            g["Wc"] = np.zeros_like(self.p["Wc"])
            g["bc"] = np.zeros_like(self.p["bc"])

        dlp = dlang * (lp > 0)
        g["Wl"] = bow.T @ dlp
        g["bl"] = dlp.sum(axis=0)
        return policy_loss + COLOR_LOSS_WEIGHT * color_loss, g

    def adam_step(self, g, lr=1e-3, b1=0.9, b2=0.999, eps=1e-8, wd=0.0):
        self.t += 1
        for k in self.p:
            grad = g[k] + wd * self.p[k]
            self.m[k] = b1 * self.m[k] + (1 - b1) * grad
            self.v[k] = b2 * self.v[k] + (1 - b2) * (grad * grad)
            mh = self.m[k] / (1 - b1 ** self.t)
            vh = self.v[k] / (1 - b2 ** self.t)
            self.p[k] -= lr * mh / (np.sqrt(vh) + eps)


def train(model, data, epochs, batch, lr=1e-3, seed=0, val=None, verbose=True):
    rng = np.random.default_rng(seed)
    n = len(data["label"])
    best = None
    best_acc = -1.0
    best_score = -1.0
    for ep in range(epochs):
        idx = rng.permutation(n)
        tot = 0.0
        nb = 0
        for s in range(0, n, batch):
            j = idx[s:s + batch]
            wall = data["wall"][j].astype(np.float32)
            goal = data["goal"][j].astype(np.float32)
            dist = data["dist"][j]
            pos = data["pos"][j]
            bow = data["bow"][j].astype(np.float32)
            y = data["label"][j].astype(np.int64)
            cy = data["target"][j].astype(np.int64)
            loss, g = model.loss_and_grads(wall, goal, dist, pos, bow, y, cy)
            model.adam_step(g, lr=lr)
            tot += loss
            nb += 1
        acc = None
        cacc = None
        if val is not None:
            acc = action_accuracy(model, val)
            cacc = color_accuracy(model, val)
            score = acc + cacc
            if score > best_score:
                best_score = score
                best_acc = acc
                best = {k: v.copy() for k, v in model.p.items()}
        if verbose:
            msg = f"epoch {ep+1:02d}/{epochs} loss {tot/nb:.4f}"
            if acc is not None:
                msg += (f" val_acc {acc*100:.2f}% colour {cacc*100:.2f}%"
                        f" (best joint {best_score:.3f})")
            print(msg, flush=True)
    if best is not None:
        model.p = best
    return model


def action_accuracy(model, data, batch=1024):
    n = len(data["label"])
    correct = 0
    color_correct = 0
    for s in range(0, n, batch):
        j = slice(s, s + batch)
        pred, cpred = model.predict(
            data["wall"][j].astype(np.float32),
            data["goal"][j].astype(np.float32),
            data["dist"][j],
            data["pos"][j],
            data["bow"][j].astype(np.float32))
        correct += int((pred == data["label"][j]).sum())
        color_correct += int((cpred == data["target"][j]).sum())
    return correct / n


def valid_action_accuracy(model, data, batch=1024):
    """Fraction of predictions that take a maximum distance-gain action."""
    n = len(data["label"])
    correct = 0
    for s in range(0, n, batch):
        j = slice(s, s + batch)
        pred, _ = model.predict(
            data["wall"][j].astype(np.float32),
            data["goal"][j].astype(np.float32),
            data["dist"][j],
            data["pos"][j],
            data["bow"][j].astype(np.float32))
        for k, (i, a) in enumerate(zip(range(j.start, j.stop), pred)):
            d = data["dist"][i].reshape(N, N).astype(float)
            w = data["wall"][i].reshape(N, N)
            x = int(round(data["pos"][i, 0] * (N - 1)))
            y = int(round(data["pos"][i, 1] * (N - 1)))
            gains = []
            for aa in range(4):
                nx, ny = x + DX[aa], y + DY[aa]
                if 0 <= nx < N and 0 <= ny < N and w[ny, nx] == 0 and d[ny, nx] >= 0:
                    gains.append(d[y, x] - d[ny, nx])
                else:
                    gains.append(-999)
            if gains[int(a)] >= max(gains) - 1e-6:
                correct += 1
    return correct / n


def color_accuracy(model, data, batch=1024):
    n = len(data["label"])
    correct = 0
    for s in range(0, n, batch):
        j = slice(s, s + batch)
        _, cpred = model.predict(
            data["wall"][j].astype(np.float32),
            data["goal"][j].astype(np.float32),
            data["dist"][j],
            data["pos"][j],
            data["bow"][j].astype(np.float32))
        correct += int((cpred == data["target"][j]).sum())
    return correct / n


# --------------------------------------------------------------------------
# closed-loop evaluation / DAgger
# --------------------------------------------------------------------------

def obs_from_map(m, x, y):
    return (m["wall"].reshape(-1)[None].astype(np.float32),
            goal_channels(m).reshape(-1)[None].astype(np.float32),
            m["dist"].reshape(-1)[None],
            np.array([[x / (N - 1), y / (N - 1)]], dtype=np.float32))


def episode(model, m, tokens, max_steps=64, trace=False):
    b = bow_vector(tokens)[None].astype(np.float32)
    x, y = m["robot"]
    tx, ty = m["goals"][m["target"]]
    wrong = {m["goals"][c] for c in range(3) if c != m["target"]}
    collisions = 0
    path = [(x, y)]
    for step in range(max_steps):
        wall, goal, dist, pos = obs_from_map(m, x, y)
        a, _ = model.predict(wall, goal, dist, pos, b)
        a = int(a[0])
        nx, ny = x + DX[a], y + DY[a]
        if not (0 <= nx < N and 0 <= ny < N) or m["wall"][ny, nx]:
            collisions += 1
            continue
        if (nx, ny) == (tx, ty):
            path.append((nx, ny))
            return {"success": True, "steps": step + 1, "wrong": False,
                    "collisions": collisions, "path": path if trace else None}
        if (nx, ny) in wrong:
            path.append((nx, ny))
            return {"success": False, "steps": step + 1, "wrong": True,
                    "collisions": collisions, "path": path if trace else None}
        x, y = nx, ny
        path.append((x, y))
    return {"success": False, "steps": max_steps, "wrong": False,
            "collisions": collisions, "path": path if trace else None}


def episode_deploy(model, m, tokens, max_steps=64):
    """Use the model's language grounding to pick the distance field."""
    b = bow_vector(tokens)[None].astype(np.float32)
    wall = m["wall"].reshape(-1)[None].astype(np.float32)
    goal = goal_channels(m).reshape(-1)[None].astype(np.float32)
    x, y = m["robot"]
    pos = np.array([[x / (N - 1), y / (N - 1)]], dtype=np.float32)
    _, cpred = model.predict(wall, goal, m["dist"].reshape(-1)[None], pos, b)
    color = int(cpred[0])
    dmap = m["dist_by_color"][color]
    tx, ty = m["goals"][m["target"]]
    wrong = {m["goals"][c] for c in range(3) if c != m["target"]}
    collisions = 0
    for step in range(max_steps):
        pos = np.array([[x / (N - 1), y / (N - 1)]], dtype=np.float32)
        a, _ = model.predict(wall, goal, dmap.reshape(-1)[None], pos, b)
        a = int(a[0])
        nx, ny = x + DX[a], y + DY[a]
        if not (0 <= nx < N and 0 <= ny < N) or m["wall"][ny, nx]:
            collisions += 1
            continue
        if (nx, ny) == (tx, ty):
            return {"success": True, "steps": step + 1, "wrong": False,
                    "collisions": collisions, "path": None}
        if (nx, ny) in wrong:
            return {"success": False, "steps": step + 1, "wrong": True,
                    "collisions": collisions, "path": None}
        x, y = nx, ny
    return {"success": False, "steps": max_steps, "wrong": False,
            "collisions": collisions, "path": None}


def evaluate(model, maps, seed=1234, max_steps=64):
    rng = np.random.default_rng(seed)
    res = []
    for m in maps:
        tokens = make_instruction(rng, m["target"])
        res.append(episode_deploy(model, m, tokens, max_steps=max_steps))
    n = len(res)
    return {
        "episodes": n,
        "success": sum(r["success"] for r in res) / n,
        "wrong_goal": sum(r["wrong"] for r in res) / n,
        "timeout": sum((not r["success"] and not r["wrong"]) for r in res) / n,
        "avg_steps": float(np.mean([r["steps"] for r in res])),
        "avg_collisions": float(np.mean([r["collisions"] for r in res])),
    }


def dagger_collect(model, maps, rng, max_steps=64, max_states=40000):
    walls, goals, dists, poss, bows, labels, targets = [], [], [], [], [], [], []
    for m in maps:
        tokens = make_instruction(rng, m["target"])
        b = bow_vector(tokens)
        g = goal_channels(m).reshape(-1)
        x, y = m["robot"]
        tx, ty = m["goals"][m["target"]]
        wrong = {m["goals"][c] for c in range(3) if c != m["target"]}
        for _ in range(max_steps):
            if m["first"][y, x] >= 0:
                walls.append(m["wall"].reshape(-1))
                goals.append(g)
                dists.append(m["dist"].reshape(-1).astype(np.int16))
                poss.append(np.array([x / (N - 1), y / (N - 1)], dtype=np.float32))
                bows.append(b)
                labels.append(int(m["first"][y, x]))
                targets.append(m["target"])
            wall, goal, dist, pos = obs_from_map(m, x, y)
            a, _ = model.predict(wall, goal, dist, pos, b[None].astype(np.float32))
            a = int(a[0])
            nx, ny = x + DX[a], y + DY[a]
            if not (0 <= nx < N and 0 <= ny < N) or m["wall"][ny, nx]:
                continue
            if (nx, ny) == (tx, ty) or (nx, ny) in wrong:
                break
            x, y = nx, ny
            if len(labels) >= max_states:
                break
        if len(labels) >= max_states:
            break
    if not labels:
        return None
    return {
        "wall": np.stack(walls).astype(np.uint8),
        "goal": np.stack(goals).astype(np.uint8),
        "dist": np.stack(dists).astype(np.int16),
        "pos": np.stack(poss).astype(np.float32),
        "bow": np.stack(bows).astype(np.uint8),
        "label": np.array(labels, dtype=np.uint8),
        "map_id": np.zeros(len(labels), dtype=np.int32),
        "target": np.array(targets, dtype=np.uint8),
    }


# --------------------------------------------------------------------------
# export
# --------------------------------------------------------------------------

def export_binary(model, path):
    p = model.p
    with open(path, "wb") as f:
        f.write(b"TVLA")
        f.write(struct.pack("<8I", 3, IN_DIM, H1, H2, N_ACTIONS,
                            LANG_DIM, len(VOCAB), COLOR_DIM))
        for k in ["Wl", "bl", "Wc", "bc", "W1", "b1", "W2", "b2", "W3", "b3"]:
            f.write(np.ascontiguousarray(p[k], dtype="<f4").tobytes())
    print(f"wrote {path} ({os.path.getsize(path)} bytes)")


def export_test_vectors(model, data, path, count=8, seed=7):
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(data["label"]), size=count, replace=False)
    with open(path, "wb") as f:
        f.write(b"TVT3")
        f.write(struct.pack("<3I", count, IN_DIM, len(VOCAB)))
        for i in idx:
            f.write(data["wall"][i].tobytes())
            f.write(data["goal"][i].tobytes())
            f.write(np.asarray(data["dist"][i], dtype="<i2").tobytes())
            f.write(np.asarray(data["pos"][i], dtype="<f4").tobytes())
            f.write(data["bow"][i].tobytes())
            logits, color_logits, cp, _ = model.forward(
                data["wall"][i][None].astype(np.float32),
                data["goal"][i][None].astype(np.float32),
                data["dist"][i][None],
                data["pos"][i][None],
                data["bow"][i][None].astype(np.float32))
            f.write(np.asarray(logits[0], dtype="<f4").tobytes())
            f.write(np.asarray(color_logits[0], dtype="<f4").tobytes())
    print(f"wrote {path} ({os.path.getsize(path)} bytes)")


def export_vocab_rs(path):
    def arr(name, xs):
        return f"pub const {name}: &[&str] = &[" + ", ".join(json.dumps(x) for x in xs) + "];\n"
    with open(path, "w") as f:
        f.write("// generated by train/vla_train.py\n")
        f.write(arr("VOCAB", VOCAB))
        f.write(arr("COLORS", COLORS))
        f.write(arr("NOUNS", NOUNS))
        f.write(arr("VERBS", VERBS))
        f.write(arr("PREPS", PREPS))
        f.write(f"pub const N_ACTIONS: usize = {N_ACTIONS};\n")
        f.write(f"pub const N: usize = {N};\n")
        f.write(f"pub const R: usize = {R};\n")
        f.write(f"pub const LANG_DIM: usize = {LANG_DIM};\n")
        f.write(f"pub const COLOR_DIM: usize = {COLOR_DIM};\n")
    print(f"wrote {path}")


# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------

def main():
    global H1, H2
    ap = argparse.ArgumentParser()
    ap.add_argument("--maps", type=int, default=12000)
    ap.add_argument("--test-maps", type=int, default=1500)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--dagger-rounds", type=int, default=2)
    ap.add_argument("--dagger-maps", type=int, default=2500)
    ap.add_argument("--dagger-states", type=int, default=40000)
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--h1", type=int, default=H1)
    ap.add_argument("--h2", type=int, default=H2)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--out", default="src/tiny-vla/weights")
    args = ap.parse_args()
    H1, H2 = args.h1, args.h2
    if args.quick:
        args.maps = min(args.maps, 2500)
        args.test_maps = min(args.test_maps, 400)
        args.epochs = min(args.epochs, 8)
        args.dagger_rounds = 0

    t0 = time.time()
    print(f"generating {args.maps} train maps ...", flush=True)
    data, maps = gen_dataset(args.maps, seed=args.seed)
    print(f"train samples {len(data['label'])}, {time.time()-t0:.1f}s", flush=True)
    print(f"generating {args.test_maps} test maps ...", flush=True)
    _, test_maps = gen_dataset(args.test_maps, seed=args.seed + 99991, verbose=False)

    val_map_cut = int(args.maps * 0.9)
    val_mask = data["map_id"] >= val_map_cut
    tr = {k: data[k][~val_mask] for k in data}
    val = {k: data[k][val_mask] for k in data}
    print(f"train {len(tr['label'])}  val {len(val['label'])}", flush=True)

    model = Model(seed=args.seed)
    model = train(model, tr, args.epochs, args.batch, lr=args.lr,
                  seed=args.seed, val=val)
    stats = evaluate(model, test_maps, seed=args.seed + 5)
    print("eval after BC:", json.dumps(stats), flush=True)
    print(f"colour grounding accuracy: {color_accuracy(model, val)*100:.2f}%", flush=True)

    for r in range(args.dagger_rounds):
        print(f"DAgger round {r+1}/{args.dagger_rounds} ...", flush=True)
        _, dmaps = gen_dataset(args.dagger_maps, seed=args.seed + 1000 + r * 17, verbose=False)
        extra = dagger_collect(model, dmaps, np.random.default_rng(args.seed + 2000 + r),
                               max_states=args.dagger_states)
        if extra is None:
            print("  no DAgger samples", flush=True)
            break
        print(f"  collected {len(extra['label'])} samples", flush=True)
        keys = ["wall", "goal", "dist", "pos", "bow", "label", "map_id", "target"]
        merged = {k: np.concatenate([tr[k], extra[k]]) for k in keys}
        model = train(model, merged, max(6, args.epochs // 3), args.batch,
                      lr=args.lr * 0.5, seed=args.seed + 100 + r, val=val)
        stats = evaluate(model, test_maps, seed=args.seed + 5)
        print(f"eval after DAgger {r+1}:", json.dumps(stats), flush=True)

    os.makedirs(args.out, exist_ok=True)
    np.savez(os.path.join(args.out, "vla_model.npz"), **model.p)
    export_binary(model, os.path.join(args.out, "vla_weights.bin"))
    export_test_vectors(model, val, os.path.join(args.out, "vla_test.bin"))
    export_vocab_rs(os.path.join(args.out, "vocab.rs"))
    with open(os.path.join(args.out, "eval.json"), "w") as f:
        json.dump(stats, f, indent=2)
    print("DONE", json.dumps(stats), flush=True)


if __name__ == "__main__":
    main()
