from __future__ import annotations

import csv
import datetime as dt
import hashlib
import importlib.metadata as md
import json
import subprocess
from pathlib import Path

import numpy as np


def now_iso():
    return dt.datetime.now().astimezone().isoformat(timespec="seconds")


def sha256_file(path: Path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def package_version(name):
    try:
        return md.version(name)
    except Exception:
        return "not-installed/unknown"


def run_cmd(cmd):
    try:
        p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                           text=True, check=False)
        return p.stdout.strip()
    except Exception as exc:
        return f"[command failed: {exc}]"


def write_json(path: Path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, allow_nan=True), encoding="utf-8")


def read_json(path: Path, default=None):
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default


def write_rows(path: Path, rows, fieldnames=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = list(rows)
    if fieldnames is None:
        fieldnames = []
        for r in rows:
            for k in r:
                if k not in fieldnames:
                    fieldnames.append(k)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def read_rows(path: Path):
    if not path.exists():
        return []
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def as_float(x, default=float("nan")):
    try:
        return float(x)
    except Exception:
        return default


def safe_mean(xs):
    a = np.asarray(list(xs), dtype=float)
    if a.size == 0:
        return float("nan")
    return float(np.nanmean(a))


def safe_std(xs):
    a = np.asarray(list(xs), dtype=float)
    if a.size <= 1:
        return 0.0
    return float(np.nanstd(a, ddof=1))


def entropy_from_probs(p):
    p = np.asarray(p, dtype=float)
    p = np.clip(p, 1e-12, 1.0)
    p = p / p.sum(axis=-1, keepdims=True)
    return -np.sum(p * np.log(p), axis=-1)


def autocorr_time(x, max_lag=None):
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    n = x.size
    if n < 8:
        return 0.5
    x = x - x.mean()
    var = np.dot(x, x) / n
    if var <= 1e-18:
        return 0.5
    if max_lag is None:
        max_lag = min(n // 2, 200)
    tau = 0.5
    for lag in range(1, max_lag + 1):
        c = np.dot(x[:-lag], x[lag:]) / (n - lag) / var
        if c <= 0:
            break
        tau += c
        if lag > 6.0 * tau:
            break
    return float(max(0.5, tau))


def bootstrap_group_metric(y_true, y_pred, groups, metric_fn, n_boot=300, seed=1234):
    rng = np.random.default_rng(seed)
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    groups = np.asarray(groups)
    uniq = np.unique(groups)
    vals = []
    for _ in range(n_boot):
        gs = rng.choice(uniq, size=len(uniq), replace=True)
        idx = np.concatenate([np.where(groups == g)[0] for g in gs])
        try:
            vals.append(metric_fn(y_true[idx], y_pred[idx]))
        except Exception:
            pass
    if not vals:
        return (float("nan"), float("nan"))
    return (float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5)))
