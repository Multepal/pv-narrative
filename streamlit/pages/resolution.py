"""
Choosing n_chunks by the stability of the k* partition across resolutions.

n_chunks is a choice of resolution, and cross-edition agreement can't optimize it (its chance
baseline shifts with n). Stability can: at each n, the k* segmentation of each edition is
compared token by token with its segmentations at nearby resolutions (n ± a relative
neighborhood). Below some n, small changes of resolution reorganize the partition; above it,
they don't. The rule is to take the onset of that stability plateau — the coarsest stable
resolution, since larger chunks give more reliable vocabulary profiles. The procedure compares
the model only with itself and never uses the editors (stability-based model selection:
Ben-Hur, Elisseeff & Guyon 2002; Lange, Roth, Braun & Buhmann 2004).
"""

import os
import yaml
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.metrics import adjusted_rand_score
from grid_search_boundary_core import cluster_labels
from utils import find_token_file, load_tokens

APP_DIR = os.path.dirname(os.path.abspath(__file__))

with open(os.path.join(APP_DIR, "../config.yaml"), encoding="utf-8") as _f:
    _cfg = yaml.safe_load(_f)

RES = _cfg["controls"]["resolution"]


@st.cache_data(show_spinner="Measuring stability across resolutions…")
def resolution_stability(method, k, min_df, max_df, ngram_range=(1, 1)) -> pd.DataFrame:
    """Per n in [n_min, n_max]: mean and minimum (over editions) token-level ARI between the
    k-cluster partition at n and those at n ± neighborhood·n, plus a smoothed mean."""
    n_lo, n_hi, rel = RES["n_min"], RES["n_max"], RES["neighborhood"]
    reach = max(1, round(rel * n_hi))
    labels = {}
    for src_id in _cfg["sources"]:
        tp = find_token_file(src_id)
        n_tok = len(load_tokens(src_id, tp))
        pos = np.arange(0, n_tok, RES["token_step"])
        for n in range(max(3 * k, n_lo - reach), n_hi + reach + 1):
            la = cluster_labels(method, src_id, tp, n, min_df, max_df, k, ngram_range)
            if la is not None:
                labels[(src_id, n)] = np.asarray(la)[np.minimum((pos * n) // n_tok, n - 1)]
    rows = []
    for n in range(n_lo, n_hi + 1):
        r = max(1, round(rel * n))
        per_ed = [np.mean([adjusted_rand_score(labels[(s, n)], labels[(s, m)])
                           for m in range(n - r, n + r + 1) if m != n and (s, m) in labels])
                  for s in _cfg["sources"] if (s, n) in labels]
        rows.append({"n_chunks": n, "neighborhood": r, "stability": float(np.mean(per_ed)),
                     "stability_min": float(np.min(per_ed))})
    df = pd.DataFrame(rows)
    df["smoothed"] = df["stability"].rolling(RES["smooth"], center=True, min_periods=1).mean()
    return df


def stability_plateau(df: pd.DataFrame) -> tuple[int, int]:
    """(onset, end) of the contiguous run of n containing the smoothed maximum whose smoothed
    stability is within plateau_tol of that maximum. The onset is the recommended n_chunks."""
    sm = df["smoothed"].to_numpy()
    i_max = int(np.argmax(sm))
    ok = sm >= sm[i_max] - RES["plateau_tol"]
    lo = i_max
    while lo > 0 and ok[lo - 1]:
        lo -= 1
    hi = i_max
    while hi < len(sm) - 1 and ok[hi + 1]:
        hi += 1
    return int(df["n_chunks"].iloc[lo]), int(df["n_chunks"].iloc[hi])
