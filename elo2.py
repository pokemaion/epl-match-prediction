"""
elo2.py — the fix for overconfidence on favourites.

DIAGNOSIS
  v1 said 75% on its biggest favourites; they won 67%. Two things had to be
  checked before writing a line of code:

  1. Is it real, or is it 190 matches of noise?
       train:  predicted 0.710, actual 0.719   (z = +0.30)  <- fine
       test:   predicted 0.752, actual 0.668   (z = -2.67)  <- not fine
     Real enough to chase, but note it does NOT appear in-sample. You cannot
     fix a bug by fitting on data where the bug doesn't exist.

  2. What causes it?
     Goals are a tiny, noisy sample of a football match. A team that wins 1-0
     off one lucky deflection gets the same rating bump as one that wins 1-0
     while taking twenty shots. Elo therefore over-credits lucky winners, and
     lucky winners regress — hardest at the top, where the model is most sure.

FIX
  Rate teams on shots on target as well as goals. Shots are a bigger sample of
  the same underlying thing, so ratings get less noise per match.

METHOD (this part matters more than the fix)
  Three-way split: train / validation / test.
    - ratings + probability layer fitted on TRAIN
    - the blend weight chosen on VALIDATION
    - TEST touched exactly once, at the end
  If you pick the weight by looking at test scores, you have not improved the
  model. You have memorised the test set. That is the mistake this whole file
  is arranged to prevent.

RESULT (out of sample, test set never used for any decision)
    log loss        1.0100  ->  0.9959
    top bucket      0.755 predicted / 0.668 actual   (error -0.087)
                ->  0.753 predicted / 0.711 actual   (error -0.042)
  Error roughly halved. Not eliminated — see 'still broken' at the bottom.

Run:
    python fetch_data.py
    python elo2.py --data data
"""

from __future__ import annotations

import argparse
import glob
import os

import numpy as np
import pandas as pd
from scipy.optimize import minimize


# ----------------------------------------------------------------------------
# DATA
# ----------------------------------------------------------------------------

def load(folder: str) -> pd.DataFrame:
    files = sorted(glob.glob(os.path.join(folder, "*.csv")))
    if not files:
        raise SystemExit(f"No CSVs found in {folder}/ — run fetch_data.py first")

    frames = []
    for f in files:
        d = pd.read_csv(f)
        d["season"] = os.path.splitext(os.path.basename(f))[0]
        frames.append(d)

    df = pd.concat(frames, ignore_index=True)
    df["Date"] = pd.to_datetime(df["Date"], dayfirst=True, format="mixed")
    df = df.sort_values("Date").reset_index(drop=True)

    need = ["Date", "season", "HomeTeam", "AwayTeam", "FTHG", "FTAG"]
    extra = [c for c in ("HST", "AST", "B365H", "B365D", "B365A") if c in df.columns]
    if "HST" not in extra:
        print("! No shots-on-target columns (HST/AST) — the fix needs them.")
    return df[need + extra].dropna(subset=["FTHG", "FTAG"])


# ----------------------------------------------------------------------------
# ELO, NOW WITH A SHOTS CHANNEL
# ----------------------------------------------------------------------------

class Elo:
    """
    shot_w = 0.0  -> pure goals (v1 behaviour)
    shot_w = 1.0  -> pure shots on target
    In between blends both. 0.7 won on validation.
    """

    def __init__(self, k=20.0, home_adv=65.0, base=1500.0,
                 regress=0.25, shot_w=0.0):
        self.k, self.home_adv, self.base = k, home_adv, base
        self.regress, self.shot_w = regress, shot_w
        self.r: dict[str, float] = {}

    def rating(self, t: str) -> float:
        return self.r.setdefault(t, self.base)

    def diff(self, home: str, away: str) -> float:
        return self.rating(home) + self.home_adv - self.rating(away)

    def new_season(self) -> None:
        for t in self.r:
            self.r[t] = self.base + (self.r[t] - self.base) * (1 - self.regress)

    def update(self, home, away, hg, ag, hst=np.nan, ast=np.nan) -> None:
        d = self.diff(home, away)
        exp = 1.0 / (1.0 + 10 ** (-d / 400))

        actual = 1.0 if hg > ag else (0.5 if hg == ag else 0.0)
        margin = float(abs(hg - ag))

        # Blend in the shots verdict. Shots margins are capped at 6 so one
        # freak 18-shot game can't detonate a rating.
        if self.shot_w > 0 and not (np.isnan(hst) or np.isnan(ast)):
            s_actual = 1.0 if hst > ast else (0.5 if hst == ast else 0.0)
            actual = (1 - self.shot_w) * actual + self.shot_w * s_actual
            margin = (1 - self.shot_w) * margin + \
                     self.shot_w * min(abs(hst - ast), 6)

        mult = np.log(margin + 1) * (2.2 / (0.001 * (d if actual > 0.5 else -d) + 2.2))
        delta = self.k * mult * (actual - exp)
        self.r[home] += delta
        self.r[away] -= delta


# ----------------------------------------------------------------------------
# RATING GAP -> THREE PROBABILITIES
# ----------------------------------------------------------------------------

def probs(diff, beta, c1, c2):
    z = beta * np.asarray(diff) / 100.0
    p_away = 1 / (1 + np.exp(z - c1))
    p_not_home = 1 / (1 + np.exp(z - c2))
    return np.column_stack([p_away, np.clip(p_not_home - p_away, 1e-9, 1), 1 - p_not_home])


def fit_ordered_logit(diff, outcome):
    def nll(p):
        beta, c1, gap = p
        pr = probs(diff, beta, c1, c1 + abs(gap))
        return -np.log(pr[np.arange(len(outcome)), outcome] + 1e-12).sum()

    r = minimize(nll, [1.0, -1.0, 1.0], method="Nelder-Mead",
                 options={"maxiter": 5000, "xatol": 1e-6, "fatol": 1e-6})
    beta, c1, gap = r.x
    return float(beta), float(c1), float(c1 + abs(gap))


# ----------------------------------------------------------------------------
# WALK FORWARD — predict, then update
# ----------------------------------------------------------------------------

def walk(df, shot_w, k=20.0, home_adv=65.0, regress=0.25, params=None):
    elo = Elo(k, home_adv, regress=regress, shot_w=shot_w)
    diffs, outs, season = [], [], None

    for m in df.itertuples():
        if season is not None and m.season != season:
            elo.new_season()
        season = m.season

        diffs.append(elo.diff(m.HomeTeam, m.AwayTeam))
        outs.append(2 if m.FTHG > m.FTAG else (1 if m.FTHG == m.FTAG else 0))

        elo.update(m.HomeTeam, m.AwayTeam, m.FTHG, m.FTAG,
                   getattr(m, "HST", np.nan), getattr(m, "AST", np.nan))

    diffs, outs = np.array(diffs), np.array(outs)
    if params is None:
        return diffs, outs, elo.r
    return probs(diffs, *params), outs, elo.r


# ----------------------------------------------------------------------------
# SCORING
# ----------------------------------------------------------------------------

def log_loss(p, o):
    return float(-np.log(p[np.arange(len(o)), o] + 1e-12).mean())


def brier(p, o):
    y = np.zeros_like(p)
    y[np.arange(len(o)), o] = 1
    return float(((p - y) ** 2).sum(axis=1).mean())


def calibration(p, o, bins=8):
    d = pd.DataFrame({"p": p[:, 2], "hit": (o == 2).astype(int)})
    d["bucket"] = pd.qcut(d["p"], bins, duplicates="drop")
    g = d.groupby("bucket", observed=True).agg(
        n=("hit", "size"), predicted=("p", "mean"), actual=("hit", "mean"))
    g["error"] = g["actual"] - g["predicted"]
    # z-score: is this bucket's miss bigger than chance would produce?
    g["z"] = g["error"] / np.sqrt(g["predicted"] * (1 - g["predicted"]) / g["n"])
    return g.round(3)


def favourite_check(p, o, q=0.875):
    """Zoom in on the top 12.5% of home favourites — where v1 broke."""
    ph, hit = p[:, 2], (o == 2).astype(int)
    m = ph >= np.quantile(ph, q)
    pred, act, n = ph[m].mean(), hit[m].mean(), int(m.sum())
    se = np.sqrt(pred * (1 - pred) / n)
    return dict(n=n, predicted=pred, actual=act, error=act - pred, z=(act - pred) / se)


# ----------------------------------------------------------------------------
# MAIN
# ----------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data")
    ap.add_argument("--grid", type=str, default="0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8")
    args = ap.parse_args()

    df = load(args.data)
    n = len(df)
    a, b = int(n * 0.45), int(n * 0.60)      # 45% train, 15% val, 40% test
    warm = a // 5                             # ignore burn-in, ratings start blind

    print(f"\n{n} matches, {df['Date'].min().date()} to {df['Date'].max().date()}")
    print(f"train {a}   validation {b-a}   test {n-b}\n")

    # ---- choose the blend weight on VALIDATION -----------------------------
    print("--- selecting shot weight on validation (test not touched) ---")
    grid = [float(x) for x in args.grid.split(",")]
    scores = {}
    for sw in grid:
        d, o, _ = walk(df.iloc[:a], sw)
        par = fit_ordered_logit(d[warm:], o[warm:])
        p, ov, _ = walk(df.iloc[:b], sw, params=par)
        scores[sw] = log_loss(p[a:], ov[a:])
        print(f"  shot_w={sw:.1f}   validation log loss {scores[sw]:.4f}")

    best_sw = min(scores, key=scores.get)
    print(f"\n  chosen: shot_w = {best_sw}\n")

    # ---- refit on train+validation, evaluate once on TEST -------------------
    results = {}
    for label, sw in (("v1 goals only", 0.0), (f"v2 shots blend {best_sw}", best_sw)):
        d, o, _ = walk(df.iloc[:b], sw)
        par = fit_ordered_logit(d[warm:], o[warm:])
        p, oa, ratings = walk(df, sw, params=par)
        results[label] = (p[b:], oa[b:], ratings, par)

    print("--- TEST SET (used once, right here) ---")
    print(f"{'model':<26}{'log loss':>10}{'Brier':>10}")
    for label, (p, o, _, _) in results.items():
        print(f"{label:<26}{log_loss(p, o):>10.4f}{brier(p, o):>10.4f}")

    print(f"\n--- THE BUG: top 12.5% of home favourites ---")
    print(f"{'model':<26}{'n':>5}{'predicted':>11}{'actual':>9}{'error':>9}{'z':>7}")
    for label, (p, o, _, _) in results.items():
        f = favourite_check(p, o)
        print(f"{label:<26}{f['n']:>5}{f['predicted']:>11.3f}{f['actual']:>9.3f}"
              f"{f['error']:>+9.3f}{f['z']:>+7.2f}")

    label_v2 = f"v2 shots blend {best_sw}"
    p2, o2, ratings2, _ = results[label_v2]

    print(f"\n--- FULL CALIBRATION, v2 ---")
    print("|z| under about 2 means the miss is within normal sampling noise")
    print(calibration(p2, o2).to_string())

    print(f"\n--- RATINGS, v2 (top 10) ---")
    print(pd.Series(ratings2).sort_values(ascending=False).head(10).round(0).to_string())

    print("""
--- STILL BROKEN ---
The top bucket is still short by about 4 points. Shots are a better sample of
performance than goals, but the model still has no idea about:
  injuries and suspensions      squad rotation before a European tie
  managerial changes            a team already safe in May with nothing to play for
Each of those is a feature you could add, and each is another chance to fool
yourself. Same rule every time: choose it on validation, or it isn't real.
""")


if __name__ == "__main__":
    main()
