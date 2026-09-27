"""
elo.py — an Elo rating system for football, and an honest evaluation of it.

What this does:
  1. Rates teams with Elo, updated match by match, never peeking at the future.
  2. Converts a rating gap into three real probabilities: home / draw / away.
  3. Checks whether those probabilities are actually TRUE (calibration).
  4. Compares your number to a bookmaker's number and sizes a bet if there's edge.

The model is the easy half. Steps 3 and 4 are the half that matters.

Data:
  Free Premier League CSVs, no API key:
    https://raw.githubusercontent.com/datasets/football-datasets/master/datasets/premier-league/season-2425.csv
  For real bookmaker odds, use football-data.co.uk's own files instead — same
  columns plus B365H / B365D / B365A. This script auto-detects those.

Run:
    python elo.py --data data/
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
        raise SystemExit(f"No CSVs in {folder}")

    frames = []
    for f in files:
        d = pd.read_csv(f)
        d["season"] = os.path.splitext(os.path.basename(f))[0]
        frames.append(d)

    df = pd.concat(frames, ignore_index=True)
    df["Date"] = pd.to_datetime(df["Date"], dayfirst=True, format="mixed")
    df = df.sort_values("Date").reset_index(drop=True)

    keep = ["Date", "season", "HomeTeam", "AwayTeam", "FTHG", "FTAG", "FTR"]
    odds = [c for c in ("B365H", "B365D", "B365A") if c in df.columns]
    return df[keep + odds].dropna(subset=["FTHG", "FTAG"])


# ----------------------------------------------------------------------------
# ELO
# ----------------------------------------------------------------------------

class Elo:
    """
    Standard Elo with two football-specific adjustments.

    home_adv : rating points added to the home side before comparing.
               The model doesn't 'know' home advantage exists unless you tell it.
    mov      : bigger wins move ratings more, with diminishing returns, and
               damped for favourites so a 5-0 stroll doesn't overreact.
    """

    def __init__(self, k: float = 20, home_adv: float = 65,
                 base: float = 1500, regress: float = 0.25):
        self.k, self.home_adv, self.base, self.regress = k, home_adv, base, regress
        self.r: dict[str, float] = {}

    def rating(self, team: str) -> float:
        return self.r.setdefault(team, self.base)

    def expected(self, home: str, away: str) -> float:
        """Expected score for the home team, on a 0-1 scale (draw counts 0.5)."""
        d = self.rating(home) + self.home_adv - self.rating(away)
        return 1.0 / (1.0 + 10 ** (-d / 400))

    def new_season(self) -> None:
        """Pull every team partway back to average. Squads change over summer."""
        for t in self.r:
            self.r[t] = self.base + (self.r[t] - self.base) * (1 - self.regress)

    def update(self, home: str, away: str, hg: int, ag: int) -> None:
        exp = self.expected(home, away)
        actual = 1.0 if hg > ag else (0.5 if hg == ag else 0.0)

        margin = abs(hg - ag)
        d = self.rating(home) + self.home_adv - self.rating(away)
        # FiveThirtyEight's multiplier: log-scaled margin, damped for favourites.
        mult = np.log(margin + 1) * (2.2 / (0.001 * (d if actual == 1 else -d) + 2.2))

        delta = self.k * mult * (actual - exp)
        self.r[home] += delta
        self.r[away] -= delta


# ----------------------------------------------------------------------------
# THREE-OUTCOME PROBABILITIES
#
# Elo gives one number. Football has three results. An ordered logit turns the
# rating gap into P(home) / P(draw) / P(away) by carving the line at two
# cutpoints — the gap between them IS the draw zone.
# ----------------------------------------------------------------------------

def _probs(diff: np.ndarray, beta: float, c1: float, c2: float):
    z = beta * diff / 100.0
    p_away = 1 / (1 + np.exp(z - c1))          # P(result <= away)
    p_not_home = 1 / (1 + np.exp(z - c2))      # P(result <= draw)
    return np.column_stack([p_away, np.clip(p_not_home - p_away, 1e-9, 1), 1 - p_not_home])


def fit_ordered_logit(diff: np.ndarray, outcome: np.ndarray):
    """outcome: 0 = away win, 1 = draw, 2 = home win."""

    def nll(p):
        beta, c1, gap = p
        pr = _probs(diff, beta, c1, c1 + abs(gap))
        return -np.log(pr[np.arange(len(outcome)), outcome] + 1e-12).sum()

    res = minimize(nll, [1.0, -1.0, 1.0], method="Nelder-Mead",
                   options={"maxiter": 5000, "xatol": 1e-6, "fatol": 1e-6})
    beta, c1, gap = res.x
    return float(beta), float(c1), float(c1 + abs(gap))


# ----------------------------------------------------------------------------
# WALK-FORWARD BACKTEST
# Predict, THEN update. Never the other way round.
# ----------------------------------------------------------------------------

def walk_forward(df: pd.DataFrame, k: float, home_adv: float,
                 regress: float, params: tuple) -> pd.DataFrame:
    elo = Elo(k=k, home_adv=home_adv, regress=regress)
    beta, c1, c2 = params
    rows, season = [], None

    for m in df.itertuples():
        if season is not None and m.season != season:
            elo.new_season()
        season = m.season

        diff = elo.rating(m.HomeTeam) + elo.home_adv - elo.rating(m.AwayTeam)
        p = _probs(np.array([diff]), beta, c1, c2)[0]

        rows.append({
            "Date": m.Date, "season": m.season,
            "home": m.HomeTeam, "away": m.AwayTeam,
            "elo_diff": diff,
            "p_away": p[0], "p_draw": p[1], "p_home": p[2],
            "outcome": 2 if m.FTHG > m.FTAG else (1 if m.FTHG == m.FTAG else 0),
            **{c: getattr(m, c) for c in ("B365H", "B365D", "B365A") if hasattr(m, c)},
        })

        elo.update(m.HomeTeam, m.AwayTeam, m.FTHG, m.FTAG)

    out = pd.DataFrame(rows)
    out.attrs["final_ratings"] = elo.r
    return out


def collect_diffs(df: pd.DataFrame, k: float, home_adv: float, regress: float):
    """One pass just to harvest (rating gap, outcome) pairs for fitting."""
    elo = Elo(k=k, home_adv=home_adv, regress=regress)
    diffs, outs, season = [], [], None
    for m in df.itertuples():
        if season is not None and m.season != season:
            elo.new_season()
        season = m.season
        diffs.append(elo.rating(m.HomeTeam) + elo.home_adv - elo.rating(m.AwayTeam))
        outs.append(2 if m.FTHG > m.FTAG else (1 if m.FTHG == m.FTAG else 0))
        elo.update(m.HomeTeam, m.AwayTeam, m.FTHG, m.FTAG)
    return np.array(diffs), np.array(outs)


# ----------------------------------------------------------------------------
# SCORING
# ----------------------------------------------------------------------------

def log_loss(res: pd.DataFrame) -> float:
    p = res[["p_away", "p_draw", "p_home"]].to_numpy()
    return float(-np.log(p[np.arange(len(res)), res["outcome"]] + 1e-12).mean())


def brier(res: pd.DataFrame) -> float:
    p = res[["p_away", "p_draw", "p_home"]].to_numpy()
    y = np.zeros_like(p)
    y[np.arange(len(res)), res["outcome"]] = 1
    return float(((p - y) ** 2).sum(axis=1).mean())


def baseline_log_loss(res: pd.DataFrame) -> float:
    """Predict the long-run base rate for every match. Beat this or go home."""
    rates = np.bincount(res["outcome"], minlength=3) / len(res)
    return float(-np.log(rates[res["outcome"]]).mean())


def calibration(res: pd.DataFrame, bins: int = 8) -> pd.DataFrame:
    """
    THE key diagnostic. Take every match you called ~60% home win.
    Did the home team win ~60% of them? If not, your numbers are decoration.
    """
    d = pd.DataFrame({"p": res["p_home"], "hit": (res["outcome"] == 2).astype(int)})
    d["bucket"] = pd.qcut(d["p"], bins, duplicates="drop")
    g = d.groupby("bucket", observed=True).agg(
        n=("hit", "size"), predicted=("p", "mean"), actual=("hit", "mean"))
    g["error"] = g["actual"] - g["predicted"]
    return g.round(3)


# ----------------------------------------------------------------------------
# BOOKMAKER ODDS -> EDGE -> STAKE
# ----------------------------------------------------------------------------

def devig(odds: np.ndarray) -> np.ndarray:
    """
    Decimal odds -> true implied probabilities.

    Raw 1/odds across all outcomes sums to MORE than 1. That excess is the vig,
    the book's cut. Scaling it away gives the book's honest opinion — which is
    the number you must actually beat.
    """
    raw = 1.0 / odds
    return raw / raw.sum(axis=-1, keepdims=True)


def kelly(p: float, decimal_odds: float, fraction: float = 0.25) -> float:
    """
    Fraction of bankroll to stake. Full Kelly maximises long-run growth and
    swings brutally, so real bettors use a quarter of it. Negative = no bet.
    """
    b = decimal_odds - 1
    f = (p * b - (1 - p)) / b
    return max(0.0, f) * fraction


def find_edges(res: pd.DataFrame, min_edge: float = 0.03) -> pd.DataFrame:
    """Where does the model disagree with the book by more than min_edge?"""
    cols = ["B365H", "B365D", "B365A"]
    if not all(c in res.columns for c in cols):
        return pd.DataFrame()

    odds = res[cols].to_numpy(float)              # H, D, A
    fair = devig(odds)
    mine = res[["p_home", "p_draw", "p_away"]].to_numpy()
    edge = mine - fair

    out = []
    for i in range(len(res)):
        j = int(np.argmax(edge[i]))
        if edge[i, j] < min_edge:
            continue
        won = res["outcome"].iat[i] == (2, 1, 0)[j]
        stake = kelly(mine[i, j], odds[i, j])
        out.append({
            "Date": res["Date"].iat[i],
            "match": f"{res['home'].iat[i]} v {res['away'].iat[i]}",
            "side": ("Home", "Draw", "Away")[j],
            "my_p": round(mine[i, j], 3),
            "book_p": round(fair[i, j], 3),
            "edge": round(edge[i, j], 3),
            "odds": odds[i, j],
            "stake_pct": round(stake * 100, 2),
            "won": won,
            "profit_units": stake * (odds[i, j] - 1) if won else -stake,
        })
    return pd.DataFrame(out)


# ----------------------------------------------------------------------------
# MAIN
# ----------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data")
    ap.add_argument("--train-frac", type=float, default=0.6)
    ap.add_argument("--tune", action="store_true", help="grid-search K and home advantage")
    args = ap.parse_args()

    df = load(args.data)
    cut = int(len(df) * args.train_frac)
    train, test = df.iloc[:cut], df.iloc[cut:]

    print(f"\n{len(df)} matches, {df['Date'].min().date()} to {df['Date'].max().date()}")
    print(f"train: {len(train)}   test: {len(test)}   (test is NEVER used for fitting)\n")

    # --- tune on train only -------------------------------------------------
    best = {"k": 20, "home_adv": 65, "regress": 0.25}
    if args.tune:
        print("tuning on training set...")
        scores = []
        for k in (10, 15, 20, 25, 30, 40):
            for ha in (40, 55, 65, 80, 95):
                d, o = collect_diffs(train, k, ha, 0.25)
                warm = len(d) // 5                       # ignore burn-in matches
                pars = fit_ordered_logit(d[warm:], o[warm:])
                r = walk_forward(train, k, ha, 0.25, pars)
                scores.append((log_loss(r.iloc[warm:]), k, ha))
        scores.sort()
        _, best["k"], best["home_adv"] = scores[0]
        print(f"best on train: K={best['k']}, home_adv={best['home_adv']}\n")

    # --- fit the probability layer on train, evaluate on test ---------------
    d, o = collect_diffs(train, best["k"], best["home_adv"], best["regress"])
    warm = len(d) // 5
    params = fit_ordered_logit(d[warm:], o[warm:])
    print(f"ordered logit: beta={params[0]:.3f}  cut1={params[1]:.3f}  cut2={params[2]:.3f}")
    print(f"home advantage: {best['home_adv']:.0f} Elo pts   K: {best['k']}\n")

    full = walk_forward(df, best["k"], best["home_adv"], best["regress"], params)
    res_test = full.iloc[cut:]

    print("--- OUT-OF-SAMPLE PERFORMANCE ---")
    print(f"log loss        {log_loss(res_test):.4f}   (lower is better)")
    print(f"base-rate model {baseline_log_loss(res_test):.4f}   <- the bar to clear")
    print(f"Brier score     {brier(res_test):.4f}\n")

    print("--- CALIBRATION (home win) ---")
    print("if 'predicted' and 'actual' track each other, the probabilities are real")
    print(calibration(res_test).to_string(), "\n")

    ratings = pd.Series(full.attrs["final_ratings"]).sort_values(ascending=False)
    print("--- CURRENT RATINGS (top 10) ---")
    print(ratings.head(10).round(0).to_string(), "\n")

    edges = find_edges(res_test)
    if edges.empty:
        print("No bookmaker odds in this dataset — download football-data.co.uk")
        print("files instead (same columns + B365H/B365D/B365A) to run the betting side.")
    else:
        n = len(edges)
        won = edges["won"].sum()
        profit = edges["profit_units"].sum()
        staked = edges["stake_pct"].sum() / 100
        print(f"--- BETS THE MODEL WOULD HAVE MADE ---")
        print(f"{n} bets, {won} won ({won/n:.1%}), {profit:+.2f} units on {staked:.2f} staked")
        print(f"ROI: {profit/staked:+.1%}\n")
        print(edges.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
