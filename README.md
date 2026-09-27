# Elo Match Prediction — Premier League

An Elo rating system for football that outputs calibrated win/draw/loss probabilities, plus the tooling to check whether those probabilities are actually true and whether they beat a bookmaker.

Built on 3,800 Premier League matches, 2015–2025.

The interesting result isn't the model. It's that v1 had a measurable flaw, the flaw was diagnosed, and the fix was validated in a way that couldn't fool me.

---

## Results

All figures out-of-sample. The test set was used once, at the end.

| Model | Log loss | Brier |
|---|---|---|
| Base rate (predict the long-run average every match) | 1.0616 | — |
| v1 — Elo on goals | 1.0100 | 0.6006 |
| **v2 — Elo on goals + shots on target** | **0.9959** | **0.5918** |

Log loss punishes confident wrong answers. The base-rate row is the bar: clear it and the ratings carry real information.

### Calibration (v2)

The model's probabilities are checked against reality, bucketed:

| Predicted | Actual | n | z |
|---|---|---|---|
| 0.184 | 0.147 | 190 | −1.31 |
| 0.309 | 0.305 | 190 | −0.10 |
| 0.374 | 0.321 | 190 | −1.52 |
| 0.429 | 0.395 | 190 | −0.96 |
| 0.480 | 0.521 | 190 | +1.14 |
| 0.537 | 0.553 | 190 | +0.42 |
| 0.610 | 0.605 | 190 | −0.13 |
| 0.753 | 0.711 | 190 | −1.37 |

Every bucket sits inside |z| = 2, i.e. every miss is within sampling noise. Matches called 60% happen 60% of the time. That's what makes the output usable for anything downstream.

---

## The bug, and how it was fixed

**Symptom.** v1's biggest favourites were called at 75.5% and won 66.8%.

**Was it real, or 190 matches of bad luck?** Checked against the training set:

```
train:  predicted 0.710  actual 0.719   z = +0.30
test:   predicted 0.755  actual 0.668   z = −2.79
```

Real. Also absent in-sample — which rules out fixing it by refitting on training data.

**Cause.** Goals are a very small sample of a football match. A team winning 1–0 off a deflection earns the same rating bump as one winning 1–0 while taking twenty shots. Elo over-credits lucky winners; lucky winners regress; the damage concentrates where the model is most confident.

**Fix.** Blend shots on target into the rating update alongside goals. Shots are a larger sample of the same underlying performance, so each match carries less noise.

**Result.**

| | predicted | actual | error | z |
|---|---|---|---|---|
| v1 | 0.755 | 0.668 | −0.087 | −2.79 |
| v2 | 0.753 | 0.711 | −0.043 | −1.37 |

Error roughly halved; the residual miss is no longer distinguishable from noise.

---

## Methodology

This is the part that mattered most.

**Three-way split — 45% train / 15% validation / 40% test.**
Ratings and the probability layer are fit on train. The shots blend weight is selected on **validation**. The test set is read once, at the end, after every decision is locked.

Selecting the blend weight against test scores would have produced a better-looking number and no real improvement — it would just be memorising the test set. Validation independently chose 0.7, having never seen test data.

**No look-ahead.** `walk()` records a prediction from the ratings as they stood *before* each match, then updates. Reversing those two lines produces a model that looks brilliant and predicts nothing.

**Burn-in discarded.** Every team starts at 1500, so the first ~20% of matches are the model guessing. Those are excluded from all fitting and scoring.

**Season regression.** Ratings are pulled 25% toward the mean each summer. Squads change.

---

## What's in here

```
fetch_data.py   downloads 10 seasons of results (--odds for bookmaker lines)
elo.py          v1 — the baseline, plus the betting tooling
elo2.py         v2 — the shots fix, three-way split, before/after comparison
```

Core components:

- `Elo` — ratings with home advantage, log-scaled margin of victory damped for favourites, season regression, and an optional shots channel
- `fit_ordered_logit` — maps one rating gap to three probabilities. Two cutpoints carve the number line; the gap between them is the draw zone.
- `walk` — walk-forward evaluation, predict-then-update
- `calibration` — bucketed predicted vs actual, with z-scores so you can tell a real miss from noise
- `devig` — strips the bookmaker's margin from decimal odds to recover their honest probability
- `kelly` — quarter-Kelly stake sizing
- `find_edges` — where the model disagrees with the book by >3%, and whether those bets cashed

---

## Run it

```bash
pip install pandas numpy scipy
python fetch_data.py --odds
python elo2.py --data data
```

`--odds` pulls from football-data.co.uk, which ships the same matches plus Bet365 lines. Without it you get results only, and the betting section stays dark.

---

## Known limitations

The top bucket is still ~4 points short. The model has no knowledge of:

- injuries and suspensions
- rotation ahead of European fixtures
- managerial changes
- teams already safe in May with nothing to play for

Each is a candidate feature. Each is also another chance to fool yourself, so the rule holds: selected on validation, or it isn't real.

**On the betting side:** a standard −110 line requires winning 52.4% just to break even, and the closing line is close to the sharpest available estimate of a match, having absorbed every professional's money. Beating it consistently is hard, and this model has not been shown to do it. The `find_edges` output is there to test that claim honestly, not to assume it.

---

## Next

- [ ] Measure closing line value — the real test of whether the edges are edges
- [ ] Attack/defence ratings (Dixon–Coles Poisson) instead of a single strength number
- [ ] Lineup strength from starting XI data
- [ ] Extend to other leagues; check whether the blend weight transfers or needs refitting
