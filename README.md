# Elo Match Prediction — Premier League

An Elo rating system for football that outputs calibrated win/draw/loss probabilities, plus the tooling to check whether those probabilities are actually true and whether they beat a bookmaker.

Built on 3,800 Premier League matches, 2015–2025.

---

## Results

All figures are out-of-sample. The test set was used once, at the end.

| Model | Log loss | Brier |
| --- | --- | --- |
| Base rate (predict the long-run average every match) | 1.062 | — |
| v1 — Elo on goals | 0.9663 | 0.5743 |
| **v2 — Elo on goals + shots on target** | **0.9642** | **0.5730** |

Log loss punishes confident wrong answers. The base-rate row is the bar: clearing it means the ratings carry real information.

### Calibration (v2)

The model's probabilities, bucketed and checked against what actually happened:

| Predicted | Actual | n | z |
| --- | --- | --- | --- |
| 0.147 | 0.137 | 190 | −0.39 |
| 0.262 | 0.242 | 190 | −0.63 |
| 0.339 | 0.411 | 190 | +2.08 |
| 0.405 | 0.358 | 190 | −1.31 |
| 0.474 | 0.495 | 190 | +0.58 |
| 0.550 | 0.542 | 190 | −0.21 |
| 0.649 | 0.611 | 190 | −1.12 |
| 0.785 | 0.768 | 190 | −0.55 |

|z| under about 2 means a miss is within normal sampling noise. Seven of eight buckets are. One bucket at +2.08 is roughly what chance alone produces across eight buckets, so it isn't treated as a real effect.

---

## The false alarm, and what it taught

**Symptom.** An early run showed v1's biggest favourites called at 75.5% but winning only 66.8% (z = −2.79), which looked like real overconfidence.

**Cause.** A date-parsing error. The loader read dates day-first, so ISO-formatted dates like `2015-11-01` were read as 11 January. Any date where both numbers were 12 or under got flipped, which partially scrambled match order. Elo updates in time order, so the ratings were learning from a shuffled history.

**After fixing it:**

| Model | n | Predicted | Actual | Error | z |
| --- | --- | --- | --- | --- | --- |
| v1, scrambled dates | 190 | 0.755 | 0.668 | −0.087 | −2.79 |
| v1, correct dates | 190 | 0.786 | 0.737 | −0.049 | −1.65 |
| v2, correct dates | 190 | 0.785 | 0.768 | −0.016 | −0.55 |

With correct dates, v1's miss on favourites is inside sampling noise. Most of the apparent flaw was the data bug, not the model.

**Shots on target still help, modestly.** Goals are a small, noisy sample of a match: a lucky 1–0 counts the same as a dominant one. Blending shots on target into rating updates gives each match more signal. Out of sample, this improved log loss from 0.9663 to 0.9642 and shrank the favourite gap further. The gain is real but small.

**Lesson.** A clean causal story ("goals are noisy, so favourites get overrated") fit the symptom well, and was partly explaining a data bug. The check that caught it was boring: football seasons don't start in January.

---

## Methodology

**Three-way split: 45% train / 15% validation / 40% test.** Ratings and the probability layer are fit on train. The shots blend weight is selected on validation. The test set is read once, after every decision is locked. Validation chose a blend weight of 0.4 (40% shots, 60% goals).

**No look-ahead.** `walk()` records each prediction from the ratings as they stood *before* the match, then updates. Reversing those two steps would let the model see each result before predicting it.

**Burn-in discarded.** Every team starts at 1500, so the earliest matches are the model guessing. Those are excluded from fitting and scoring.

**Season regression.** Ratings are pulled 25% toward the mean each summer, because squads change.

---

## What's in here

```
fetch_data.py   downloads 10 seasons of results (--odds for bookmaker lines)
elo.py          v1: the baseline, plus the betting tooling
elo2.py         v2: shots blend, three-way split, before/after comparison
```

Core components:

- `Elo`: ratings with home advantage, log-scaled margin of victory damped for favourites, season regression, and an optional shots channel
- `fit_ordered_logit`: maps one rating gap to three probabilities. Two cutpoints split the number line; the zone between them is the draw.
- `walk`: walk-forward evaluation, predict then update
- `calibration`: bucketed predicted vs actual, with z-scores to separate real misses from noise
- `devig`: strips the bookmaker's margin from decimal odds to recover their implied probability
- `kelly`: quarter-Kelly stake sizing
- `find_edges`: matches where the model disagrees with the bookmaker by more than 3 points, and whether those bets won

---

## Run it

```
pip install pandas numpy scipy
python fetch_data.py --odds
python elo2.py --data data
```

`--odds` pulls from football-data.co.uk, which includes Bet365 lines. Without it you get results only, and the betting section is skipped.

---

## Known limitations

The top bucket is still about 1.7 points short. The model has no knowledge of:

- injuries and suspensions
- rotation ahead of European fixtures
- managerial changes
- teams already safe in May with nothing to play for

Each is a candidate feature, and each is another chance to fool yourself. Same rule every time: select it on validation, or it isn't real.

**On the betting side:** beating a bookmaker's closing line consistently is hard, and this model has not been shown to do it. `find_edges` exists to test that claim honestly, not to assume it.

---

## Next

- [ ] Measure closing line value, the real test of whether the edges are edges
- [ ] Attack/defence ratings (Dixon–Coles Poisson) instead of a single strength number
- [ ] Lineup strength from starting XI data
- [ ] Extend to other leagues and check whether the blend weight transfers
