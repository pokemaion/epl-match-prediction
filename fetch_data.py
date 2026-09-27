"""
fetch_data.py — grab free Premier League results.

    python fetch_data.py            # results only (no odds)
    python fetch_data.py --odds     # results + Bet365 odds, needed for the betting side

Then:
    python elo.py --data data
"""

import argparse
import os
import urllib.request

GITHUB = ("https://raw.githubusercontent.com/datasets/football-datasets/"
          "master/datasets/premier-league/season-{s}.csv")

# football-data.co.uk ships the same matches PLUS bookmaker odds columns
# (B365H / B365D / B365A). That's what you need to hunt for edges.
FOOTBALL_DATA = "https://www.football-data.co.uk/mmz4281/{s}/E0.csv"

SEASONS = ["1516", "1617", "1718", "1819", "1920",
           "2021", "2122", "2223", "2324", "2425"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--odds", action="store_true", help="use football-data.co.uk (has odds)")
    ap.add_argument("--out", default="data")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    src = FOOTBALL_DATA if args.odds else GITHUB

    for s in SEASONS:
        url = src.format(s=s)
        dest = os.path.join(args.out, f"epl-{s}.csv")
        try:
            urllib.request.urlretrieve(url, dest)
            print(f"  ok  {s}  {os.path.getsize(dest):,} bytes")
        except Exception as e:
            print(f"  --  {s}  {e}")

    print(f"\nSaved to {args.out}/  ->  python elo.py --data {args.out}")


if __name__ == "__main__":
    main()
