
# Dynasty Bros. Trade Calculator — v5

## What’s new
1) **Roster-aware Fairness Verdict** — insights about surpluses/needs and how incoming positions fit.
2) **Clear Modifiers** — renamed with plain-English intent + tooltips.
3) **Trade Finder 2.0** — multiple suggestions (1–3 players + optional pick) and a **Suggest another trade** button to reshuffle.
4) **(Optional) Sleeper Import** — paste a League ID to fetch rosters/users (requires internet on your machine).

## CSVs
- `data/player_ranks.csv` — Player, Pos (QB/RB/WR/TE), Rank (1=best)
- `data/rosters.csv` — Team, Player
- `data/settings.csv` — Target_QB/RB/WR/TE and PosMult_*

## Run
```
python -m venv .venv
.venv\Scripts\activate  # Windows
# or
source .venv/bin/activate # Mac
pip install -r requirements.txt
streamlit run app.py
```

## Tips
- If obvious mismatches still look close, increase **Rank importance** and/or **Elite gap**.
- If depth beats stars, raise **2-for-1 tax**.
- If picks feel hot, lower **Pick max** and/or increase **In-round drop-off**.
