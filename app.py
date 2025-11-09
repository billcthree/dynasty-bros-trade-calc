import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
from itertools import combinations
import random
import requests

# -------------------- Page config --------------------
st.set_page_config(
    page_title="Dynasty Bros. Trade Calculator",
    layout="wide"
)

st.markdown("## Dynasty Bros. Trade Calculator")
st.caption("Superflex PPR trade calculator using FantasyPros rankings and live Sleeper rosters.")

st.markdown(
    "Using **FantasyPros Dynasty Superflex PPR** rankings as the baseline for player value "
    "(latest list from FantasyPros: "
    "[FantasyPros Dynasty SF PPR rankings](https://www.fantasypros.com/nfl/rankings/dynasty-superflex.php))."
)

# ====================================================
# Helpers: name normalization, Sleeper fetch w/ picks
# ====================================================

def normalize_name(name: str) -> str:
    """Normalize player names so Brian Thomas / Brian Thomas Jr. match."""
    if not isinstance(name, str):
        return ""
    name = name.lower().replace(".", " ").replace(",", " ").strip()
    tokens = [
        t for t in name.split()
        if t not in {"jr", "sr", "ii", "iii", "iv", "v"} and len(t) > 1
    ]
    return " ".join(tokens)


@st.cache_data(show_spinner=False)
def load_sleeper_league_v2(league_id: str):
    """
    Fetch Sleeper league info + users + rosters + records + NFL player DB + traded picks.

    Returns:
      rosters_df: Team, Player, Pos
      records_df: Team, Wins, Losses, Ties
      picks_by_team: { current_owner_team: [pick_label, ...] }
      pick_label_to_original: { pick_label: original_team_name }
      future_years: list[int] of seasons we built picks for
    """
    base = f"https://api.sleeper.app/v1/league/{league_id}"

    league_info = requests.get(base, timeout=20).json()
    season = int(league_info.get("season", datetime.now().year))
    draft_rounds = int(league_info.get("draft_rounds", 4))

    # Only care about drafts that haven't happened yet: next 3 after current season
    future_years = [season + i for i in [1, 2, 3]]

    users = requests.get(base + "/users", timeout=20).json()
    rosts = requests.get(base + "/rosters", timeout=20).json()
    traded = requests.get(base + "/traded_picks", timeout=20).json()
    players_nfl = requests.get("https://api.sleeper.app/v1/players/nfl", timeout=30).json()

    # Map owner_id -> nice team name
    id_to_name = {}
    for i, u in enumerate(users):
        meta = u.get("metadata") or {}
        team_name = meta.get("team_name") or u.get("display_name") or f"Team {i+1}"
        id_to_name[u.get("user_id")] = team_name

    rows = []
    records = {}
    rosterid_to_team = {}

    for r in rosts:
        owner_id = r.get("owner_id")
        roster_id = r.get("roster_id")
        team_label = id_to_name.get(owner_id, f"Team {roster_id}")
        rosterid_to_team[roster_id] = team_label

        settings = r.get("settings") or {}
        wins = settings.get("wins", 0)
        losses = settings.get("losses", 0)
        ties = settings.get("ties", 0)
        records[team_label] = {
            "Team": team_label,
            "Wins": wins,
            "Losses": losses,
            "Ties": ties,
        }

        for pid in (r.get("players") or []):
            pl = players_nfl.get(str(pid), {})
            full_name = pl.get("full_name") or (
                ((pl.get("first_name") or "") + " " + (pl.get("last_name") or "")).strip()
            )
            pos = pl.get("position")
            if not full_name or pos not in ["QB", "RB", "WR", "TE"]:
                continue
            rows.append({"Team": team_label, "Player": full_name, "Pos": pos})

    rosters_df = pd.DataFrame(rows)
    records_df = pd.DataFrame(records.values()) if records else pd.DataFrame(
        columns=["Team", "Wins", "Losses", "Ties"]
    )

    # ---------- Build future pick ownership ----------
    picks_current_owner = {}  # (year, round, original_roster_id) -> current_owner_team

    # Initial ownership = original owner
    for r in rosts:
        rid = r.get("roster_id")
        original_team = rosterid_to_team.get(rid)
        if not original_team:
            continue
        for yr in future_years:
            for rnd in range(1, draft_rounds + 1):
                picks_current_owner[(yr, rnd, rid)] = original_team

    # Apply traded picks list (final state)
    for tp in traded or []:
        try:
            yr = int(tp.get("season", 0))
            rnd = int(tp.get("round", 0))
            orig_rid = tp.get("roster_id")
            new_owner_rid = tp.get("owner_id")
        except Exception:
            continue

        if yr not in future_years:
            continue
        if rnd < 1 or rnd > draft_rounds:
            continue
        if orig_rid not in rosterid_to_team or new_owner_rid not in rosterid_to_team:
            continue

        new_owner_team = rosterid_to_team[new_owner_rid]
        picks_current_owner[(yr, rnd, orig_rid)] = new_owner_team

    picks_by_team = {}              # current_owner -> [labels]
    pick_label_to_original = {}     # label -> original team (whose record/strength matters for value)

    for (yr, rnd, orig_rid), current_owner in picks_current_owner.items():
        original_team = rosterid_to_team.get(orig_rid)
        if not original_team or not current_owner:
            continue
        label = f"{yr} R{rnd} ({original_team})"
        picks_by_team.setdefault(current_owner, []).append(label)
        pick_label_to_original[label] = original_team

    for tm in picks_by_team:
        picks_by_team[tm] = sorted(picks_by_team[tm])

    return rosters_df, records_df, picks_by_team, pick_label_to_original, future_years


@st.cache_data(show_spinner=False)
def load_sleeper_logos(league_id: str):
    """
    Optional: league + team logos from Sleeper.
    Returns (league_logo_url, {team_name: logo_url})
    """
    base = f"https://api.sleeper.app/v1/league/{league_id}"
    league_info = requests.get(base, timeout=20).json()
    users = requests.get(base + "/users", timeout=20).json()
    rosts = requests.get(base + "/rosters", timeout=20).json()

    league_logo = None
    if league_info.get("avatar"):
        league_logo = f"https://sleepercdn.com/avatars/{league_info['avatar']}"

    # owner_id -> (team_name, avatar_url)
    owner_to_team = {}
    for i, u in enumerate(users):
        meta = u.get("metadata") or {}
        team_name = meta.get("team_name") or u.get("display_name") or f"Team {i+1}"
        avatar_id = meta.get("team_avatar") or u.get("avatar")
        avatar_url = f"https://sleepercdn.com/avatars/{avatar_id}" if avatar_id else None
        owner_to_team[u.get("user_id")] = (team_name, avatar_url)

    team_logo = {}
    for r in rosts:
        owner_id = r.get("owner_id")
        if owner_id not in owner_to_team:
            continue
        team_name, avatar_url = owner_to_team[owner_id]
        if avatar_url:
            team_logo[team_name] = avatar_url

    return league_logo, team_logo


@st.cache_data(show_spinner=False)
def load_ppr_curves():
    """
    Load PPR scoring curves from data/ppr_curves.xlsx.

    Expected sheets:
      - QB24, RB24, WR24, TE24
    Each sheet should have:
      - '#'  : position rank (1 = top scorer at that position)
      - 'TTL': season total PPR points
    """
    try:
        xls = pd.ExcelFile("data/ppr_curves.xlsx")
    except Exception:
        return None

    curves = {}
    mapping = {"QB": "QB24", "RB": "RB24", "WR": "WR24", "TE": "TE24"}
    for pos, sheet in mapping.items():
        if sheet not in xls.sheet_names:
            continue
        df = xls.parse(sheet)
        if "#" not in df.columns or "TTL" not in df.columns:
            continue
        tmp = df[["#", "TTL"]].copy()
        tmp = tmp.rename(columns={"#": "PosRank", "TTL": "Points"})
        tmp["PosRank"] = pd.to_numeric(tmp["PosRank"], errors="coerce")
        tmp["Points"] = pd.to_numeric(tmp["Points"], errors="coerce")
        tmp = tmp.dropna().sort_values("PosRank")
        if not tmp.empty:
            curves[pos] = tmp.reset_index(drop=True)

    return curves if curves else None


@st.cache_data(show_spinner=False)
def load_age_table():
    """
    Load player ages from data/fantasyage.csv (Player, Age or Player, Yrs).
    Used only for small adjustments to pick values.
    """
    try:
        df = pd.read_csv("data/fantasyage.csv")
    except Exception:
        return None

    cols = [c.lower() for c in df.columns]
    df.columns = cols
    if "player" not in cols or ("age" not in cols and "yrs" not in cols):
        return None

    age_col = "age" if "age" in cols else "yrs"
    out = df[["player", age_col]].rename(columns={"player": "Player", age_col: "Age"})
    out["Age"] = pd.to_numeric(out["Age"], errors="coerce")
    out = out.dropna(subset=["Age"])
    out["Norm"] = out["Player"].apply(normalize_name)
    return out


# ====================================================
# Sidebar: data source + modifiers
# ====================================================

st.sidebar.header("1) Data Source")

use_live = st.sidebar.checkbox(
    "Use live Sleeper + FantasyPros data",
    value=True,
    help="On: rosters/records from Sleeper API, rankings from data/player_ranks.csv (FantasyPros export)."
)

league_id = st.sidebar.text_input(
    "Sleeper League ID",
    value="1194681871141023744",
    help="Paste your Sleeper league ID (from league settings in the Sleeper app)."
)

st.sidebar.caption("You can still upload CSVs below to override or test things manually.")
up_players = st.sidebar.file_uploader("Player Ranks CSV (Player, Pos, Rank)", type=["csv"])
up_rosters = st.sidebar.file_uploader("Rosters CSV (Team, Player, Pos optional)", type=["csv"])

# -------- Modifiers ----------
st.sidebar.header("2) Player value tuning (optional)")
st.sidebar.caption("If you're not sure what to do here, you can safely leave the defaults.")

ELITE_GAP = st.sidebar.slider(
    "How valuable is the #1 overall player?",
    800.0, 2200.0, 1500.0, 50.0,
    help="Higher = bigger gap between elite players and everyone else."
)

RANK_IMPORTANCE = st.sidebar.slider(
    "How fast does value drop as players get lower in the rankings?",
    0.015, 0.060, 0.038, 0.001,
    help="Higher = rank differences matter more (e.g., Rank 26 >> Rank 137)."
)

NEED_WEIGHT = st.sidebar.slider(
    "How much do roster needs matter?",
    0.0, 0.6, 0.20, 0.05,
    help="Lower = mostly pure rankings. Higher = small boost when filling a thin position."
)

PACKAGE_PENALTY = st.sidebar.slider(
    "2-for-1 tax (multiple smaller players vs one stud)",
    0.0, 1.0, 0.75, 0.05,
    help="Higher = quantity counts less vs quality. Prevents 3 mid players from beating 1 superstar."
)

st.sidebar.header("3) Advanced pick tuning (optional)")
with st.sidebar.expander("Show pick value sliders"):
    N_STRENGTH = st.sidebar.slider(
        "Team strength depth (how many best players define team strength)",
        6, 20, 10, 1,
        help="Used to judge how strong a team is when valuing its future picks."
    )

    PICK_MAX = st.sidebar.slider(
        "Pick max (value for earliest 1st-round pick)",
        250.0, 900.0, 500.0, 25.0,
        help="Upper limit for the very best first-round pick."
    )

    PICK_SLOT_DECAY = st.sidebar.slider(
        "How quickly picks lose value within a round",
        0.08, 0.35, 0.20, 0.01,
        help="Higher = early picks are much better than late picks in the same round."
    )

    R2_SCALE = st.sidebar.slider("Round 2 value vs Round 1", 0.20, 0.60, 0.40, 0.01)
    R3_SCALE = st.sidebar.slider("Round 3 value vs Round 1", 0.08, 0.35, 0.20, 0.01)
    R4_SCALE = st.sidebar.slider("Round 4 value vs Round 1", 0.03, 0.25, 0.10, 0.01)

    YEAR2_DISC = st.sidebar.slider("Discount for next-year picks", 0.70, 0.95, 0.85, 0.01)
    YEAR3_DISC = st.sidebar.slider("Discount for picks 2+ years out", 0.50, 0.90, 0.70, 0.01)

st.sidebar.header("4) Update")
manual = st.sidebar.checkbox("Manual mode (click button to recalc)", value=False)
recalc = True
if manual:
    recalc = st.sidebar.button("Recalculate now")

# ====================================================
# Data loading & preparation
# ====================================================

DEFAULT_TARGETS = {"QB": 2, "RB": 4, "WR": 5, "TE": 2}
DEFAULT_POSMULT = {"QB": 1.10, "RB": 1.00, "WR": 1.00, "TE": 0.95}

# These will be filled depending on mode
picks_by_team = {}
pick_label_to_original = {}
future_pick_years = []

# ---- Live mode: Sleeper + local FantasyPros CSV ----
if use_live and league_id.strip():
    try:
        with st.spinner("Loading Sleeper league data + local FantasyPros rankings..."):
            rosters_live, records_df, picks_by_team, pick_label_to_original, future_pick_years = load_sleeper_league_v2(
                league_id.strip()
            )

            fp_ranks = pd.read_csv("data/player_ranks.csv")
            fp_ranks["Pos"] = fp_ranks["Pos"].astype(str).str.upper().str.strip()
            fp_ranks["Rank"] = pd.to_numeric(fp_ranks["Rank"], errors="coerce")
            fp_ranks = fp_ranks.dropna(subset=["Rank"]).reset_index(drop=True)
            fp_ranks["Norm"] = fp_ranks["Player"].apply(normalize_name)

        fp_max_rank = int(fp_ranks["Rank"].max())
        fp2 = fp_ranks.copy()

        roster_enriched = rosters_live.copy()
        roster_enriched["Norm"] = roster_enriched["Player"].apply(normalize_name)
        roster_enriched = roster_enriched.merge(
            fp2[["Norm", "Rank", "Pos"]].rename(columns={"Pos": "FP_Pos"}),
            on="Norm",
            how="left",
        )

        roster_enriched["Pos_use"] = roster_enriched["FP_Pos"].fillna(roster_enriched["Pos"])
        roster_enriched["Rank_use"] = roster_enriched["Rank"].fillna(fp_max_rank + 80)

        players_df = (
            roster_enriched[["Player", "Pos_use", "Rank_use"]]
            .drop_duplicates("Player")
            .rename(columns={"Pos_use": "Pos", "Rank_use": "Rank"})
        )

        rosters_df = roster_enriched[["Team", "Player"]].copy()

        targets = DEFAULT_TARGETS
        posmult = DEFAULT_POSMULT

        st.success("Loaded Sleeper rosters + local FantasyPros rankings.")
    except Exception as e:
        st.error(f"Live data load failed: {e}")
        st.info("Falling back to CSV uploads (if provided).")
        use_live = False
else:
    use_live = False

# ---- Non-live mode: CSV uploads only ----
if not use_live:
    base_players = pd.DataFrame(columns=["Player", "Pos", "Rank"])
    base_rosters = pd.DataFrame(columns=["Team", "Player"])
    players_df = base_players
    rosters_df = base_rosters

    if up_players is not None:
        players_df = pd.read_csv(up_players)
    if up_rosters is not None:
        rosters_df = pd.read_csv(up_rosters)

    if players_df.empty or not {"Player", "Pos", "Rank"}.issubset(players_df.columns):
        st.error("Please provide player_ranks.csv (Player, Pos, Rank) or enable live mode.")
        st.stop()
    if rosters_df.empty or not {"Team", "Player"}.issubset(rosters_df.columns):
        st.error("Please provide rosters.csv (Team, Player) or enable live mode.")
        st.stop()

    players_df["Pos"] = players_df["Pos"].astype(str).str.upper().str.strip()
    players_df["Rank"] = pd.to_numeric(players_df["Rank"], errors="coerce")
    players_df = players_df.dropna(subset=["Rank"]).reset_index(drop=True)

    rosters_df["Team"] = rosters_df["Team"].astype(str).str.strip()
    rosters_df["Player"] = rosters_df["Player"].astype(str).str.strip()

    records_df = (
        rosters_df[["Team"]].drop_duplicates().assign(Wins=0, Losses=0, Ties=0)
    )

    targets = DEFAULT_TARGETS
    posmult = DEFAULT_POSMULT

    current_year = datetime.now().year
    future_pick_years = [current_year + i for i in [1, 2, 3]]
    picks_by_team = {}
    pick_label_to_original = {}
    for tm in rosters_df["Team"].unique():
        for yr in future_pick_years:
            for rnd in [1, 2, 3, 4]:
                label = f"{yr} R{rnd} ({tm})"
                picks_by_team.setdefault(tm, []).append(label)
                pick_label_to_original[label] = tm
    for tm in picks_by_team:
        picks_by_team[tm] = sorted(picks_by_team[tm])

# ---- Optional: logos ----
league_logo = None
team_logo_map = {}
if use_live and league_id.strip():
    try:
        league_logo, team_logo_map = load_sleeper_logos(league_id.strip())
    except Exception:
        league_logo, team_logo_map = None, {}

if league_logo:
    st.image(league_logo, width=72)

# ====================================================
# Core valuation helpers (with PPR curves & scarcity)
# ====================================================

players_df = players_df.copy()
players_df["Rank"] = pd.to_numeric(players_df["Rank"], errors="coerce")
players_df = players_df.dropna(subset=["Rank"]).reset_index(drop=True)

players_df["Pos"] = players_df["Pos"].astype(str).str.upper().str.strip()
players_df["PosRank"] = players_df.groupby("Pos")["Rank"].rank(method="first")

TOP_CUTOFF = 100
top_slice = players_df[players_df["Rank"] <= TOP_CUTOFF]
counts = top_slice["Pos"].value_counts()
avg_count = counts.mean() if len(counts) else 1.0

scarcity_mult = {}
for pos in ["QB", "RB", "WR", "TE"]:
    c = counts.get(pos, 1)
    raw = avg_count / c
    raw_clamped = max(0.5, min(2.0, raw))
    scarcity = 0.85 + (raw_clamped - 0.5) * (1.15 - 0.85) / (2.0 - 0.5)
    scarcity_mult[pos] = scarcity

posmult_effective = {}
for pos in ["QB", "RB", "WR", "TE"]:
    base_mult = DEFAULT_POSMULT.get(pos, 1.0)
    eff = base_mult * scarcity_mult.get(pos, 1.0)
    if pos == "TE":
        eff = min(eff, 0.95)  # cap TE so it doesn't beat elite WR/RB just for scarcity
    posmult_effective[pos] = eff

players_df["PosMult"] = players_df["Pos"].map(posmult_effective).fillna(1.0)

ppr_curves = load_ppr_curves()

def estimate_points(row):
    pos = row["Pos"]
    pos_rank = row["PosRank"]
    if not ppr_curves or pos not in ppr_curves:
        return np.nan
    df = ppr_curves[pos]
    k = int(round(pos_rank))
    subset = df[df["PosRank"] >= k]
    if not subset.empty:
        return float(subset.iloc[0]["Points"])
    else:
        return float(df["Points"].iloc[-1])

players_df["ModelPoints"] = players_df.apply(estimate_points, axis=1)

if players_df["ModelPoints"].isna().any():
    fallback_idx = players_df["ModelPoints"].isna()
    max_rank_tmp = players_df["Rank"].max()
    players_df.loc[fallback_idx, "ModelPoints"] = (
        max_rank_tmp - players_df.loc[fallback_idx, "Rank"] + 1
    )

max_pts = players_df["ModelPoints"].max()

if max_pts <= 0:
    # Fallback if something goes wrong with the PPR curves
    players_df["BaseValue"] = (
        ELITE_GAP * np.exp(-RANK_IMPORTANCE * (players_df["Rank"] - 1))
    ).round(2)
else:
    # 1) Start from relative PPR points
    rel_pts = (players_df["ModelPoints"] / max_pts).clip(0.0001, 1.0)

    # 2) Make the curve steeper so elites separate more from mid-tier
    #    Bigger RANK_IMPORTANCE still makes the drop-off faster.
    curve_power = 1.6 + (RANK_IMPORTANCE - 0.015) * 30.0
    base_curve = np.power(rel_pts, curve_power)

    # 3) Tier multiplier: top overall ranks get a little extra bump
    #    This keeps guys like CeeDee / Waddle clearly above WR20–WR30 types.
    overall_rank = players_df["Rank"].rank(method="first")

    tier_mult = np.where(
        overall_rank <= 12, 1.30,        # true elite tier
        np.where(
            overall_rank <= 24, 1.18,    # strong WR1 / RB1 / QB1 range
            np.where(
                overall_rank <= 48, 1.08,  # solid starters
                1.00                       # everyone else
            )
        )
    )

    players_df["BaseValue"] = (ELITE_GAP * base_curve * tier_mult).round(2)


ages_table = load_age_table()
team_avg_age = {}
league_avg_age = None
if ages_table is not None and not rosters_df.empty:
    ages_norm = ages_table.copy()
    roster_with_norm = rosters_df.copy()
    roster_with_norm["Norm"] = roster_with_norm["Player"].apply(normalize_name)
    ages_norm["Norm"] = ages_norm["Norm"].astype(str)
    roster_with_norm = roster_with_norm.merge(
        ages_norm[["Norm", "Age"]], on="Norm", how="left"
    )
    grp = roster_with_norm.groupby("Team")["Age"].mean()
    team_avg_age = grp.to_dict()
    if len(team_avg_age):
        league_avg_age = float(np.nanmean(list(team_avg_age.values())))

def team_pos_counts(team: str):
    names = set(rosters_df.loc[rosters_df["Team"] == team, "Player"].tolist())
    sub = players_df[players_df["Player"].isin(names)]
    return {p: int((sub["Pos"] == p).sum()) for p in ["QB", "RB", "WR", "TE"]}

def need_multiplier(count, target):
    diff = count - target
    if diff <= -2:
        return 1.10
    if diff == -1:
        return 1.05
    if diff == 0:
        return 1.00
    if diff == 1:
        return 0.97
    return 0.93

def apply_need(pos, team_counts):
    base_mult = need_multiplier(team_counts.get(pos, 0), DEFAULT_TARGETS.get(pos, 0))
    return 1.0 + NEED_WEIGHT * (base_mult - 1.0)

def roster_players(team: str):
    names = rosters_df.loc[rosters_df["Team"] == team, "Player"].tolist()
    names = [n for n in names if n in set(players_df["Player"])]
    names.sort(key=lambda x: x.lower())
    return names

label_map = {
    row["Player"]: f'{row["Player"]} ({row["Pos"]})'
    for _, row in players_df.iterrows()
}

def team_strength(team: str):
    names = rosters_df.loc[rosters_df["Team"] == team, "Player"].tolist()
    sub = players_df[players_df["Player"].isin(names)].copy()
    vals = (sub["BaseValue"] * sub["PosMult"]).sort_values(ascending=False).head(N_STRENGTH)
    return float(vals.sum()) if len(vals) else 0.0

team_list = sorted(rosters_df["Team"].unique().tolist())
strengths = {t: team_strength(t) for t in team_list}

records_df = records_df.set_index("Team")
def get_record(team: str):
    if team in records_df.index:
        row = records_df.loc[team]
        return float(row.get("Wins", 0)), float(row.get("Losses", 0))
    return 0.0, 0.0

sorted_for_picks = sorted(
    team_list,
    key=lambda t: (get_record(t)[0], strengths[t])
)
team_pick_slot = {t: i + 1 for i, t in enumerate(sorted_for_picks)}

original_pick_counts = {}
for lbl, orig in pick_label_to_original.items():
    original_pick_counts[orig] = original_pick_counts.get(orig, 0) + 1
avg_num_picks = float(np.mean(list(original_pick_counts.values()))) if original_pick_counts else 0.0

def pick_factor_for_round(rnd: int) -> float:
    mapping = {1: 1.0, 2: R2_SCALE, 3: R3_SCALE, 4: R4_SCALE}
    return mapping.get(rnd, 0.0)

def pick_base_value(slot: int, rnd: int) -> float:
    round_mult = pick_factor_for_round(rnd)
    maxslot = max(1, len(team_list))
    slot = min(max(1, int(slot)), maxslot)
    slot_val = np.exp(-PICK_SLOT_DECAY * (slot - 1))
    return PICK_MAX * round_mult * slot_val

def pick_value(original_team: str, year: int, rnd: int) -> float:
    current_year = datetime.now().year
    diff_years = year - current_year
    if diff_years <= 0:
        year_mult = 1.0
    elif diff_years == 1:
        year_mult = YEAR2_DISC
    else:
        year_mult = YEAR3_DISC ** diff_years

    slot = team_pick_slot.get(original_team, len(team_list))
    base = pick_base_value(slot, rnd)

    age_mult = 1.0
    if league_avg_age and original_team in team_avg_age:
        delta_age = team_avg_age[original_team] - league_avg_age
        delta_age = max(-3.0, min(3.0, delta_age))
        age_mult = 1.0 + 0.02 * delta_age

    inv_mult = 1.0
    if avg_num_picks > 0 and original_team in original_pick_counts:
        delta_picks = avg_num_picks - original_pick_counts[original_team]
        delta_picks = max(-3.0, min(3.0, delta_picks))
        inv_mult = 1.0 + 0.01 * delta_picks

    return round(base * year_mult * age_mult * inv_mult, 1)

def build_pick_labels_for_team(team: str):
    return picks_by_team.get(team, [])

def parse_pick_label(label: str):
    try:
        parts = label.split()
        yr = int(parts[0])
        rnd = int(parts[1].replace("R", ""))
        original_team = label[label.find("(") + 1:label.find(")")]
        return yr, rnd, original_team
    except Exception:
        return None, None, None

def label_value(lbl: str) -> float:
    yr, rnd, original_team = parse_pick_label(lbl)
    if yr is None:
        return 0.0
    return pick_value(original_team, yr, rnd)

def package_sum(values):
    if not values:
        return 0.0
    values = sorted(values, reverse=True)
    total = 0.0
    for i, v in enumerate(values, start=1):
        weight = (i ** (-PACKAGE_PENALTY)) if PACKAGE_PENALTY > 0 else 1.0
        total += v * weight
    return total

def sum_players_value(player_list, team_for_need):
    vals = []
    details = []
    counts = team_pos_counts(team_for_need)
    for n in player_list:
        r = players_df.loc[players_df["Player"] == n]
        if r.empty:
            continue
        pos = r["Pos"].iloc[0]
        rank = int(r["Rank"].iloc[0])
        base_before_need = float(r["BaseValue"].iloc[0]) * float(r["PosMult"].iloc[0])
        mult = apply_need(pos, counts)
        val = base_before_need * mult
        vals.append(val)
        details.append((n, pos, rank, base_before_need, mult, val))
    return package_sum(vals), details

def sum_picks_value(pick_labels):
    vals = []
    details = []
    for lbl in pick_labels:
        v = label_value(lbl)
        vals.append(v)
        details.append((lbl, v))
    return package_sum(vals), details

def counts_after_trade(team, send_players, receive_players):
    counts = team_pos_counts(team).copy()
    for n in send_players:
        r = players_df.loc[players_df["Player"] == n]
        if r.empty:
            continue
        pos = r["Pos"].iloc[0]
        counts[pos] = counts.get(pos, 0) - 1
    for n in receive_players:
        r = players_df.loc[players_df["Player"] == n]
        if r.empty:
            continue
        pos = r["Pos"].iloc[0]
        counts[pos] = counts.get(pos, 0) + 1
    for p in ["QB", "RB", "WR", "TE"]:
        counts[p] = max(0, counts.get(p, 0))
    return counts

def needs_and_risk_summary(team, before_counts, after_counts, incoming_details):
    notes = []
    inc_counts = {"QB": 0, "RB": 0, "WR": 0, "TE": 0}
    for _, pos, _, _, _, _ in incoming_details:
        inc_counts[pos] = inc_counts.get(pos, 0) + 1

    for pos in ["QB", "RB", "WR", "TE"]:
        tgt = int(DEFAULT_TARGETS.get(pos, 0))
        before = before_counts.get(pos, 0)
        after = after_counts.get(pos, 0)
        inc = inc_counts.get(pos, 0)

        if inc > 0 and before < tgt:
            if before == 0 and pos == "QB":
                notes.append(
                    f"{team} badly needs a starting QB (goes from {before} to {after}). "
                    f"Adding a QB here is especially valuable."
                )
            else:
                notes.append(
                    f"{team} could use more {pos}s (goes from {before} to {after} at {pos}), "
                    f"so this trade helps their depth."
                )

        if after < max(1, tgt) and before >= after:
            notes.append(
                f"After this trade, {team} would have {after} {pos}(s) "
                f"(we usually aim for around {tgt}), so they might worry about {pos} depth."
            )

    return notes

def pick_summary(pick_details):
    if not pick_details:
        return "No picks involved for this side."
    total = sum(v for _, v in pick_details)
    firsts = [lbl for lbl, _ in pick_details if " R1 " in lbl]
    others = [lbl for lbl, _ in pick_details if " R1 " not in lbl]
    parts = []
    if firsts:
        parts.append(f"{len(firsts)} first-round pick(s)")
    if others:
        parts.append(f"{len(others)} later-round pick(s)")
    return f"Receiving {', '.join(parts)} worth about {total:,.0f} value points."

def suggest_trade_balance(team_favored, team_behind, gap, favored_send_players, favored_send_picks):
    if gap < 50:
        return None

    candidates = []
    for lbl in build_pick_labels_for_team(team_favored):
        if lbl in favored_send_picks:
            continue
        v = label_value(lbl)
        if v <= 0:
            continue
        candidates.append(("pick", lbl, v))

    for p in roster_players(team_favored):
        if p in favored_send_players:
            continue
        v, _ = sum_players_value([p], team_behind)
        if v <= 0:
            continue
        candidates.append(("player", p, v))

    if not candidates:
        return None

    best = None
    for kind, name, v in candidates:
        if best is None or abs(v - gap) < abs(best[2] - gap):
            best = (kind, name, v)
    return best

# ====================================================
# UI layout: tabs
# ====================================================

tab_trade, tab_finder = st.tabs(["💼 Trade Calculator", "🔍 Trade Finder"])

# ----------------------------------------------------
# TAB 1: TRADE CALCULATOR
# ----------------------------------------------------
with tab_trade:
    st.subheader("Build a Trade")

    colA, colB = st.columns(2)
    with colA:
        teamA = st.selectbox("Team A (left side)", team_list, index=0 if team_list else None)
        if teamA in team_logo_map:
            st.image(team_logo_map[teamA], width=40)
    with colB:
        teamB_choices = [t for t in team_list if t != (teamA if team_list else None)]
        teamB = st.selectbox("Team B (right side)", teamB_choices, index=0 if teamB_choices else None)
        if teamB in team_logo_map:
            st.image(team_logo_map[teamB], width=40)

    if not teamA or not teamB:
        st.info("Once your Sleeper league loads, pick two teams and add players/picks to see a trade evaluation.")
    else:
        a_players_list = roster_players(teamA)
        b_players_list = roster_players(teamB)

        left, right = st.columns(2)
        with left:
            st.markdown(f"#### {teamA} sends → {teamB}")
            a_send_players = st.multiselect(
                "Players",
                a_players_list,
                format_func=lambda x: label_map.get(x, x),
                key="a_send_players",
            )
            a_send_picks = st.multiselect(
                "Picks",
                build_pick_labels_for_team(teamA),
                key="a_send_picks",
            )
        with right:
            st.markdown(f"#### {teamB} sends → {teamA}")
            b_send_players = st.multiselect(
                "Players",
                b_players_list,
                format_func=lambda x: label_map.get(x, x),
                key="b_send_players",
            )
            b_send_picks = st.multiselect(
                "Picks",
                build_pick_labels_for_team(teamB),
                key="b_send_picks",
            )

        if not manual or recalc:
            if not (a_send_players or a_send_picks or b_send_players or b_send_picks):
                st.info("Build a trade by selecting at least one player or pick on either side.")
            else:
                # Each side RECEIVES:
                A_get_players, A_get_p_det = sum_players_value(b_send_players, teamA)
                A_get_picks,   A_get_pk_det = sum_picks_value(b_send_picks)
                A_get_total = A_get_players + A_get_picks

                B_get_players, B_get_p_det = sum_players_value(a_send_players, teamB)
                B_get_picks,   B_get_pk_det = sum_picks_value(a_send_picks)
                B_get_total = B_get_players + B_get_picks

                m1, m2 = st.columns(2)
                m1.metric(f"{teamA} receives (model value)", f"{A_get_total:,.0f}")
                m2.metric(f"{teamB} receives (model value)", f"{B_get_total:,.0f}")

                countsA_before = team_pos_counts(teamA)
                countsB_before = team_pos_counts(teamB)
                countsA_after = counts_after_trade(teamA, a_send_players, b_send_players)
                countsB_after = counts_after_trade(teamB, b_send_players, a_send_players)

                A_notes = needs_and_risk_summary(teamA, countsA_before, countsA_after, A_get_p_det)
                B_notes = needs_and_risk_summary(teamB, countsB_before, countsB_after, B_get_p_det)

                st.markdown("---")
                st.subheader("Fairness verdict")

                diff = A_get_total - B_get_total
                larger = max(A_get_total, B_get_total, 1.0)
                pct = diff / larger

                def grade(pct_diff: float) -> str:
                    ad = abs(pct_diff)
                    if ad < 0.05:
                        return "The numbers here are very close, so both sides could reasonably feel okay about it."
                    elif ad < 0.12:
                        return "One side has a modest edge by the numbers, but it's still within a range where both managers might be comfortable."
                    else:
                        return "One side is getting a noticeable edge on pure value; the other side would likely need strong non-numbers reasons (team fit, age profile, risk) to accept."

                if abs(pct) < 0.03:
                    headline = (
                        f"By the model, this trade is roughly **even**. "
                        f"{teamA} and {teamB} are getting similar overall value."
                    )
                elif pct > 0:
                    headline = (
                        f"This trade likely **favors {teamA}**. "
                        f"Our model has {teamA} receiving about **{abs(diff):,.0f} more value points** "
                        f"than {teamB}, which is roughly **{abs(pct)*100:.1f}% more** than what {teamB} is getting."
                    )
                else:
                    headline = (
                        f"This trade likely **favors {teamB}**. "
                        f"Our model has {teamB} receiving about **{abs(diff):,.0f} more value points** "
                        f"than {teamA}, which is roughly **{abs(pct)*100:.1f}% more** than what {teamA} is getting."
                    )

                st.write(headline)
                st.caption(
                    '"Value points" are on this app’s internal scale – higher means more expected future production. '
                    "The % compares the two sides to show how far apart they are."
                )
                st.info(grade(pct))

                def summarize_side(details):
                    if not details:
                        return ""
                    df = pd.DataFrame(details, columns=["Player", "Pos", "FP Rank", "Base Value", "Need Mult", "Final Value"])
                    df = df.sort_values("Final Value", ascending=False)
                    tops = df.head(2)
                    parts = []
                    for _, r in tops.iterrows():
                        parts.append(f"{r['Player']} (Rank {int(r['FP Rank'])} {r['Pos']})")
                    return ", ".join(parts)

                A_summary = summarize_side(A_get_p_det)
                B_summary = summarize_side(B_get_p_det)

                if A_summary or B_summary or A_get_pk_det or B_get_pk_det:
                    st.markdown("**Biggest pieces each side is getting (by this model):**")
                    if A_summary:
                        st.markdown(f"- {teamA} is mainly gaining: {A_summary}.")
                    if A_get_pk_det:
                        st.markdown(f"- For {teamA}, picks add: {pick_summary(A_get_pk_det)}")
                    if B_summary:
                        st.markdown(f"- {teamB} is mainly gaining: {B_summary}.")
                    if B_get_pk_det:
                        st.markdown(f"- For {teamB}, picks add: {pick_summary(B_get_pk_det)}")

                st.markdown("### How to even this trade out")

                gap = abs(diff)
                if abs(pct) < 0.03 or gap < 50:
                    st.write(
                        "This already looks fairly close. If managers want it perfectly even, "
                        "they could consider a small late-round pick swap or adding a minor depth piece."
                    )
                else:
                    if diff > 0:
                        favored_team = teamA
                        behind_team = teamB
                        suggestion = suggest_trade_balance(teamA, teamB, gap, a_send_players, a_send_picks)
                    else:
                        favored_team = teamB
                        behind_team = teamA
                        suggestion = suggest_trade_balance(teamB, teamA, gap, b_send_players, b_send_picks)

                    if suggestion is None:
                        st.write(
                            f"The model sees a noticeable gap, but couldn't find a clean single asset "
                            f"from {favored_team} that closely matches the difference. "
                            "Consider combining a smaller player with a mid/late pick."
                        )
                    else:
                        kind, name, v = suggestion
                        new_gap = abs(gap - v)
                        if kind == "pick":
                            st.write(
                                f"To bring things closer, **{favored_team}** could add pick **{name}** "
                                f"(worth ~{v:,.0f} value points) going to **{behind_team}**. "
                                f"That would shrink the gap from about {gap:,.0f} down to roughly {new_gap:,.0f}."
                            )
                        else:
                            st.write(
                                f"To bring things closer, **{favored_team}** could add depth player **{label_map.get(name, name)}** "
                                f"(worth ~{v:,.0f} value points) going to **{behind_team}**. "
                                f"That would shrink the gap from about {gap:,.0f} down to roughly {new_gap:,.0f}."
                            )

                st.markdown("### Roster context — why managers might feel differently")

                if A_notes:
                    st.markdown(f"**For {teamA}:**")
                    for n in A_notes[:4]:
                        st.markdown(f"- {n}")
                if B_notes:
                    st.markdown(f"**For {teamB}:**")
                    for n in B_notes[:4]:
                        st.markdown(f"- {n}")
                if not (A_notes or B_notes):
                    st.info(
                        "Neither team has obvious positional red flags in this deal. "
                        "At that point, it often comes down to personal preference, age windows, "
                        "and risk tolerance."
                    )

                def key_pieces(details):
                    if not details:
                        return pd.DataFrame()
                    df = pd.DataFrame(details, columns=["Player", "Pos", "FP Rank", "Base Value", "Need Mult", "Final Value"])
                    df = df.sort_values("Final Value", ascending=False)
                    return df.head(5)

                with st.expander("See the numbers behind this suggestion (FantasyPros ranks & model values)"):
                    st.write(f"**What {teamA} receives (players)** — including FantasyPros Superflex PPR ranks")
                    dfA = key_pieces(A_get_p_det)
                    if not dfA.empty:
                        st.table(dfA)
                    if A_get_pk_det:
                        st.write(f"**Picks to {teamA}**")
                        st.table(pd.DataFrame(A_get_pk_det, columns=["Pick", "Value"]))

                    st.write(f"**What {teamB} receives (players)** — including FantasyPros Superflex PPR ranks")
                    dfB = key_pieces(B_get_p_det)
                    if not dfB.empty:
                        st.table(dfB)
                    if B_get_pk_det:
                        st.write(f"**Picks to {teamB}**")
                        st.table(pd.DataFrame(B_get_pk_det, columns=["Pick", "Value"]))


# ----------------------------------------------------
# TAB 2: TRADE FINDER
# ----------------------------------------------------
with tab_finder:
    st.subheader("Trade Finder — suggest packages for a target")

    tf_from = st.selectbox("Your team (offering)", team_list, key="tf_from")
    if tf_from in team_logo_map:
        st.image(team_logo_map[tf_from], width=40)
    tf_to_choices = [t for t in team_list if t != tf_from]
    tf_to = st.selectbox("Trade partner", tf_to_choices, key="tf_to")
    if tf_to in team_logo_map:
        st.image(team_logo_map[tf_to], width=40)

    target_list = roster_players(tf_to)
    tf_target = st.selectbox(
        f"Target player on {tf_to}",
        target_list,
        format_func=lambda x: label_map.get(x, x),
    )

    def player_value_for_team(player_name, team_for_need):
        v, det = sum_players_value([player_name], team_for_need)
        if not det:
            return 0.0, 1.0, None
        _, pos, rank, base_val, mult, _ = det[0]
        return v, mult, rank

    target_val, target_mult, target_rank = player_value_for_team(tf_target, tf_to)
    st.caption(
        f"Estimated value of {label_map.get(tf_target, tf_target)} **to {tf_to}**: "
        f"~{target_val:,.0f} (FantasyPros Rank {target_rank}, need multiplier: {target_mult:.2f}). "
        "Aim for roughly 90–110% of this value in offers."
    )

    your_names = roster_players(tf_from)
    cand = players_df[players_df["Player"].isin(your_names)].copy()

    def val_to_partner(name):
        v, _ = sum_players_value([name], tf_to)
        return v

    from_counts = team_pos_counts(tf_from)
    def surplus_score(pos):
        return max(0, from_counts.get(pos, 0) - DEFAULT_TARGETS.get(pos, 0))

    cand["ValueToPartner"] = cand["Player"].apply(val_to_partner)
    cand["Surplus"] = cand["Pos"].apply(surplus_score)
    cand["Score"] = cand["ValueToPartner"] * (1 + 0.2 * cand["Surplus"])
    cand = cand.sort_values(["Score", "ValueToPartner"], ascending=[False, False]).reset_index(drop=True)

    TOPN = min(12, len(cand))
    LOW, HIGH = 0.90, 1.10

    def pick_labels_for(team):
        return build_pick_labels_for_team(team)

    def package_value(asset_names):
        vals = [val_to_partner(n) for n in asset_names if n in label_map]
        return package_sum(vals)

    def bridge_with_pick(team, current_val, target):
        need = max(0.0, target - current_val)
        if need <= 0:
            return None, 0.0
        best_lbl, best_delta = None, 0.0
        for lbl in pick_labels_for(team):
            v = label_value(lbl)
            if v >= need * 0.6 and (best_lbl is None or abs(v - need) < abs(best_delta - need)):
                best_lbl, best_delta = lbl, v
        if best_lbl is None:
            for lbl in pick_labels_for(team):
                v = label_value(lbl)
                if v > best_delta:
                    best_lbl, best_delta = lbl, v
        return best_lbl, best_delta

    if "tf_seed" not in st.session_state:
        st.session_state["tf_seed"] = 0
    if st.button("Suggest another trade"):
        st.session_state["tf_seed"] += 1
    random.seed(st.session_state["tf_seed"])

    suggestions = []
    base_list = cand.head(TOPN)["Player"].tolist()
    random.shuffle(base_list)

    for size in [1, 2, 3]:
        for combo in combinations(base_list, size):
            assets = list(combo)
            base_val = package_value(assets)
            pkg_val = base_val
            if pkg_val < target_val * LOW:
                lbl, dv = bridge_with_pick(tf_from, pkg_val, target_val)
                if lbl:
                    assets = assets + [lbl]
                    pkg_val = package_sum([base_val, dv])
            if target_val * LOW <= pkg_val <= target_val * HIGH:
                suggestions.append((assets, pkg_val))

    seen = set()
    uniq = []
    for assets, v in suggestions:
        key = tuple(sorted(assets))
        if key in seen:
            continue
        seen.add(key)
        uniq.append((assets, v))

    uniq.sort(key=lambda x: (len(x[0]), abs(x[1] - target_val), -x[1]))

    st.markdown("#### Suggested packages")
    if not uniq:
        st.write(
            "No clean suggestions found in the 90–110% range. "
            "Try lowering the 2-for-1 tax or rank importance, or click **Suggest another trade**."
        )
    else:
        for i, (assets, v) in enumerate(uniq[:5], start=1):
            pct = v / target_val if target_val > 0 else 0
            st.write(f"**Suggestion {i}** — value to {tf_to}: {v:,.0f} (~{pct:.0%} of the target player's value)")
            st.write("- " + "\n- ".join(assets))

# ----------------------------------------------------
# Explainer / assumptions
# ----------------------------------------------------
with st.expander("How this calculator works & assumptions"):
    st.markdown(
        """
- **Player values** start from FantasyPros **Dynasty Superflex PPR** ranks  
  (lower rank number = better player).
- We then look at last season's **PPR scoring curves** by position  
  (QB / RB / WR / TE) from your `data/ppr_curves.xlsx` file.
- For each player:
  - We find their position rank (for example WR8).
  - We map that to an expected PPR total based on historical curves.
  - We normalize across all players so the very top asset is around your
    “How valuable is the #1 player?” slider.
- **Position value** is adjusted two ways:
  - A small fixed bump (QB a bit higher, TE a bit lower by default).
  - A scarcity tweak (positions with fewer top options get a small boost).  
    TE is capped so mid TEs do not jump ahead of elite WR/RB purely on scarcity.
- **Roster needs**: if a team is light at a position, incoming players at that
  position get a small boost. This is intentionally a light factor; rankings and
  scoring curves are still the core.
- **Packages**: multiple smaller players do not fully equal one stud. The
  2-for-1 slider controls that tax.
- **Future picks**:
  - We use Sleeper's traded picks endpoint to see who owns which future picks.
  - Value depends on the original team's record and roster strength (worse team ⇒
    earlier, more valuable pick).
  - There are small nudges based on how old the original roster is and how many
    picks they already have.
- The fairness verdict is meant as guidance, not a law. It explains:
  - Which side likely gets the better deal on pure value.
  - How far apart the packages are (both in raw value points and %).
  - Roster context that might make a manager like or dislike the trade anyway.
"""
    )
