import streamlit as st
import pandas as pd
import numpy as np
import requests
from datetime import datetime
import math
import random

# -------------------- Page config --------------------

st.set_page_config(page_title="Dynasty Bros. Trade Calculator", layout="wide")

# ====================================================
# Helpers: name normalization, Sleeper + FantasyPros
# ====================================================

def normalize_name(name: str) -> str:
    """Normalize player names so Brian Thomas / Brian Thomas Jr. match."""
    if not isinstance(name, str):
        return ""
    name = (
        name.lower()
        .replace(".", " ")
        .replace(",", " ")
        .replace("'", "")
        .strip()
    )
    tokens = [
        t
        for t in name.split()
        if t not in {"jr", "sr", "ii", "iii", "iv", "v"} and len(t) > 1
    ]
    return " ".join(tokens)


@st.cache_data(show_spinner=False)
def load_sleeper_league(league_id: str):
    """Fetch rosters, records, and traded picks from Sleeper."""
    base = f"https://api.sleeper.app/v1/league/{league_id}"
    league_info = requests.get(base, timeout=20).json()
    season = int(league_info.get("season", datetime.now().year))
    draft_rounds = int(league_info.get("draft_rounds", 4))

    # Only care about drafts that haven't happened yet: next 3 seasons
    future_years = [season + i for i in (1, 2, 3)]

    users = requests.get(base + "/users", timeout=20).json()
    rosters_raw = requests.get(base + "/rosters", timeout=20).json()
    traded = requests.get(base + "/traded_picks", timeout=20).json()
    players_nfl = requests.get(
        "https://api.sleeper.app/v1/players/nfl", timeout=30
    ).json()

    # Map owner_id -> nice team name
    owner_id_to_team = {}
    for i, u in enumerate(users):
        meta = u.get("metadata") or {}
        team_name = meta.get("team_name") or u.get("display_name") or f"Team {i+1}"
        owner_id_to_team[u.get("user_id")] = team_name

    rows = []
    records = {}
    rosterid_to_team = {}

    for r in rosters_raw:
        owner_id = r.get("owner_id")
        roster_id = r.get("roster_id")
        team_label = owner_id_to_team.get(owner_id, f"Team {roster_id}")
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
                (pl.get("first_name") or "") + " " + (pl.get("last_name") or "")
            ).strip()
            pos = pl.get("position")
            if not full_name or pos not in ["QB", "RB", "WR", "TE"]:
                continue
            rows.append({"Team": team_label, "Player": full_name, "Pos": pos})

    rosters_df = pd.DataFrame(rows)
    records_df = (
        pd.DataFrame(records.values())
        if records
        else pd.DataFrame(columns=["Team", "Wins", "Losses", "Ties"])
    )

    # ---------- Build future pick ownership ----------
    # (year, round, original_roster_id) -> current_owner_team
    picks_current_owner = {}

    # Initial ownership = original owner
    for r in rosters_raw:
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
        if orig_rid not in rosterid_to_team or new_owner_rid not in rosterid_to_team:
            continue

        new_owner_team = rosterid_to_team[new_owner_rid]
        picks_current_owner[(yr, rnd, orig_rid)] = new_owner_team

    picks_by_team = {}           # current_owner -> [labels]
    pick_label_to_original = {}  # label -> original team

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
    """Return (league_logo_url, {team_name: logo_url}) from Sleeper (best effort)."""
    base = f"https://api.sleeper.app/v1/league/{league_id}"
    league_info = requests.get(base, timeout=20).json()
    users = requests.get(base + "/users", timeout=20).json()
    rosters_raw = requests.get(base + "/rosters", timeout=20).json()

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
    for r in rosters_raw:
        owner_id = r.get("owner_id")
        if owner_id not in owner_to_team:
            continue
        team_name, avatar_url = owner_to_team[owner_id]
        if avatar_url:
            team_logo[team_name] = avatar_url

    return league_logo, team_logo


@st.cache_data(show_spinner=False)
def load_ppr_curves():
    """Load PPR scoring curves from data/ppr_curves.xlsx (if present)."""
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
    """Load player ages from data/fantasyage.csv (Player, Age or Player, Yrs)."""
    try:
        df = pd.read_csv("data/fantasyage.csv")
    except Exception:
        return None

    cols_lower = [c.lower() for c in df.columns]
    df.columns = cols_lower
    if "player" not in cols_lower or ("age" not in cols_lower and "yrs" not in cols_lower):
        return None

    age_col = "age" if "age" in cols_lower else "yrs"
    out = df[["player", age_col]].rename(columns={"player": "Player", age_col: "Age"})
    out["Age"] = pd.to_numeric(out["Age"], errors="coerce")
    out = out.dropna(subset=["Age"])
    out["Norm"] = out["Player"].apply(normalize_name)
    return out


@st.cache_data(show_spinner=False)
def load_fp_ranks():
    """Load FantasyPros Dynasty SF PPR rankings (local CSV or scrape)."""
    import os

    paths = ["data/player_ranks.csv", "player_ranks.csv"]
    df = None
    for path in paths:
        if os.path.exists(path):
            try:
                df = pd.read_csv(path)
                break
            except Exception:
                df = None

    if df is None:
        url = "https://www.fantasypros.com/nfl/rankings/dynasty-superflex.php"
        resp = requests.get(url, timeout=20)
        resp.raise_for_status()
        tables = pd.read_html(resp.text)
        if not tables:
            raise RuntimeError("Could not read FantasyPros table.")
        df = tables[0]

    # Normalize columns: want Player, Pos, Rank
    cols = list(df.columns)

    # Player / Name column
    player_col = None
    for c in cols:
        cl = c.lower()
        if "player" in cl or "name" in cl:
            player_col = c
            break
    if player_col is None:
        player_col = cols[0]

    # Position
    pos_col = None
    for c in cols:
        cl = c.lower()
        if cl.startswith("pos") or "position" in cl:
            pos_col = c
            break

    # Rank
    rank_col = None
    for c in cols:
        cl = c.lower()
        if cl.startswith("rank") or cl == "rk" or cl == "#":
            rank_col = c
            break
    if rank_col is None:
        rank_col = cols[0]

    out = pd.DataFrame()
    out["Player"] = df[player_col].astype(str).str.strip()
    if pos_col:
        out["Pos"] = df[pos_col].astype(str).str.upper().str.strip()
    else:
        out["Pos"] = "WR"
    out["Rank"] = pd.to_numeric(df[rank_col], errors="coerce")

    out = out.dropna(subset=["Rank"])
    out = out[out["Pos"].isin(["QB", "RB", "WR", "TE"])]
    out = out.reset_index(drop=True)
    return out


# ====================================================
# Sidebar: settings & modifiers
# ====================================================

st.sidebar.header("Settings")

use_live = st.sidebar.checkbox(
    "Use live Sleeper + FantasyPros data",
    value=True,
    help="On: rosters/records from Sleeper API, rankings from FantasyPros site.",
)

league_id = st.sidebar.text_input(
    "Sleeper League ID",
    value="1194681871141023744",
    help="Your Sleeper league ID (from league settings).",
)

st.sidebar.caption("You can still upload CSVs below to override or test things manually.")
up_players = st.sidebar.file_uploader(
    "Custom Player Ranks CSV (Player, Pos, Rank)", type=["csv"]
)
up_rosters = st.sidebar.file_uploader(
    "Custom Rosters CSV (Team, Player, Pos optional)", type=["csv"]
)

st.sidebar.header("Player value tuning")
st.sidebar.caption("If you're unsure, just leave these sliders at their defaults.")

ELITE_GAP = st.sidebar.slider(
    "How valuable is the #1 overall player?",
    800.0,
    2200.0,
    1500.0,
    50.0,
)

RANK_IMPORTANCE = st.sidebar.slider(
    "How much do rankings matter?",
    0.015,
    0.060,
    0.038,
    0.001,
)

NEED_WEIGHT = st.sidebar.slider(
    "How much do roster needs matter?",
    0.0,
    0.6,
    0.20,
    0.05,
)

PACKAGE_PENALTY = st.sidebar.slider(
    "Quantity vs quality (2-for-1 tax)",
    0.0,
    1.0,
    0.75,
    0.05,
)

RISK_PREF = st.sidebar.slider(
    "Youth / upside vs safety",
    0.0,
    1.0,
    0.5,
    0.05,
)

st.sidebar.header("Pick tuning (optional)")
with st.sidebar.expander("Show pick value sliders"):
    N_STRENGTH = st.sidebar.slider(
        "Team strength depth (how many best players define team strength)",
        6,
        20,
        10,
        1,
    )
    PICK_MAX = st.sidebar.slider(
        "Pick max (best possible 1st-round pick)",
        250.0,
        900.0,
        500.0,
        25.0,
    )
    PICK_SLOT_DECAY = st.sidebar.slider(
        "How quickly picks lose value within a round",
        0.08,
        0.35,
        0.20,
        0.01,
    )
    R2_SCALE = st.sidebar.slider("Round 2 value vs Round 1", 0.20, 0.60, 0.40, 0.01)
    R3_SCALE = st.sidebar.slider("Round 3 value vs Round 1", 0.08, 0.35, 0.20, 0.01)
    R4_SCALE = st.sidebar.slider("Round 4 value vs Round 1", 0.03, 0.25, 0.10, 0.01)
    YEAR2_DISC = st.sidebar.slider(
        "Discount for next-year picks", 0.70, 0.95, 0.85, 0.01
    )
    YEAR3_DISC = st.sidebar.slider(
        "Discount for picks 2+ years out", 0.50, 0.90, 0.70, 0.01
    )

manual = st.sidebar.checkbox("Manual mode (click button to recalc)", value=False)
recalc = True
if manual:
    recalc = st.sidebar.button("Recalculate now")

# ====================================================
# Data loading & preparation
# ====================================================

DEFAULT_TARGETS = {"QB": 2, "RB": 4, "WR": 5, "TE": 2}
DEFAULT_POSMULT = {"QB": 1.10, "RB": 1.00, "WR": 1.00, "TE": 0.95}

picks_by_team = {}
pick_label_to_original = {}
future_pick_years = []

# ---- Live mode ----
if use_live and league_id.strip():
    try:
        with st.spinner("Loading Sleeper league data + FantasyPros rankings..."):
            (
                rosters_live,
                records_df,
                picks_by_team,
                pick_label_to_original,
                future_pick_years,
            ) = load_sleeper_league(league_id.strip())

            if up_players is not None:
                fp_ranks = pd.read_csv(up_players)
            else:
                fp_ranks = load_fp_ranks()

            fp_ranks["Pos"] = fp_ranks["Pos"].astype(str).str.upper().str.strip()
            fp_ranks["Rank"] = pd.to_numeric(fp_ranks["Rank"], errors="coerce")
            fp_ranks = fp_ranks.dropna(subset=["Rank"]).reset_index(drop=True)
            fp_ranks["Norm"] = fp_ranks["Player"].apply(normalize_name)
            fp_max_rank = int(fp_ranks["Rank"].max())

            roster_enriched = rosters_live.copy()
            roster_enriched["Norm"] = roster_enriched["Player"].apply(normalize_name)
            roster_enriched = roster_enriched.merge(
                fp_ranks[["Norm", "Rank", "Pos"]].rename(columns={"Pos": "FP_Pos"}),
                on="Norm",
                how="left",
            )
            roster_enriched["Pos_use"] = roster_enriched["FP_Pos"].fillna(
                roster_enriched["Pos"]
            )
            roster_enriched["Rank_use"] = roster_enriched["Rank"].fillna(
                fp_max_rank + 80
            )

            players_df = (
                roster_enriched[["Player", "Pos_use", "Rank_use"]]
                .drop_duplicates("Player")
                .rename(columns={"Pos_use": "Pos", "Rank_use": "Rank"})
            )
            rosters_df = roster_enriched[["Team", "Player"]].copy()

        st.success("Loaded live Sleeper rosters + FantasyPros rankings.")
    except Exception as e:
        st.error(f"Live data load failed: {e}")
        st.info("Falling back to CSV uploads (if provided).")
        use_live = False
else:
    use_live = False

# ---- Offline mode ----
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
        st.error("Please provide a player_ranks CSV (Player, Pos, Rank) or enable live mode.")
        st.stop()
    if rosters_df.empty or not {"Team", "Player"}.issubset(rosters_df.columns):
        st.error("Please provide a rosters CSV (Team, Player) or enable live mode.")
        st.stop()

    players_df["Pos"] = players_df["Pos"].astype(str).str.upper().str.strip()
    players_df["Rank"] = pd.to_numeric(players_df["Rank"], errors="coerce")
    players_df = players_df.dropna(subset=["Rank"]).reset_index(drop=True)

    rosters_df["Team"] = rosters_df["Team"].astype(str).str.strip()
    rosters_df["Player"] = rosters_df["Player"].astype(str).str.strip()

    records_df = (
        rosters_df[["Team"]].drop_duplicates().assign(Wins=0, Losses=0, Ties=0)
    )

    current_year = datetime.now().year
    future_pick_years = [current_year + i for i in (1, 2, 3)]
    picks_by_team = {}
    pick_label_to_original = {}
    for tm in rosters_df["Team"].unique():
        for yr in future_pick_years:
            for rnd in (1, 2, 3, 4):
                label = f"{yr} R{rnd} ({tm})"
                picks_by_team.setdefault(tm, []).append(label)
                pick_label_to_original[label] = tm
    for tm in picks_by_team:
        picks_by_team[tm] = sorted(picks_by_team[tm])

# Logos
league_logo = None
team_logo_map = {}
if league_id.strip():
    try:
        league_logo, team_logo_map = load_sleeper_logos(league_id.strip())
    except Exception:
        league_logo, team_logo_map = None, {}

# ====================================================
# Core valuation logic (PPR curves, age, scarcity)
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

# Scarcity multiplier: positions with few top players get a small boost
scarcity_mult = {}
for pos in ["QB", "RB", "WR", "TE"]:
    c = counts.get(pos, 1)
    raw = avg_count / c
    raw_clamped = max(0.5, min(2.0, raw))
    scarcity = 0.85 + (raw_clamped - 0.5) * (1.15 - 0.85) / (2.0 - 0.5)
    scarcity_mult[pos] = scarcity

DEFAULT_POSMULT = {"QB": 1.10, "RB": 1.00, "WR": 1.00, "TE": 0.95}

posmult_effective = {}
for pos in ["QB", "RB", "WR", "TE"]:
    base_mult = DEFAULT_POSMULT.get(pos, 1.0)
    eff = base_mult * scarcity_mult.get(pos, 1.0)
    if pos == "TE":
        eff = min(eff, 0.85)  # TEs kept safely below similarly ranked WR/QB
    posmult_effective[pos] = eff

players_df["PosMult"] = players_df["Pos"].map(posmult_effective).fillna(1.0)

ppr_curves = load_ppr_curves()
ages_table = load_age_table()

players_df["Norm"] = players_df["Player"].apply(normalize_name)
if ages_table is not None:
    ages_merge = ages_table[["Norm", "Age"]].copy()
    players_df = players_df.merge(ages_merge, on="Norm", how="left")
else:
    players_df["Age"] = np.nan

AGE_CFG = {
    "QB": (27, 33, 0.80, 0.06),
    "RB": (24, 27, 0.55, 0.10),
    "WR": (26, 29, 0.65, 0.08),
    "TE": (27, 30, 0.70, 0.06),
}

def age_mult_for_player(pos, age):
    if pd.isna(age):
        return 1.0
    pos = str(pos).upper()
    peak, decline_start, floor_mult, young_boost = AGE_CFG.get(
        pos, (26, 30, 0.7, 0.06)
    )
    age = float(age)
    if age <= peak:
        span = 6.0
        factor = (peak - age) / span
        factor = max(0.0, min(1.0, factor))
        return 1.0 + young_boost * factor
    if age <= decline_start:
        return 1.0
    years_over = age - decline_start
    decline_span = 6.0
    per_year = (1.0 - floor_mult) / decline_span
    mult = 1.0 - per_year * years_over
    return max(floor_mult, mult)

def risk_mult_for_player(pos, age):
    if pd.isna(age):
        return 1.0
    pos = str(pos).upper()
    peak, _, _, _ = AGE_CFG.get(pos, (26, 30, 0.7, 0.06))
    age = float(age)
    youth_score = (peak - age) / 8.0
    youth_score = max(-1.0, min(1.0, youth_score))
    factor = 1.0 + (RISK_PREF - 0.5) * 0.2 * youth_score
    return float(max(0.9, min(1.1, factor)))

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
    players_df["BaseValueRaw"] = (
        ELITE_GAP * np.exp(-RANK_IMPORTANCE * (players_df["Rank"] - 1))
    ).round(2)
else:
    rel_pts = (players_df["ModelPoints"] / max_pts).clip(0.0001, 1.0)
    curve_power = 1.6 + (RANK_IMPORTANCE - 0.015) * 30.0
    base_curve = np.power(rel_pts, curve_power)

    overall_rank = players_df["Rank"].rank(method="first")
    tier_mult_raw = np.where(
        overall_rank <= 12,
        1.30,
        np.where(overall_rank <= 24, 1.18, np.where(overall_rank <= 48, 1.08, 1.00)),
    )
    pos_sc = players_df["Pos"].map(scarcity_mult).fillna(1.0).values
    tier_mult = tier_mult_raw * (1.0 + 0.35 * (pos_sc - 1.0))

    players_df["BaseValueRaw"] = (ELITE_GAP * base_curve * tier_mult).round(2)

if "Age" in players_df.columns:
    players_df["AgeMult"] = players_df.apply(
        lambda r: age_mult_for_player(r["Pos"], r["Age"]), axis=1
    )
    players_df["RiskMult"] = players_df.apply(
        lambda r: risk_mult_for_player(r["Pos"], r["Age"]), axis=1
    )
else:
    players_df["AgeMult"] = 1.0
    players_df["RiskMult"] = 1.0

players_df["BaseValue"] = (
    players_df["BaseValueRaw"]
    * players_df["PosMult"]
    * players_df["AgeMult"]
    * players_df["RiskMult"]
).round(2)

# --------- Monotonic smoothing by overall rank ----------
players_df = players_df.sort_values("Rank").reset_index(drop=True)
vals = players_df["BaseValue"].to_numpy()
for i in range(1, len(vals)):
    if vals[i] > vals[i - 1]:
        vals[i] = vals[i - 1] * 0.999
players_df["BaseValue"] = vals

# --------- Tier groups (for Trade Finder filters) ----------
def tier_group(rank: float) -> str:
    """Map overall FP rank to High-Tier / Starter / Flex / Depth."""
    if rank <= 36:
        return "High-Tier"
    elif rank <= 96:
        return "Starter"
    elif rank <= 144:
        return "Flex"
    else:
        return "Depth"

players_df["TierGroup"] = players_df["Rank"].apply(tier_group)
players_df["IsYoung"] = players_df["Age"].apply(
    lambda a: bool(not pd.isna(a) and a <= 26)
)

# Team ages for pick context
team_avg_age = {}
league_avg_age = None
if load_age_table() is not None and not rosters_df.empty:
    ages_norm = load_age_table().copy()
    roster_with_norm = rosters_df.copy()
    roster_with_norm["Norm"] = roster_with_norm["Player"].apply(normalize_name)
    ages_norm = ages_norm[["Norm", "Age"]]
    roster_with_norm = roster_with_norm.merge(ages_norm, on="Norm", how="left")
    grp = roster_with_norm.groupby("Team")["Age"].mean()
    team_avg_age = grp.to_dict()
    if len(team_avg_age):
        league_avg_age = float(np.nanmean(list(team_avg_age.values())))

# ====================================================
# Helper functions for values & picks
# ====================================================

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

def apply_need(pos, counts):
    base_mult = need_multiplier(counts.get(pos, 0), DEFAULT_TARGETS.get(pos, 0))
    return 1.0 + NEED_WEIGHT * (base_mult - 1.0)

def roster_players(team: str):
    names = rosters_df.loc[rosters_df["Team"] == team, "Player"].tolist()
    name_set = set(players_df["Player"])
    names = [n for n in names if n in name_set]
    names.sort(key=lambda x: x.lower())
    return names

label_map = {
    row["Player"]: f'{row["Player"]} ({row["Pos"]} #{int(row["Rank"])})'
    for _, row in players_df.iterrows()
}

def team_strength(team: str):
    names = rosters_df.loc[rosters_df["Team"] == team, "Player"].tolist()
    sub = players_df[players_df["Player"].isin(names)].copy()
    vals = sub["BaseValue"].sort_values(ascending=False).head(10)
    return float(vals.sum()) if len(vals) else 0.0

team_list = sorted(rosters_df["Team"].unique().tolist())
strengths = {t: team_strength(t) for t in team_list}

records_df = records_df.set_index("Team")

def get_record(team: str):
    if team in records_df.index:
        row = records_df.loc[team]
        return float(row.get("Wins", 0)), float(row.get("Losses", 0))
    return 0.0, 0.0

# sort for pick slots: worst record + weakest strength get best picks
sorted_for_picks = sorted(
    team_list,
    key=lambda t: (get_record(t)[0], strengths[t]),
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
    slot_val = math.exp(-PICK_SLOT_DECAY * (slot - 1))
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

    age_mult_pick = 1.0
    if league_avg_age is not None and original_team in team_avg_age:
        delta_age = team_avg_age[original_team] - league_avg_age
        delta_age = max(-3.0, min(3.0, delta_age))
        age_mult_pick = 1.0 + 0.02 * delta_age

    inv_mult = 1.0
    if avg_num_picks > 0 and original_team in original_pick_counts:
        delta_picks = avg_num_picks - original_pick_counts[original_team]
        delta_picks = max(-3.0, min(3.0, delta_picks))
        inv_mult = 1.0 + 0.01 * delta_picks

    return round(base * year_mult * age_mult_pick * inv_mult, 1)

def build_pick_labels_for_team(team: str):
    return picks_by_team.get(team, [])

def parse_pick_label(label: str):
    try:
        parts = label.split()
        yr = int(parts[0])
        rnd = int(parts[1].replace("R", ""))
        original_team = label[label.find("(") + 1 : label.find(")")]
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

def sum_players_value(player_list, counts_for_need):
    vals = []
    details = []
    for n in player_list:
        r = players_df.loc[players_df["Player"] == n]
        if r.empty:
            continue
        pos = r["Pos"].iloc[0]
        rank = int(r["Rank"].iloc[0])
        base_before_need = float(r["BaseValue"].iloc[0])
        mult = apply_need(pos, counts_for_need)
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

def build_needs_summary(team, before_counts, after_counts, incoming_details):
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
                    f"{team} badly needs a starting QB (goes from {before} to {after})."
                )
            else:
                notes.append(
                    f"{team} was a bit light at {pos} and adds {inc} player(s) there."
                )
        elif inc > 0 and before >= tgt + 2:
            notes.append(
                f"{team} already had strong depth at {pos}, so these additions are more luxury than necessity."
            )

    return notes

# Helper: max reasonable pick round for a given FP rank
def max_pick_round_for_rank(rank: float) -> int:
    """
    Max *good* pick round you'd reasonably send straight up for this rank.
    Lower number = better pick. We disallow offering picks better than this.
    """
    if rank > 300:
        # Really deep — 4th at most
        return 4
    elif rank > 240:
        # Late/very deep bench — 3rd at most
        return 3
    elif rank > 150:
        # Deeper flex/bench — 2nd at most
        return 2
    else:
        # Starter/top-ish — 1sts allowed
        return 1

# ====================================================
# Trade evaluation text helpers
# ====================================================

def trade_category_and_blurb(side_a_value, side_b_value, teamA, teamB, A_desc, B_desc):
    if side_a_value <= 0 and side_b_value <= 0:
        return "No trade", "No assets selected yet."

    diff = side_a_value - side_b_value
    bigger = max(side_a_value, side_b_value, 1.0)
    pct = abs(diff) / bigger

    if pct < 0.07:
        cat = "Perfect Fit"
    elif pct < 0.18:
        cat = "Reasonable"
    elif pct < 0.30:
        cat = "Questionable"
    elif pct < 0.45:
        cat = "Not Good"
    else:
        cat = "Call the Commissioner if They Accept"

    if abs(diff) < 1e-6:
        summary = (
            "This deal is extremely close in total value. Most managers would see it as pretty fair "
            "once you factor in rankings, positions, and small team-need tweaks."
        )
    elif diff > 0:
        winner, loser = teamA, teamB
        margin_side = side_a_value - side_b_value
        summary = (
            f"This trade likely leans toward **{winner}**, by about **{margin_side:,.0f} value "
            f"points (~{pct*100:.1f}% more value)** than {loser}. "
            "In practice, that usually means the side getting more value is consolidating slightly better-ranked pieces "
            "or a bit more future flexibility."
        )
    else:
        winner, loser = teamB, teamA
        margin_side = side_b_value - side_a_value
        summary = (
            f"This trade likely leans toward **{winner}**, by about **{margin_side:,.0f} value "
            f"points (~{pct*100:.1f}% more value)** than {loser}. "
            "It's the kind of deal many leagues would still allow, but some managers might push back if they see "
            "one side stacking more of the higher-ranked assets."
        )

    extra = (
        f"\n\n**{teamA} receives:** {A_desc}\n\n"
        f"**{teamB} receives:** {B_desc}\n\n"
        "From a big-picture standpoint, think in terms of who is gaining (or giving up) more of the better-ranked, "
        "core pieces versus depth or future dart throws. The numbers help frame that, but each manager's timeline and "
        "risk tolerance will tilt things one way or the other."
    )

    return cat, summary + extra

def format_player_detail_table(details):
    if not details:
        return pd.DataFrame(columns=["Player", "Pos", "FP Rank", "Base", "Need x", "Final"])
    df = pd.DataFrame(
        details,
        columns=["Player", "Pos", "FP Rank", "Base", "Need Mult", "Final Value"],
    )
    df["Base"] = df["Base"].round(1)
    df["Need Mult"] = df["Need Mult"].round(3)
    df["Final Value"] = df["Final Value"].round(1)
    return df

def format_pick_detail_table(details):
    if not details:
        return pd.DataFrame(columns=["Pick", "Value"])
    df = pd.DataFrame(details, columns=["Pick", "Value"])
    df["Value"] = df["Value"].round(1)
    return df

def player_rank_blurb(name: str):
    r = players_df.loc[players_df["Player"] == name]
    if r.empty:
        return None
    pos = r["Pos"].iloc[0]
    rk = int(r["Rank"].iloc[0])
    tier = r["TierGroup"].iloc[0]
    return (
        f"{name} is around overall **#{rk}** as a **{pos} ({tier})** "
        "in the FantasyPros dynasty superflex rankings."
    )

def build_suggestion_explanation(
    send_players, send_picks, recv_players, recv_picks, gap_pct
):
    lines = []

    main_recv = recv_players[0] if recv_players else None
    main_send = send_players[0] if send_players else None

    if main_recv and main_send:
        recv_blurb = player_rank_blurb(main_recv)
        send_blurb = player_rank_blurb(main_send)
        if recv_blurb and send_blurb:
            lines.append(f"{recv_blurb} In return, you're sending {send_blurb}")
        elif recv_blurb:
            lines.append(recv_blurb)
        elif send_blurb:
            lines.append(send_blurb)
    elif main_recv:
        recv_blurb = player_rank_blurb(main_recv)
        if recv_blurb:
            lines.append(recv_blurb)

    if recv_picks:
        pick_descriptions = []
        for lbl in recv_picks:
            yr, rnd, orig = parse_pick_label(lbl)
            if yr is None:
                continue
            pick_descriptions.append(
                f"{lbl} (a {yr} round {rnd} pick originally belonging to {orig})"
            )
        if pick_descriptions:
            lines.append(
                "You're also getting some draft capital: "
                + ", ".join(pick_descriptions)
                + "."
            )

    if send_picks:
        pick_descriptions = []
        for lbl in send_picks:
            yr, rnd, orig = parse_pick_label(lbl)
            if yr is None:
                continue
            pick_descriptions.append(f"{lbl} (originally from {orig})")
        if pick_descriptions:
            lines.append(
                "In exchange, you're including picks like "
                + ", ".join(pick_descriptions)
                + "."
            )

    lines.append(
        f"Overall, the model sees the two sides as being within about **{gap_pct*100:.1f}%** of each other in value, "
        "based on those FantasyPros rankings and the way future picks are discounted by year and projected finish."
    )

    return " ".join(lines)

# ====================================================
# Trade finder helpers
# ====================================================

def build_asset_lists_for_team(team):
    players = roster_players(team)
    player_vals = []
    for n in players:
        r = players_df.loc[players_df["Player"] == n]
        if r.empty:
            continue
        player_vals.append(
            (n, r["Pos"].iloc[0], int(r["Rank"].iloc[0]), float(r["BaseValue"].iloc[0]))
        )
    player_vals.sort(key=lambda x: x[3], reverse=True)

    pick_labels = build_pick_labels_for_team(team)
    pick_vals = [(lbl, label_value(lbl)) for lbl in pick_labels]
    pick_vals.sort(key=lambda x: x[1], reverse=True)

    return player_vals, pick_vals

def matches_position_type(player_name: str, selection: str) -> bool:
    """
    Filter by High-Tier / Starter / Flex / Depth / Young.

    If we *do* have age data, "Young" means Age <= 26.
    If we *don't* have reliable age data (all NaN), we fall back to
    using overall rank as a proxy for "young-ish, future-friendly assets".
    """
    if selection == "Any":
        return True
    r = players_df.loc[players_df["Player"] == player_name]
    if r.empty:
        return True

    tier = r["TierGroup"].iloc[0]
    is_young_flag = bool(r["IsYoung"].iloc[0]) if "IsYoung" in r.columns else False

    if selection == "Young":
        # If we actually have age data, use it
        if "IsYoung" in players_df.columns and players_df["IsYoung"].any():
            return is_young_flag
        # Fallback: treat reasonably high-ranked players as "young assets"
        rank_val = float(r["Rank"].iloc[0])
        return rank_val <= 120.0
    else:
        return tier == selection

def find_offer_for_target_simple(my_team, target_team, target_player_name, tolerance_pct=0.25):
    """Single-player target: only simple player/pick combos with rank-aware pick caps."""
    r = players_df.loc[players_df["Player"] == target_player_name]
    if r.empty:
        return []
    target_val = float(r["BaseValue"].iloc[0])
    target_rank = float(r["Rank"].iloc[0])
    target_tier = r["TierGroup"].iloc[0]

    if target_val <= 0:
        return []
    my_players, my_picks = build_asset_lists_for_team(my_team)

    combos = []

    # Single players
    for p in my_players[:10]:
        combos.append(([p[0]], []))

    # Single picks
    for pk in my_picks[:6]:
        combos.append(([], [pk[0]]))

    # Player + pick combos
    for p in my_players[:6]:
        for pk in my_picks[:4]:
            combos.append(([p[0]], [pk[0]]))

    # Two-player combos (no picks) to make depth-for-depth/young deals easier
    for i in range(min(6, len(my_players))):
        for j in range(i + 1, min(9, len(my_players))):
            combos.append(([my_players[i][0], my_players[j][0]], []))

    suggestions = []
    for give_players, give_picks in combos:
        vals_p, _ = sum_players_value(give_players, team_pos_counts(my_team))
        vals_pk, _ = sum_picks_value(give_picks)
        total = vals_p + vals_pk
        if total <= 0:
            continue

        # Rank-aware cap on how good a pick you can send for this target
        offered_rounds = []
        for lbl in give_picks:
            yr, rnd, orig = parse_pick_label(lbl)
            if rnd is not None:
                offered_rounds.append(rnd)
        if offered_rounds:
            best_round = min(offered_rounds)  # 1 is best, 4 worst
            max_round_allowed = max_pick_round_for_rank(target_rank)
            if best_round < max_round_allowed:
                # e.g. target rank 320 => max_round_allowed=4, but you're offering R2 -> skip
                continue

        # Extra protection: don't use a premium pick for pure depth
        send_contains_first = any(
            parse_pick_label(lbl)[1] == 1 for lbl in give_picks if parse_pick_label(lbl)[1] is not None
        )
        if send_contains_first and target_tier == "Depth":
            continue

        pct_diff = abs(total - target_val) / max(target_val, total)
        if pct_diff <= tolerance_pct:
            suggestions.append((give_players, give_picks, total, pct_diff))

    suggestions.sort(key=lambda x: x[3])
    return suggestions[:1]

def find_offer_for_multi_target_simple(
    my_team, target_team, target_player_names, tolerance_pct=0.28
):
    """Multi-player target: used when user selects multiple positions (RB+WR, etc.)."""
    target_vals = []
    ranks = []
    tiers = []
    for name in target_player_names:
        r = players_df.loc[players_df["Player"] == name]
        if r.empty:
            return []
        target_vals.append(float(r["BaseValue"].iloc[0]))
        ranks.append(float(r["Rank"].iloc[0]))
        tiers.append(r["TierGroup"].iloc[0])
    target_total = sum(target_vals)
    if target_total <= 0:
        return []

    worst_rank = max(ranks)  # lowest-quality among targets

    my_players, my_picks = build_asset_lists_for_team(my_team)
    combos = []

    # single player
    for p in my_players[:10]:
        combos.append(([p[0]], []))

    # two players
    for i in range(min(6, len(my_players))):
        for j in range(i + 1, min(9, len(my_players))):
            combos.append(([my_players[i][0], my_players[j][0]], []))

    # player + pick
    for p in my_players[:6]:
        for pk in my_picks[:4]:
            combos.append(([p[0]], [pk[0]]))

    suggestions = []
    for give_players, give_picks in combos:
        vals_p, _ = sum_players_value(give_players, team_pos_counts(my_team))
        vals_pk, _ = sum_picks_value(give_picks)
        total = vals_p + vals_pk
        if total <= 0:
            continue

        # Rank-aware cap on pick quality vs worst target rank
        offered_rounds = []
        for lbl in give_picks:
            yr, rnd, orig = parse_pick_label(lbl)
            if rnd is not None:
                offered_rounds.append(rnd)
        if offered_rounds:
            best_round = min(offered_rounds)
            max_round_allowed = max_pick_round_for_rank(worst_rank)
            if best_round < max_round_allowed:
                continue

        pct_diff = abs(total - target_total) / max(target_total, total)
        if pct_diff <= tolerance_pct:
            suggestions.append((give_players, give_picks, total, pct_diff))

    suggestions.sort(key=lambda x: x[3])
    return suggestions[:1]

def find_return_for_package(
    my_team,
    other_team,
    outgoing_players,
    outgoing_picks,
    tolerance_pct=0.18,
):
    """Given what I'm sending, find packages from the other team that roughly match value."""
    counts_my = team_pos_counts(my_team)
    val_p, _ = sum_players_value(outgoing_players, counts_my)
    val_pk, _ = sum_picks_value(outgoing_picks)
    target_total = val_p + val_pk
    if target_total <= 0:
        return []

    players_other, picks_other = build_asset_lists_for_team(other_team)

    combos = []

    # single player
    for p in players_other[:10]:
        combos.append(([p[0]], []))

    # player + pick
    for p in players_other[:6]:
        for pk in picks_other[:4]:
            combos.append(([p[0]], [pk[0]]))

    suggestions = []
    for recv_players, recv_picks in combos:
        counts_after = counts_after_trade(other_team, [], recv_players)
        val_p_other, _ = sum_players_value(recv_players, counts_after)
        val_pk_other, _ = sum_picks_value(recv_picks)
        total = val_p_other + val_pk_other
        if total <= 0:
            continue
        pct_diff = abs(total - target_total) / max(target_total, total)
        if pct_diff <= tolerance_pct:
            suggestions.append((recv_players, recv_picks, total, pct_diff))

    suggestions.sort(key=lambda x: x[3])
    return suggestions[:3]

# ====================================================
# Header layout (sleeker)
# ====================================================

header_cols = st.columns([1, 5])
with header_cols[0]:
    if league_logo:
        st.image(league_logo, width=72)
with header_cols[1]:
    st.markdown("## Dynasty Bros. Trade Calculator")
    st.caption("Dynasty Superflex PPR trade helper powered by FantasyPros & Sleeper.")

st.markdown("---")

if rosters_df.empty:
    st.error("No rosters found. Check your Sleeper league ID or upload a rosters file.")
    st.stop()

# Random default teams for dropdowns
if team_list:
    if len(team_list) >= 2:
        default_teamA = random.choice(team_list)
        remaining = [t for t in team_list if t != default_teamA]
        default_teamB = random.choice(remaining) if remaining else default_teamA
    else:
        default_teamA = default_teamB = team_list[0]
else:
    default_teamA = default_teamB = None

tab_calc, tab_finder = st.tabs(["🔁 Trade Calculator", "🧭 Trade Finder"])

# -------------------- Trade Calculator --------------------
with tab_calc:
    st.subheader("Trade Calculator")

    colA, colB = st.columns(2)
    with colA:
        teamA = st.selectbox(
            "Select first team here",
            team_list,
            index=team_list.index(default_teamA) if default_teamA in team_list else 0,
            key="teamA_calc",
        )
        if teamA in team_logo_map:
            st.image(team_logo_map[teamA], width=56)
    with colB:
        teamB = st.selectbox(
            "Select second team here",
            team_list,
            index=team_list.index(default_teamB) if default_teamB in team_list else 0,
            key="teamB_calc",
        )
        if teamB in team_logo_map:
            st.image(team_logo_map[teamB], width=56)

    if teamA == teamB:
        st.warning("Select two different teams to evaluate a trade.")
    else:
        a_players = roster_players(teamA)
        b_players = roster_players(teamB)

        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f"**{teamA} sends**")
            a_send_players = st.multiselect(
                "Players to send",
                a_players,
                format_func=lambda x: label_map.get(x, x),
                key="a_send_players",
            )
            a_send_picks = st.multiselect(
                "Picks to send",
                build_pick_labels_for_team(teamA),
                key="a_send_picks",
            )
        with col2:
            st.markdown(f"**{teamB} sends**")
            b_send_players = st.multiselect(
                "Players to send",
                b_players,
                format_func=lambda x: label_map.get(x, x),
                key="b_send_players",
            )
            b_send_picks = st.multiselect(
                "Picks to send",
                build_pick_labels_for_team(teamB),
                key="b_send_picks",
            )

        if recalc:
            countsA_before = team_pos_counts(teamA)
            countsB_before = team_pos_counts(teamB)

            countsA_after = counts_after_trade(teamA, a_send_players, b_send_players)
            countsB_after = counts_after_trade(teamB, b_send_players, a_send_players)

            # Values for incoming assets, using AFTER-trade needs
            A_in_val_players, A_in_details = sum_players_value(
                b_send_players, countsA_after
            )
            A_in_val_picks, A_in_pick_details = sum_picks_value(b_send_picks)
            A_in_total = A_in_val_players + A_in_val_picks

            B_in_val_players, B_in_details = sum_players_value(
                a_send_players, countsB_after
            )
            B_in_val_picks, B_in_pick_details = sum_picks_value(a_send_picks)
            B_in_total = B_in_val_players + B_in_val_picks

            # Values for outgoing assets (for context)
            A_out_val_players, _ = sum_players_value(a_send_players, countsA_before)
            A_out_val_picks, _ = sum_picks_value(a_send_picks)
            A_out_total = A_out_val_players + A_out_val_picks

            B_out_val_players, _ = sum_players_value(b_send_players, countsB_before)
            B_out_val_picks, _ = sum_picks_value(b_send_picks)
            B_out_total = B_out_val_players + B_out_val_picks

            m1, m2, m3, m4 = st.columns(4)
            m1.metric(f"{teamA} gives", f"{A_out_total:,.0f}")
            m2.metric(f"{teamA} receives", f"{A_in_total:,.0f}")
            m3.metric(f"{teamB} gives", f"{B_out_total:,.0f}")
            m4.metric(f"{teamB} receives", f"{B_in_total:,.0f}")

            st.markdown("---")
            st.subheader("Analysis")

            with st.expander("What do these categories mean?"):
                st.markdown(
                    """
- **Perfect Fit** – Extremely close in value, and the swap lines up well with what each roster needs. These are the rare deals almost everyone calls fair.
- **Reasonable** – Slight lean to one side, but still the kind of offer that a reasonable manager might accept without much drama.
- **Questionable** – Value gap is noticeable. Not impossible, but it usually requires one manager to be higher or lower on specific players than the general market.
- **Not Good** – Clearly lopsided. It might happen, but it would raise eyebrows and could spark league chatter.
- **Call the Commissioner if They Accept** – Wildly uneven. If this goes through, most leagues at least talk about vetoing it.
"""
                )

            A_desc_players = ", ".join(
                f"{n} ({pos} #{rk})" for (n, pos, rk, _, _, _) in A_in_details
            ) or "no players"
            A_desc_picks = (
                ", ".join(lbl for lbl, _ in A_in_pick_details) or "no picks"
            )
            A_desc = f"{A_desc_players}; picks: {A_desc_picks}"

            B_desc_players = ", ".join(
                f"{n} ({pos} #{rk})" for (n, pos, rk, _, _, _) in B_in_details
            ) or "no players"
            B_desc_picks = (
                ", ".join(lbl for lbl, _ in B_in_pick_details) or "no picks"
            )
            B_desc = f"{B_desc_players}; picks: {B_desc_picks}"

            cat, summary = trade_category_and_blurb(
                A_in_total, B_in_total, teamA, teamB, A_desc, B_desc
            )

            st.markdown(f"**Category:** {cat}")
            st.write(summary)

            notesA = build_needs_summary(teamA, countsA_before, countsA_after, A_in_details)
            notesB = build_needs_summary(teamB, countsB_before, countsB_after, B_in_details)

            with st.expander("Full breakdown: roster context & player details", expanded=True):
                st.markdown(f"### {teamA} receives")
                if notesA:
                    for n in notesA:
                        st.write("- " + n)
                else:
                    st.write("- No dramatic positional shifts; mostly a rankings/value decision.")
                dfA = format_player_detail_table(A_in_details)
                st.dataframe(
                    dfA,
                    use_container_width=True,
                    height=min(400, 40 + 30 * max(3, len(dfA))),
                )
                if A_in_pick_details:
                    st.write("**Incoming picks**")
                    st.dataframe(
                        format_pick_detail_table(A_in_pick_details),
                        use_container_width=True,
                        height=min(300, 40 + 30 * len(A_in_pick_details)),
                    )

                st.markdown("---")

                st.markdown(f"### {teamB} receives")
                if notesB:
                    for n in notesB:
                        st.write("- " + n)
                else:
                    st.write("- No dramatic positional shifts; mostly a rankings/value decision.")
                dfB = format_player_detail_table(B_in_details)
                st.dataframe(
                    dfB,
                    use_container_width=True,
                    height=min(400, 40 + 30 * max(3, len(dfB))),
                )
                if B_in_pick_details:
                    st.write("**Incoming picks**")
                    st.dataframe(
                        format_pick_detail_table(B_in_pick_details),
                        use_container_width=True,
                        height=min(300, 40 + 30 * len(B_in_pick_details)),
                    )

            st.markdown("#### Suggestions to improve or balance the deal")
            suggestions = []
            diff_val = A_in_total - B_in_total
            if abs(diff_val) > 0:
                if diff_val > 0:
                    winner, loser = teamA, teamB
                else:
                    winner, loser = teamB, teamA
                suggestions.append(
                    f"If **{winner}** added a small future pick or a depth piece, "
                    f"this would move closer to something most managers would label as even."
                )
                suggestions.append(
                    "You can also tweak the sliders on the left if you personally "
                    "value youth, top-end options, or future picks a bit differently."
                )
            else:
                suggestions.append(
                    "The model sees this as very tight already. Minor changes in your personal rankings "
                    "or preferences could tilt it either way."
                )
            for s_text in suggestions:
                st.write("- " + s_text)

# -------------------- Trade Finder --------------------
with tab_finder:
    st.subheader("Trade Finder (beta)")

    finder_mode = st.radio(
        "What do you want help with?",
        ["Acquire position/picks", "Trade away player/picks"],
        horizontal=True,
    )

    if team_list:
        default_my_team = random.choice(team_list)
    else:
        default_my_team = None

    my_team = st.selectbox(
        "Your team",
        team_list,
        index=team_list.index(default_my_team) if default_my_team in team_list else 0,
        key="finder_my_team",
    )
    if my_team in team_logo_map:
        st.image(team_logo_map[my_team], width=56)

    other_teams = [t for t in team_list if t != my_team]

    if finder_mode == "Acquire position/picks":
        st.markdown("**1) What are you trying to acquire?**")
        target_positions = st.multiselect(
            "Positions (optional)",
            ["QB", "RB", "WR", "TE"],
            default=["RB"],  # start with RB only
        )

        position_type_choice = st.selectbox(
            "Position type (optional)",
            ["Any", "High-Tier", "Starter", "Flex", "Depth", "Young"],
            help=(
                "High-Tier: must-start level\n"
                "Starter: usually in the lineup\n"
                "Flex: could be in a flex spot\n"
                "Depth: bench/injury fill-in\n"
                "Young: age 26 or younger (independent of tier; or top ~120 overall if age data missing)"
            ),
        )

        target_pick_rounds = st.multiselect(
            "Draft pick rounds (optional)",
            [1, 2, 3, 4],
            help="Leave empty if you only care about players.",
        )

        if st.button("Suggest trade ideas", key="btn_acquire"):
            suggestions_all = []

            for ot in other_teams:
                players_other, picks_other = build_asset_lists_for_team(ot)

                # ----- PLAYER-BASED OFFERS -----
                if target_positions:
                    positions_chosen = list(target_positions)

                    if len(positions_chosen) == 1:
                        # Single position: behave like previous version
                        pos = positions_chosen[0]
                        cand_players = [
                            p
                            for p in players_other
                            if p[1] == pos and matches_position_type(p[0], position_type_choice)
                        ][:5]
                        for p in cand_players:
                            cand_target = p[0]
                            offers = find_offer_for_target_simple(my_team, ot, cand_target, tolerance_pct=0.25)
                            for give_players, give_picks, total, pct in offers:
                                suggestions_all.append(
                                    {
                                        "From": my_team,
                                        "To": ot,
                                        "You Get": [cand_target],
                                        "You Send players": give_players,
                                        "You Send picks": give_picks,
                                        "Approx value gap": pct,
                                    }
                                )
                    else:
                        # Multiple positions: want a bundle with one player from each selected position
                        pos_to_candidates = {}
                        for pos in positions_chosen:
                            cand = [
                                p for p in players_other
                                if p[1] == pos and matches_position_type(p[0], position_type_choice)
                            ]
                            if not cand:
                                pos_to_candidates = {}
                                break
                            pos_to_candidates[pos] = cand[:3]
                        if pos_to_candidates:
                            # We'll just take best candidate per pos as one bundle
                            bundle_names = [pos_to_candidates[pos][0][0] for pos in positions_chosen]
                            offers_multi = find_offer_for_multi_target_simple(
                                my_team, ot, bundle_names, tolerance_pct=0.28
                            )
                            for give_players, give_picks, total, pct in offers_multi:
                                suggestions_all.append(
                                    {
                                        "From": my_team,
                                        "To": ot,
                                        "You Get": bundle_names,
                                        "You Send players": give_players,
                                        "You Send picks": give_picks,
                                        "Approx value gap": pct,
                                    }
                                )

                # ----- PICK-BASED OFFERS -----
                if target_pick_rounds:
                    for lbl, val in picks_other[:10]:
                        yr, rnd, orig = parse_pick_label(lbl)
                        if rnd not in target_pick_rounds:
                            continue
                        target_val = val
                        my_players, my_picks = build_asset_lists_for_team(my_team)
                        combos = []
                        for pl in my_players[:8]:
                            combos.append(([pl[0]], []))
                        for pk in my_picks[:6]:
                            combos.append(([], [pk[0]]))
                        for pl in my_players[:4]:
                            for pk in my_picks[:4]:
                                combos.append(([pl[0]], [pk[0]]))

                        for give_players, give_picks in combos:
                            vp, _ = sum_players_value(
                                give_players, team_pos_counts(my_team)
                            )
                            vpk, _ = sum_picks_value(give_picks)
                            tot = vp + vpk
                            if tot <= 0:
                                continue
                            pct_gap = abs(tot - target_val) / max(tot, target_val)
                            if pct_gap <= 0.20:
                                suggestions_all.append(
                                    {
                                        "From": my_team,
                                        "To": ot,
                                        "You Get": [lbl],
                                        "You Send players": give_players,
                                        "You Send picks": give_picks,
                                        "Approx value gap": pct_gap,
                                    }
                                )
                                break  # one idea per pick

            if not suggestions_all:
                st.info(
                    "No obvious ideas within the current tolerance. "
                    "Try broadening positions, relaxing pick rounds, or adjusting sliders."
                )
            else:
                # Sort by closeness and limit duplicate targets
                suggestions_all.sort(key=lambda s: s["Approx value gap"])
                filtered = []
                target_counts = {}  # key = tuple(sorted(You Get)) -> how many times used

                for s in suggestions_all:
                    key = tuple(sorted(s["You Get"]))
                    count = target_counts.get(key, 0)
                    # At most 2 suggestions per unique target package
                    if count >= 2:
                        continue
                    filtered.append(s)
                    target_counts[key] = count + 1
                    if len(filtered) >= 3:
                        break

                st.markdown("### Suggested ideas")
                for i, sug in enumerate(filtered, start=1):
                    st.markdown(f"**Idea #{i}** — {sug['From']} ↔ {sug['To']}")
                    st.write(
                        f"{sug['From']} sends: "
                        f"{', '.join(sug['You Send players']) or 'no players'}; "
                        f"{', '.join(sug['You Send picks']) or 'no picks'}"
                    )
                    st.write(
                        f"{sug['From']} receives: "
                        f"{', '.join(sug['You Get'])}"
                    )

                    recv_players = [
                        asset
                        for asset in sug["You Get"]
                        if asset in set(players_df["Player"])
                    ]
                    recv_picks = [
                        asset for asset in sug["You Get"] if asset not in recv_players
                    ]
                    expl = build_suggestion_explanation(
                        sug["You Send players"],
                        sug["You Send picks"],
                        recv_players,
                        recv_picks,
                        sug["Approx value gap"],
                    )
                    st.write(expl)
                    st.markdown("---")

    else:  # Trade away mode
        st.markdown("**1) What are you open to trading away?**")
        my_players = roster_players(my_team)
        send_players = st.multiselect(
            "Players to shop",
            my_players,
            format_func=lambda x: label_map.get(x, x),
            key="finder_send_players",
        )
        send_picks = st.multiselect(
            "Picks to shop",
            build_pick_labels_for_team(my_team),
            key="finder_send_picks",
        )

        if st.button("Suggest what you could get back", key="btn_trade_away"):
            suggestions_all = []
            for ot in other_teams:
                offers = find_return_for_package(
                    my_team, ot, send_players, send_picks
                )
                for recv_players, recv_picks, total, pct in offers:
                    suggestions_all.append(
                        {
                            "From": my_team,
                            "To": ot,
                            "They Send players": recv_players,
                            "They Send picks": recv_picks,
                            "You Send players": send_players,
                            "You Send picks": send_picks,
                            "Approx value gap": pct,
                        }
                    )
            if not suggestions_all:
                st.info(
                    "No simple offers found within the tolerance window. "
                    "Try different assets or tweak the sliders."
                )
            else:
                suggestions_all.sort(key=lambda s: s["Approx value gap"])
                st.markdown("### Suggested ideas")
                for i, sug in enumerate(suggestions_all[:3], start=1):
                    st.markdown(f"**Idea #{i}** — {sug['From']} ↔ {sug['To']}")
                    st.write(
                        f"{sug['From']} sends: "
                        f"{', '.join(sug['You Send players']) or 'no players'}; "
                        f"{', '.join(sug['You Send picks']) or 'no picks'}"
                    )
                    st.write(
                        f"{sug['From']} receives: "
                        f"{', '.join(sug['They Send players']) or 'no players'}; "
                        f"{', '.join(sug['They Send picks']) or 'no picks'}"
                    )

                    expl = build_suggestion_explanation(
                        sug["You Send players"],
                        sug["You Send picks"],
                        sug["They Send players"],
                        sug["They Send picks"],
                        sug["Approx value gap"],
                    )
                    st.write(expl)
                    st.markdown("---")

# -------------------- Footer & release notes --------------------
st.markdown("---")
st.caption(
    "This tool uses FantasyPros Dynasty Superflex PPR rankings + PPR scoring curves, "
    "small adjustments for positional scarcity, age profiles, live Sleeper rosters, "
    "and a light touch of team-need awareness. It's meant to flag trades that are "
    "way off or broadly reasonable — not to replace your own judgement."
)

with st.expander("For more information, click here."):
    st.markdown(
        """
### How this calculator works

- **Rankings:** Uses the latest FantasyPros **Dynasty Superflex PPR** expert consensus.
- **Scoring curves:** Looks at historical PPR scoring by position to understand how fast production falls off (for example, WR1 vs WR20).
- **Age profiles:** Younger cores get a small bump; older players get a soft haircut, especially at RB.
- **Positional tiers:** Each player is tagged as **High-Tier**, **Starter**, **Flex**, or **Depth** based on overall FantasyPros rank, plus a separate **Young (≤26)** flag.
- **Positional scarcity:** Positions with fewer reliable options get a **small** premium, but not enough to make a depth TE more valuable than a truly premium WR or QB.
- **Team needs:** Roster needs are a *minor* factor — helpful for tiebreakers, not something that overrides rankings.
- **Picks:** Future picks are valued by:
  - Round (1st vs 2nd vs 3rd vs 4th),
  - How strong the original team looks (record + top players),
  - How far out the pick is (further years are slightly discounted),
  - A very small tweak for how many picks a team already has.
- **Monotonic sanity check:** If a player is ranked ahead of another overall, the lower-ranked player cannot have more model value — even across positions.

### Rough change log

- ✅ Switched from static CSVs to **live FantasyPros + Sleeper**.
- ✅ Added **future pick valuation** based on original team strength and year.
- ✅ Smoothed player values so top-end options pull away more from mid-tier choices.
- ✅ Added **trade categories** (Perfect Fit → Call the Commissioner) for quick gut checks.
- ✅ Built a **Trade Finder** with two modes:
  - “Acquire position/picks” to find ways to target specific spots or draft capital, now with filters for High-Tier / Starter / Flex / Depth / Young.
  - “Trade away” to see what you could reasonably get back for a package.
- ✅ Tweaked layouts for mobile, simplified the header, and combined context + details into one breakdown section.
"""
    )
