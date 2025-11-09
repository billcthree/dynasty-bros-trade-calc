import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
from itertools import combinations
import random
import requests

# -------------------- Page config --------------------
st.set_page_config(page_title="Dynasty Bros. Trade Calculator", layout="wide")

st.markdown("## Dynasty Bros. Trade Calculator")
st.caption(
    "Superflex PPR trade calculator using FantasyPros rankings and live Sleeper rosters."
)
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
        t
        for t in name.split()
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
        team_name = (
            meta.get("team_name")
            or u.get("display_name")
            or f"Team {i+1}"
        )
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
    records_df = (
        pd.DataFrame(records.values())
        if records
        else pd.DataFrame(columns=["Team", "Wins", "Losses", "Ties"])
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

    picks_by_team = {}          # current_owner -> [labels]
    pick_label_to_original = {} # label -> original team

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
        team_name = (
            meta.get("team_name")
            or u.get("display_name")
            or f"Team {i+1}"
        )
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
    - '#' : position rank (1 = top scorer at that position)
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
    Used only for small adjustments to player values and pick values.
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


@st.cache_data(show_spinner=False)
def load_player_ranks(uploaded_file):
    """
    Load FantasyPros-like player ranks.

    Expected columns: Player, Pos, Rank.
    """
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_csv("data/player_ranks.csv")

    cols = {c.lower(): c for c in df.columns}
    # normalize column names
    rename_map = {}
    if "player" in cols:
        rename_map[cols["player"]] = "Player"
    if "pos" in cols:
        rename_map[cols["pos"]] = "Pos"
    if "rank" in cols:
        rename_map[cols["rank"]] = "Rank"

    df = df.rename(columns=rename_map)
    df["Norm"] = df["Player"].apply(normalize_name)
    df["Rank"] = pd.to_numeric(df["Rank"], errors="coerce")
    df = df.dropna(subset=["Rank"])
    df = df.sort_values("Rank").reset_index(drop=True)
    return df


# ====================================================
# Sidebar: data source + modifiers
# ====================================================

st.sidebar.header("1) Data Source")

use_live = st.sidebar.checkbox(
    "Use live Sleeper + FantasyPros data",
    value=True,
    help=(
        "On: rosters/records from Sleeper API, rankings from data/player_ranks.csv "
        "(FantasyPros export)."
    ),
)

league_id = st.sidebar.text_input(
    "Sleeper League ID",
    value="1194681871141023744",
    help="Paste your Sleeper league ID (from league settings in the Sleeper app).",
)

st.sidebar.caption("You can still upload CSVs below to override or test things manually.")

up_players = st.sidebar.file_uploader("Player Ranks CSV (Player, Pos, Rank)", type=["csv"])
up_rosters = st.sidebar.file_uploader(
    "Rosters CSV (Team, Player, Pos optional)", type=["csv"]
)

# -------- Modifiers ----------
st.sidebar.header("2) Player value tuning (optional)")
st.sidebar.caption(
    "If you're not sure what to do here, you can safely leave the defaults."
)

ELITE_GAP = st.sidebar.slider(
    "How valuable is the #1 overall player?",
    800.0,
    2200.0,
    1600.0,
    50.0,
    help=(
        "Higher = bigger gap between elite players and everyone else. "
        "This also exaggerates the difference between players in the top ~30 and the rest."
    ),
)

RANK_IMPORTANCE = st.sidebar.slider(
    "How fast does value drop as players get lower in the rankings?",
    0.020,
    0.080,
    0.050,
    0.001,
    help=(
        "Higher = rank differences matter more (e.g., Rank 13 >> Rank 46 >> Rank 137). "
        "This especially stretches the top end (WR1 vs WR20 etc.)."
    ),
)

NEED_WEIGHT = st.sidebar.slider(
    "How much do roster needs matter?",
    0.0,
    0.6,
    0.20,
    0.05,
    help=(
        "Lower = mostly pure rankings. Higher = small boost when a trade fills a thin "
        "position or helps a team that is very old at a spot."
    ),
)

PACKAGE_PENALTY = st.sidebar.slider(
    "2-for-1 tax (multiple smaller players vs one stud)",
    0.0,
    1.0,
    0.65,
    0.05,
    help=(
        "Higher = quantity counts less vs quality. "
        "Prevents 3 mid players from 'beating' 1 superstar."
    ),
)

RISK_PREF = st.sidebar.slider(
    "Risk vs safety preference",
    0.0,
    1.0,
    0.5,
    0.05,
    help=(
        "0 = mildly prefers safer, older producers. 1 = mildly prefers youth/upside. "
        "This is a small nudge on top of the ranking-based value."
    ),
)

# ==========================================
# Load main datasets (ranks, rosters, logos)
# ==========================================

curves = load_ppr_curves()
age_table = load_age_table()

if use_live:
    try:
        rosters_live, records_live, picks_by_team_live, pick_label_to_original, future_years = (
            load_sleeper_league_v2(league_id)
        )
    except Exception as e:
        st.error(f"Error fetching Sleeper data: {e}")
        rosters_live = records_live = None
        picks_by_team_live = {}
        pick_label_to_original = {}
        future_years = []
else:
    rosters_live = records_live = None
    picks_by_team_live = {}
    pick_label_to_original = {}
    future_years = []

# Apply manual override rosters if uploaded
if up_rosters is not None:
    rosters = pd.read_csv(up_rosters)
    if "Pos" not in rosters.columns and "Position" in rosters.columns:
        rosters = rosters.rename(columns={"Position": "Pos"})
else:
    rosters = rosters_live

players_fp = load_player_ranks(up_players)

if rosters is None or rosters.empty:
    st.error(
        "No rosters available. Check Sleeper league ID, or upload a Rosters CSV with "
        "columns (Team, Player, Pos)."
    )
    st.stop()

# ---------------- Merge Sleeper rosters with FantasyPros ranks ----------------

rosters = rosters.copy()
rosters["Norm"] = rosters["Player"].apply(normalize_name)
players_fp = players_fp.copy()

merged = pd.merge(
    rosters,
    players_fp[["Norm", "Player", "Pos", "Rank"]],
    on="Norm",
    how="left",
    suffixes=("_Sleeper", "_FP"),
)

# fallbacks if FP pos is missing
merged["Pos_final"] = merged["Pos_FP"].fillna(merged["Pos_Sleeper"])
merged["Rank"] = pd.to_numeric(merged["Rank"], errors="coerce")

# If still NaN rank, push them to very low value but not zero
max_rank = merged["Rank"].max(skipna=True)
if pd.isna(max_rank):
    max_rank = 400
merged["Rank_filled"] = merged["Rank"].fillna(max_rank + 40)

# --------------------------------------
# Build position rank & value per player
# --------------------------------------


def build_pos_rank(df: pd.DataFrame) -> pd.DataFrame:
    # rank within each position based on overall Rank_filled
    df = df.copy()
    df["PosRank"] = (
        df.sort_values("Rank_filled")
        .groupby("Pos_final")["Rank_filled"]
        .rank(method="first")
    )
    return df


merged = build_pos_rank(merged)


def get_ppr_points_for_pos(pos: str, pos_rank: float) -> float:
    """Approximate PPR points at a given position rank using curves (if available)."""
    if curves is None or pos not in curves:
        # simple fall-back: exponential decay for positions
        # slightly steeper for QB & TE due to scarcity
        base = {"QB": 380, "RB": 330, "WR": 320, "TE": 220}.get(pos, 250)
        decay = {"QB": 0.055, "RB": 0.045, "WR": 0.040, "TE": 0.050}.get(pos, 0.045)
        return base * np.exp(-decay * (pos_rank - 1))

    curve = curves[pos]
    # If we have exact rank
    if pos_rank <= curve["PosRank"].max():
        # simple interpolation
        idx = np.searchsorted(curve["PosRank"].values, pos_rank, side="right") - 1
        idx = max(0, min(idx, len(curve) - 1))
        return float(curve.iloc[idx]["Points"])

    # beyond known curve: tail off
    last_points = float(curve["Points"].iloc[-1])
    extra = pos_rank - curve["PosRank"].iloc[-1]
    return last_points * (0.96 ** extra)


def pos_weight(pos: str) -> float:
    """
    Slight positional weighting so TE doesn't randomly outrank elite WR/RB too often.
    """
    return {
        "QB": 1.05,  # very important in superflex
        "RB": 1.00,
        "WR": 1.00,
        "TE": 0.88,  # nerf TE a bit overall
    }.get(pos, 1.0)


def age_adjustment(player_name: str, base_val: float, pos: str) -> float:
    """
    Small adjustment based on age.
    RB/WR: age cliff a bit earlier
    QB: age matters less
    TE: medium
    """
    if age_table is None or base_val <= 0:
        return base_val

    norm = normalize_name(player_name)
    row = age_table.loc[age_table["Norm"] == norm]
    if row.empty:
        return base_val

    age = float(row["Age"].iloc[0])

    # target prime windows
    if pos == "QB":
        # long shelf life, mild penalty late
        if age <= 26:
            mult = 1.05
        elif age <= 30:
            mult = 1.02
        elif age <= 34:
            mult = 0.98
        else:
            mult = 0.92
    elif pos in ["RB"]:
        if age <= 23:
            mult = 1.06
        elif age <= 26:
            mult = 1.02
        elif age <= 28:
            mult = 0.96
        else:
            mult = 0.90
    elif pos == "WR":
        if age <= 24:
            mult = 1.05
        elif age <= 28:
            mult = 1.02
        elif age <= 30:
            mult = 0.97
        else:
            mult = 0.92
    else:  # TE
        if age <= 25:
            mult = 1.04
        elif age <= 29:
            mult = 1.01
        elif age <= 32:
            mult = 0.97
        else:
            mult = 0.92

    # tilt slightly based on RISK_PREF
    # 0 = prefers vets (less penalty), 1 = prefers youth (more penalty on old)
    if age >= 28:
        mult *= (0.99 + 0.05 * (1.0 - RISK_PREF))  # older gets a bit more forgiveness if risk_pref is low
    elif age <= 24:
        mult *= (1.0 + 0.05 * RISK_PREF)

    return base_val * mult


def build_player_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Combine:
      - Overall FantasyPros Rank (Rank_filled)
      - Positional rank -> PPR points curve
      - Positional weighting
      - Age adjustments
      - Stronger separation for elite players (using RANK_IMPORTANCE + ELITE_GAP)
    """
    df = df.copy()
    max_rank_local = df["Rank_filled"].max()

    values = []
    for _, row in df.iterrows():
        player = row["Player_FP"] if "Player_FP" in row and isinstance(row["Player_FP"], str) else row["Player"]
        pos = row["Pos_final"]
        overall_rank = float(row["Rank_filled"])
        pos_rank = float(row["PosRank"])

        # PPR curve
        ppr_points = get_ppr_points_for_pos(pos, pos_rank)

        # Normalize PPR per position
        if curves is not None and pos in curves:
            top_points = float(curves[pos]["Points"].iloc[0])
        else:
            top_points = get_ppr_points_for_pos(pos, 1.0)
        norm_ppr = ppr_points / top_points if top_points > 0 else 1.0

        # Rank-based exponent curve:
        # Big separation at the very top, then it flattens
        # rank_factor roughly between ~1.0 (rank 1) and small numbers at 200+
        rank_factor = np.exp(-RANK_IMPORTANCE * (overall_rank - 1.0))

        # emphasize top 20-30 extra by bending the curve
        elite_boost = 1.0
        if overall_rank <= 5:
            elite_boost = 1.30
        elif overall_rank <= 12:
            elite_boost = 1.18
        elif overall_rank <= 24:
            elite_boost = 1.10
        elif overall_rank <= 40:
            elite_boost = 1.04

        # Combine
        base_val = ELITE_GAP * rank_factor * (0.35 + 0.65 * norm_ppr) * pos_weight(pos) * elite_boost

        # Age tweak
        base_val = age_adjustment(player, base_val, pos)

        values.append(base_val)

    df["BaseValue"] = values
    return df


merged = build_player_values(merged)

# Make a simple player-value table for quick lookups
player_values = merged.groupby(["Player", "Pos_final"], as_index=False).agg(
    {"BaseValue": "mean", "Rank_filled": "mean"}
)
player_values = player_values.rename(columns={"Pos_final": "Pos", "Rank_filled": "Rank"})


def get_player_value(name: str) -> float:
    row = player_values.loc[player_values["Player"] == name]
    if row.empty:
        return 0.0
    return float(row["BaseValue"].iloc[0])


def get_player_rank_pos(name: str):
    row = player_values.loc[player_values["Player"] == name]
    if row.empty:
        return None, None, None
    return float(row["Rank"].iloc[0]), row["Pos"].iloc[0], float(row["BaseValue"].iloc[0])


# ============================================
# Roster context & team need multipliers
# ============================================

def team_pos_counts(team_name: str):
    sub = merged.loc[merged["Team"] == team_name]
    counts = {p: int((sub["Pos_final"] == p).sum()) for p in ["QB", "RB", "WR", "TE"]}
    return counts


def team_age_profile(team_name: str):
    if age_table is None:
        return {}
    sub = merged.loc[merged["Team"] == team_name]
    if sub.empty:
        return {}
    res = {}
    for pos in ["QB", "RB", "WR", "TE"]:
        pos_players = sub.loc[sub["Pos_final"] == pos]
        if pos_players.empty:
            continue
        ages = []
        for p in pos_players["Player_FP"].fillna(pos_players["Player"]):
            norm = normalize_name(p)
            arow = age_table.loc[age_table["Norm"] == norm]
            if not arow.empty:
                ages.append(float(arow["Age"].iloc[0]))
        if ages:
            res[pos] = np.mean(ages)
    return res


DEFAULT_TARGETS = {"QB": 3, "RB": 5, "WR": 6, "TE": 2}


def need_score_for_pos(count: int, target: int) -> float:
    """
    Return a small need score in [-1, +1] based on how far from target a team is.
    """
    if target <= 0:
        return 0.0
    diff = count - target
    if diff <= -2:
        return 1.0
    if diff == -1:
        return 0.5
    if diff == 0:
        return 0.0
    if diff == 1:
        return -0.4
    return -0.7


def team_need_multiplier(team_name: str, incoming_players: list[str]) -> tuple[float, list]:
    """
    Compute a combined team-need multiplier for all incoming players.

    Returns:
      total_mult (applied multiplicatively to combined value),
      details: list of (Player, Pos, per-player-mult, short_reason)
    """
    counts_before = team_pos_counts(team_name)
    details = []

    # We'll treat each incoming player separately, and then combine multiplicatively.
    total_mult = 1.0

    for p in incoming_players:
        rank, pos, base_val = get_player_rank_pos(p)
        if pos is None:
            continue

        cnt = counts_before.get(pos, 0)
        target = DEFAULT_TARGETS.get(pos, 0)
        ns = need_score_for_pos(cnt, target)

        # Use NEED_WEIGHT as the maximum additional bump
        per_mult = 1.0 + NEED_WEIGHT * ns

        # slight tweak for age windows: rebuilding teams may prefer youth, contenders vets
        # we infer a bit from team record
        if records_live is not None and not records_live.empty:
            rec_row = records_live.loc[records_live["Team"] == team_name]
            if not rec_row.empty:
                wins = float(rec_row["Wins"].iloc[0])
                losses = float(rec_row["Losses"].iloc[0])
                total = wins + losses
                win_pct = wins / total if total > 0 else 0.5
                # contending team -> slightly more weight on current production
                if win_pct >= 0.65:
                    per_mult *= 1.02
                elif win_pct <= 0.35:
                    per_mult *= 1.02

        total_mult *= per_mult

        reason = ""
        if ns > 0.4:
            reason = f"{team_name} is light at {pos}, so acquiring {p} is slightly more valuable to them."
        elif ns < -0.4:
            reason = f"{team_name} is already deep at {pos}, so {p} is a bit less of a priority."
        details.append((p, pos, round(per_mult, 3), reason))

    return total_mult, details


# =======================
# Pick valuation
# =======================

def team_power_score(team_name: str) -> float:
    """
    Estimate how strong a team is (for pick value).

    Uses:
      - Wins/Losses
      - Sum of top N player BaseValue
    """
    rec_score = 0.0
    if records_live is not None and not records_live.empty:
        row = records_live.loc[records_live["Team"] == team_name]
        if not row.empty:
            wins = float(row["Wins"].iloc[0])
            losses = float(row["Losses"].iloc[0])
            total = wins + losses
            win_pct = wins / total if total > 0 else 0.5
            rec_score = (win_pct - 0.5) * 4.0  # ~[-2, +2]

    sub = merged.loc[merged["Team"] == team_name]
    top_vals = sorted(sub["BaseValue"].tolist(), reverse=True)[:12]
    if top_vals:
        roster_score = (np.mean(top_vals) / (ELITE_GAP or 1.0)) * 4.0
    else:
        roster_score = 0.0

    return rec_score * 0.6 + roster_score * 0.4


def estimate_pick_slot(original_team: str, total_teams: int = 12) -> float:
    """
    Estimate likely draft slot (1 = 1.01, 12 = 1.12).
    Lower power_score => earlier pick.
    """
    if not original_team:
        return total_teams / 2

    all_teams = sorted(rosters["Team"].unique().tolist())
    scores = {t: team_power_score(t) for t in all_teams}
    sorted_t = sorted(all_teams, key=lambda t: scores[t])

    # worst score first => earliest pick
    try:
        idx = sorted_t.index(original_team)
    except ValueError:
        return total_teams / 2

    return float(idx + 1)


def pick_base_curve(round_num: int) -> float:
    """
    Base value for picks before slot adjustments.
    Toned down compared to studs.
    """
    if round_num == 1:
        return 650.0
    if round_num == 2:
        return 330.0
    if round_num == 3:
        return 140.0
    if round_num == 4:
        return 60.0
    return 20.0


def pick_value(label: str) -> float:
    """
    Given pick label like '2026 R1 (Wilfork Your Mom)', return numeric value.
    """
    try:
        parts = label.split()
        year = int(parts[0])
        rnd = int(parts[1].replace("R", "").replace("(", "").replace(")", ""))
    except Exception:
        return 0.0

    # parse original team name inside parentheses
    start = label.find("(")
    end = label.rfind(")")
    original_team = ""
    if start != -1 and end != -1 and end > start:
        original_team = label[start + 1 : end]

    base = pick_base_curve(rnd)

    # adjust by expected slot: early picks get a decent bump
    slot = estimate_pick_slot(original_team)
    # slot from 1 (best) to ~12 (worst)
    slot_mult = 1.25 - 0.5 * ((slot - 1) / max(slot - 1, 11))
    slot_mult = max(0.7, min(1.3, slot_mult))

    # very small fade for future years
    current_year = datetime.now().year
    year_diff = max(0, year - current_year)
    time_mult = (0.95) ** year_diff

    return base * slot_mult * time_mult


# ==============================================
# Utility: package penalty for many-for-one
# ==============================================

def apply_package_penalty(total_value: float, num_assets: int) -> float:
    if num_assets <= 1:
        return total_value
    # each extra asset gets taxed a bit
    tax = PACKAGE_PENALTY * (num_assets - 1) * 0.15
    mult = max(0.60, 1.0 - tax)
    return total_value * mult


# ==========================================
# UI: logos
# ==========================================

league_logo_url, team_logo_urls = (None, {})
if use_live:
    try:
        league_logo_url, team_logo_urls = load_sleeper_logos(league_id)
    except Exception:
        league_logo_url, team_logo_urls = (None, {})

if league_logo_url:
    st.image(league_logo_url, width=60)


# ==========================================
# Tabs: Trade Calculator vs Trade Finder
# ==========================================

tab_calc, tab_finder = st.tabs(["Trade Calculator", "Trade Finder"])

teams = sorted(rosters["Team"].unique().tolist())


# ====================================================
# TRADE CALCULATOR TAB
# ====================================================

with tab_calc:
    st.markdown("### Build a Trade")

    colA, colB = st.columns(2)
    with colA:
        teamA = st.selectbox("Team A", teams, index=0)
        if teamA in team_logo_urls:
            st.image(team_logo_urls[teamA], width=60)
    with colB:
        teamB = st.selectbox("Team B", teams, index=min(1, len(teams) - 1))
        if teamB in team_logo_urls:
            st.image(team_logo_urls[teamB], width=60)

    st.markdown("#### Assets being moved (you can select multiple players and picks per side)")

    # Player lists restricted to each roster
    players_A = sorted(rosters.loc[rosters["Team"] == teamA, "Player"].unique().tolist())
    players_B = sorted(rosters.loc[rosters["Team"] == teamB, "Player"].unique().tolist())

    picksA = sorted(picks_by_team_live.get(teamA, []))
    picksB = sorted(picks_by_team_live.get(teamB, []))

    c1, c2 = st.columns(2)
    with c1:
        st.markdown(f"**{teamA} receives from {teamB}**")
        A_get_players = st.multiselect(
            "Players to acquire",
            options=players_B,
            key="A_get_players",
        )
        A_get_picks = st.multiselect(
            "Draft picks to acquire",
            options=picksB,
            key="A_get_picks",
        )

    with c2:
        st.markdown(f"**{teamB} receives from {teamA}**")
        B_get_players = st.multiselect(
            "Players to acquire",
            options=players_A,
            key="B_get_players",
        )
        B_get_picks = st.multiselect(
            "Draft picks to acquire",
            options=picksA,
            key="B_get_picks",
        )

    # ----------- Compute values -----------
    def side_value(team_name: str, players_in: list[str], picks_in: list[str]):
        # base sum
        player_vals = []
        for p in players_in:
            v = get_player_value(p)
            player_vals.append((p, v))
        total_players = sum(v for _, v in player_vals)

        total_picks = sum(pick_value(lbl) for lbl in picks_in)

        base_total = total_players + total_picks

        # team-need multiplier
        need_mult, need_details = team_need_multiplier(team_name, players_in)
        total_after_need = base_total * need_mult

        # package tax (players only)
        total_after_package = apply_package_penalty(
            total_after_need,
            num_assets=len(players_in),
        )

        return {
            "base_total": base_total,
            "total_players": total_players,
            "total_picks": total_picks,
            "after_need": total_after_need,
            "after_package": total_after_package,
            "player_vals": player_vals,
            "need_details": need_details,
        }

    resultA = side_value(teamA, A_get_players, A_get_picks)
    resultB = side_value(teamB, B_get_players, B_get_picks)

    totalA = resultA["after_package"]
    totalB = resultB["after_package"]

    # Small guard to avoid division by zero
    bigger = max(totalA, totalB, 1.0)
    diff = totalA - totalB
    pct = diff / bigger

    st.markdown("---")
    st.subheader("Trade Value Summary")

    m1, m2, m3 = st.columns(3)
    with m1:
        st.metric(
            f"{teamA} total value received",
            f"{totalA:,.0f}",
            help="Higher number = more dynasty value coming in."
        )
    with m2:
        st.metric(
            f"{teamB} total value received",
            f"{totalB:,.0f}",
            help="Higher number = more dynasty value coming in."
        )
    with m3:
        st.metric(
            "Value gap",
            f"{abs(diff):,.0f}",
            help="Difference between what each side receives (after roster context and 2-for-1 tax)."
        )

    st.markdown("#### Fairness verdict")

    if not A_get_players and not B_get_players and not A_get_picks and not B_get_picks:
        st.info("Select some players or picks on each side to evaluate a trade.")
    else:
        # Interpret fairness more gently
        abs_pct = abs(pct)

        if abs_pct < 0.08:
            verdict = "This looks **very close** in value."
        elif abs_pct < 0.18:
            verdict = "This trade **slightly favors** one side."
        elif abs_pct < 0.35:
            verdict = "This trade **meaningfully leans** to one side."
        else:
            verdict = "This trade **strongly leans** to one side."

        if diff > 0:
            favored = teamA
            other = teamB
        elif diff < 0:
            favored = teamB
            other = teamA
        else:
            favored = None
            other = None

        if favored is None:
            st.success(
                f"{verdict} The values are almost identical once we factor in rankings, picks, "
                "roster context, and 2-for-1 tax."
            )
        else:
            st.warning(
                f"{verdict} It **likely favors {favored}** by about {abs(diff):,.0f} points "
                f"(~{abs_pct:,.1%} of the side getting more value).\n\n"
                "Remember: this is just a model. League-mates might value youth, risk, or "
                "positional scarcity differently."
            )

        # Explain what "points" mean
        st.caption(
            "_These 'points' are a synthetic dynasty value scale built from FantasyPros "
            "rankings + historical scoring curves. The percentage tells you roughly how "
            "far apart the two sides are relative to the bigger side._"
        )

    # ----------------- Detailed breakdown -----------------
    with st.expander("Player-by-player breakdown (FantasyPros rank & value)"):
        rows = []

        def add_side(team_name, players, picks, label):
            for p in players:
                r, pos, v = get_player_rank_pos(p)
                rows.append({
                    "Side": label,
                    "Team": team_name,
                    "Asset": p,
                    "Type": "Player",
                    "Pos": pos,
                    "FP Rank": None if r is None else int(round(r)),
                    "Value": v,
                })
            for pk in picks:
                rows.append({
                    "Side": label,
                    "Team": team_name,
                    "Asset": pk,
                    "Type": "Pick",
                    "Pos": "",
                    "FP Rank": None,
                    "Value": pick_value(pk),
                })

        add_side(teamA, A_get_players, A_get_picks, f"{teamA} receives")
        add_side(teamB, B_get_players, B_get_picks, f"{teamB} receives")

        if rows:
            df_detail = pd.DataFrame(rows)
            st.dataframe(
                df_detail.sort_values(["Side", "Type", "Value"], ascending=[True, True, False]),
                use_container_width=True,
            )
        else:
            st.write("No assets selected yet.")

    with st.expander("Roster context (why someone might like or dislike this trade)"):
        def roster_context(team_name, incoming_players, incoming_picks, result):
            st.markdown(f"**{team_name}**")

            counts = team_pos_counts(team_name)
            age_prof = team_age_profile(team_name)

            bullets = []

            for pos in ["QB", "RB", "WR", "TE"]:
                cnt = counts.get(pos, 0)
                tgt = DEFAULT_TARGETS.get(pos, 0)
                if cnt < tgt - 1:
                    bullets.append(f"- Light at **{pos}** ({cnt} on roster, target {tgt}).")
                elif cnt > tgt + 1:
                    bullets.append(f"- Already pretty deep at **{pos}** ({cnt} on roster).")

            if age_prof:
                for pos, avg_age in age_prof.items():
                    if avg_age >= 28:
                        bullets.append(f"- {pos} room is **veteran-heavy** (avg age ~{avg_age:.1f}).")
                    elif avg_age <= 24:
                        bullets.append(f"- {pos} room is **very young** (avg age ~{avg_age:.1f}).")

            if records_live is not None and not records_live.empty:
                rec_row = records_live.loc[records_live["Team"] == team_name]
                if not rec_row.empty:
                    w = int(rec_row["Wins"].iloc[0])
                    l = int(rec_row["Losses"].iloc[0])
                    bullets.append(f"- Current record: **{w}-{l}**.")

            if incoming_players:
                bullets.append(
                    f"- Incoming players: {', '.join(incoming_players)}."
                )

            need_details = [d for d in result["need_details"] if d[3]]
            for p, pos, mult, reason in need_details:
                bullets.append(f"- {reason} (multiplier ~{mult:.2f}).")

            if incoming_picks:
                bullets.append(
                    "- Incoming picks add some flexibility (future trades, rookie swings). "
                    "Earlier picks for weaker/or rebuilding teams will naturally be valued a bit more."
                )

            if bullets:
                for b in bullets:
                    st.write(b)
            else:
                st.write(
                    "Neither team has obvious positional red flags in this deal. At that point, "
                    "it often comes down to personal preference, age windows, and risk tolerance."
                )

        roster_context(teamA, A_get_players, A_get_picks, resultA)
        st.markdown("---")
        roster_context(teamB, B_get_players, B_get_picks, resultB)

    # --------------- Suggest tweaks section ---------------
    with st.expander("Suggestions to help balance the trade"):
        if not A_get_players and not B_get_players and not A_get_picks and not B_get_picks:
            st.write("Once you build a trade, this section will suggest simple tweaks (like adding a pick).")
        else:
            if abs(diff) < 0.05 * bigger:
                st.write("This is already quite close. Small tweaks (like swapping a minor depth piece) may be enough if someone is uneasy.")
            else:
                if diff > 0:
                    winning_team = teamA
                    losing_team = teamB
                    losing_picks = picksB
                    losing_players = players_B
                    target_gap = diff
                else:
                    winning_team = teamB
                    losing_team = teamA
                    losing_picks = picksA
                    losing_players = players_A
                    target_gap = -diff

                st.markdown(
                    f"- Right now, **{winning_team}** is getting the better side.\n"
                    f"- To make things feel more even, **{losing_team}** could consider adding something small."
                )

                # simple suggestion: add the smallest reasonable pick or depth player
                candidates = []

                for pk in losing_picks:
                    v = pick_value(pk)
                    if v > 0:
                        candidates.append(("Pick", pk, v))

                for p in losing_players:
                    v = get_player_value(p)
                    if 0 < v < target_gap * 1.5:
                        candidates.append(("Player", p, v))

                candidates = sorted(candidates, key=lambda x: x[2])

                if not candidates:
                    st.write(
                        "It's hard to find an obvious small add from the losing side. "
                        "You may need to re-shape the core pieces of the deal instead."
                    )
                else:
                    best = candidates[0]
                    kind, asset, v = best
                    st.write(
                        f"- A simple option: **{losing_team}** adds {asset} "
                        f"(rough value ~{v:,.0f}). That would pull the values much closer."
                    )


# ====================================================
# TRADE FINDER TAB
# ====================================================

with tab_finder:
    st.markdown("### Trade Finder")

    st.write(
        "Pick a player you want to acquire, and we'll search your roster + your picks "
        "for packages that land roughly in the right value range."
    )

    col1, col2 = st.columns(2)
    with col1:
        your_team = st.selectbox("Your team", teams, key="finder_your_team")
    with col2:
        other_team = st.selectbox(
            "Team you're trading with",
            [t for t in teams if t != your_team],
            key="finder_other_team",
        )

    other_players = sorted(
        rosters.loc[rosters["Team"] == other_team, "Player"].unique().tolist()
    )

    target_player = st.selectbox(
        f"Player on {other_team} you want",
        other_players,
        key="finder_target_player",
    )

    your_players = sorted(
        rosters.loc[rosters["Team"] == your_team, "Player"].unique().tolist()
    )
    your_picks = sorted(picks_by_team_live.get(your_team, []))

    max_assets = st.slider(
        "Max number of assets you want to send (players + picks)",
        1,
        4,
        3,
    )

    if st.button("Suggest a trade package"):
        # value of target
        target_val = get_player_value(target_player)

        if target_val <= 0:
            st.warning(
                "Couldn't find a solid value for that player (maybe missing from rankings)."
            )
        else:
            st.write(
                f"Estimated value for **{target_player}**: ~{target_val:,.0f} points."
            )

            # build list of your tradable assets
            assets = []
            for p in your_players:
                v = get_player_value(p)
                if v > 0:
                    assets.append(("Player", p, v))
            for pk in your_picks:
                v = pick_value(pk)
                if v > 0:
                    assets.append(("Pick", pk, v))

            # only consider top N assets to keep search manageable
            assets = sorted(assets, key=lambda x: x[2], reverse=True)[:18]

            best_candidates = []

            for r in range(1, max_assets + 1):
                for combo in combinations(assets, r):
                    total = sum(x[2] for x in combo)
                    # apply package penalty as if they all go in one side
                    total_after = apply_package_penalty(total, len(combo))

                    # we want something in ~90%–120% of target value
                    if 0.90 * target_val <= total_after <= 1.20 * target_val:
                        best_candidates.append((combo, total_after))

            if not best_candidates:
                st.info(
                    "I couldn't find a simple package in that range. "
                    "You may need to involve a core piece or accept a bigger overpay/underpay."
                )
            else:
                # shuffle for variety, then show a few
                random.shuffle(best_candidates)
                best_candidates = sorted(best_candidates, key=lambda x: abs(x[1] - target_val))

                st.markdown("#### Suggested packages (you send this to get your target)")

                for i, (combo, total_after) in enumerate(best_candidates[:3], start=1):
                    player_list = [a[1] for a in combo if a[0] == "Player"]
                    pick_list = [a[1] for a in combo if a[0] == "Pick"]
                    st.markdown(f"**Option {i}** — approx value ~{total_after:,.0f}")

                    if player_list:
                        st.write("- Players: " + ", ".join(player_list))
                    if pick_list:
                        st.write("- Picks: " + ", ".join(pick_list))

                    gap = total_after - target_val
                    pct_gap = gap / max(total_after, target_val)
                    if abs(pct_gap) < 0.08:
                        st.caption("Pretty close in value; a reasonable starting point.")
                    elif pct_gap > 0:
                        st.caption(
                            "Slight overpay on your side — you might ask for a small add-back "
                            "from the other manager."
                        )
                    else:
                        st.caption(
                            "Slight underpay — depending on how much they like your pieces, "
                            "they might still consider it."
                        )

# ============================
# Footer: how this is built
# ============================

st.markdown("---")
st.markdown(
    "_This tool blends FantasyPros Dynasty Superflex PPR rankings with historical PPR "
    "scoring curves and live Sleeper league data (rosters, records, and traded picks). "
    "Values are **guides**, not absolute truths — your league-mates may weigh age, risk, "
    "and positional scarcity differently._"
)
