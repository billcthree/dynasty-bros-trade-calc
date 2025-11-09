import math
import difflib
from functools import lru_cache

import pandas as pd
import numpy as np
import requests
import streamlit as st


LEAGUE_ID = "1194681871141023744"
CURRENT_SEASON = 2025  # used for pick time discount; adjust if needed

st.set_page_config(
    page_title="Dynasty Bros. Trade Calculator",
    layout="wide",
)

# -------------------------
# Helpers
# -------------------------


def normalize_name(name: str) -> str:
    """Lowercase, strip punctuation, drop Jr/Sr/II/III, etc., to help match names."""
    if not isinstance(name, str):
        return ""
    s = name.lower().strip()
    # Remove punctuation-ish
    for ch in [".", "'", ","]:
        s = s.replace(ch, "")
    # Collapse spaces
    s = " ".join(s.split())
    # Drop common suffixes at end
    suffixes = [" jr", " sr", " ii", " iii", " iv", " v"]
    for suf in suffixes:
        if s.endswith(suf):
            s = s[: -len(suf)]
            s = s.strip()
    return s


@st.cache_data(show_spinner=False)
def load_fp_ranks() -> pd.DataFrame:
    """Load FantasyPros Dynasty Superflex ranks from player_ranks.csv."""
    df = pd.read_csv("player_ranks.csv")
    df["Player"] = df["Player"].astype(str).str.strip()
    df["Pos"] = df["Pos"].astype(str).str.upper().str.strip()
    df["Rank"] = pd.to_numeric(df["Rank"], errors="coerce")
    df = df.dropna(subset=["Rank"]).copy()
    df["Rank"] = df["Rank"].astype(int)
    df["name_key"] = df["Player"].map(normalize_name)
    return df


@st.cache_data(show_spinner=False)
def load_age() -> pd.DataFrame | None:
    """Load fantasyage.csv if present, keyed by normalized player name."""
    try:
        df = pd.read_csv("fantasyage.csv")
    except Exception:
        return None
    # Try to find the player + age columns
    cols_lower = {c.lower(): c for c in df.columns}
    player_col = cols_lower.get("player") or cols_lower.get("player name") or cols_lower.get("player_name")
    age_col = cols_lower.get("age") or cols_lower.get("yrs") or cols_lower.get("years")
    if not player_col or not age_col:
        return None
    df[player_col] = df[player_col].astype(str).str.strip()
    df["name_key"] = df[player_col].map(normalize_name)
    df["AGE"] = pd.to_numeric(df[age_col], errors="coerce")
    return df[["name_key", "AGE"]]


@st.cache_data(show_spinner=False)
def load_ppr_curves():
    """
    Load positional scoring curves from PPR .xlsx.

    We use the MDN (median) column vs scoring rank (#) to approximate how quickly
    production falls off at each position.
    """
    try:
        xls = pd.ExcelFile("PPR .xlsx")
    except Exception:
        return {}

    pos_sheet = {"QB": "QB24", "RB": "RB24", "WR": "WR24", "TE": "TE24"}
    curves = {}

    for pos, sheet in pos_sheet.items():
        if sheet not in xls.sheet_names:
            continue
        df = pd.read_excel(xls, sheet_name=sheet)
        if "#" not in df.columns:
            continue
        # try to locate a PPR column
        score_col = None
        for c in df.columns:
            if "mdn" in str(c).lower() or "ppr" in str(c).lower() or "ttl" in str(c).lower():
                score_col = c
                break
        if score_col is None:
            continue

        df = df.rename(columns={"#": "ScoreRank"})[["ScoreRank", score_col]]
        df = df.dropna(subset=["ScoreRank", score_col]).copy()
        df["ScoreRank"] = pd.to_numeric(df["ScoreRank"], errors="coerce")
        df[score_col] = pd.to_numeric(df[score_col], errors="coerce")
        df = df.dropna(subset=["ScoreRank", score_col]).sort_values("ScoreRank")
        if df.empty:
            continue

        # choose replacement-level rank per position
        repl_rank = {"QB": 24, "RB": 30, "WR": 40, "TE": 18}.get(pos, 30)
        max_rank = int(df["ScoreRank"].max())
        rr = min(repl_rank, max_rank)
        repl_mdn = float(df.loc[df["ScoreRank"] == rr, score_col].iloc[0])

        ranks = df["ScoreRank"].to_numpy()
        mdn = df[score_col].to_numpy()

        def vorp_for_pos_rank(pos_rank: int) -> float:
            """Approximate VORP at this positional rank using the PPR curve."""
            if pos_rank <= ranks.min():
                m = mdn[0]
            elif pos_rank >= ranks.max():
                m = mdn[-1]
            else:
                hi_idx = np.searchsorted(ranks, pos_rank, side="right")
                lo_idx = hi_idx - 1
                r_lo, r_hi = ranks[lo_idx], ranks[hi_idx]
                m_lo, m_hi = mdn[lo_idx], mdn[hi_idx]
                if r_hi == r_lo:
                    m = m_lo
                else:
                    frac = (pos_rank - r_lo) / (r_hi - r_lo)
                    m = m_lo + frac * (m_hi - m_lo)
            vorp = max(0.0, m - repl_mdn)
            return float(vorp)

        curves[pos] = vorp_for_pos_rank

    return curves


def apply_age_multiplier(base: float, age: float | None, pos: str) -> float:
    """Apply a small age bump / haircut (a bit stronger for RB, mild for others)."""
    if age is None or (isinstance(age, float) and (math.isnan(age))):
        return base

    if age <= 23:
        mult = 1.10
    elif age <= 26:
        mult = 1.05
    elif age <= 29:
        mult = 1.00
    elif age <= 31:
        mult = 0.94
    else:
        mult = 0.88

    # Age matters more for RBs, a bit less for TEs
    if pos == "RB":
        mult = 1 + (mult - 1) * 1.4
    elif pos == "TE":
        mult = 1 + (mult - 1) * 0.8

    return base * mult


def build_player_values() -> pd.DataFrame:
    """
    Build the core player table with a ValueIndex column.

    ValueIndex is based on:
      - FantasyPros Dynasty Superflex rank per position
      - PPR median scoring drop-off curves by position
      - Light age adjustments
      - Position weight (QB > RB > WR >> TE)
      - Final monotonic smoothing so a lower-ranked player never grades higher than a higher-ranked one.
    """
    fp = load_fp_ranks()
    age = load_age()
    curves = load_ppr_curves()

    # merge age by normalized name
    if age is not None and not age.empty:
        fp = fp.merge(
            age,
            on="name_key",
            how="left",
        )
    else:
        fp["AGE"] = np.nan

    # positional rank (1st WR, 2nd WR, etc.) based on FantasyPros overall rank
    fp = fp.sort_values("Rank").copy()
    fp["PosRank"] = fp.groupby("Pos").cumcount() + 1

    # TE heavily discounted, QB rewarded, RB slightly rewarded
    pos_weights = {"QB": 1.20, "RB": 1.10, "WR": 1.00, "TE": 0.55}
    base_values = []

    for _, row in fp.iterrows():
        pos = row["Pos"]
        pos_rank = int(row["PosRank"])

        # Core VORP from PPR curves – makes the top few guys pop
        if pos in curves:
            base_vorp = curves[pos](pos_rank)
        else:
            # Fallback: steeper curve at the very top
            base_vorp = max(0.0, 220.0 / math.sqrt(pos_rank))

        pos_mult = pos_weights.get(pos, 1.0)
        score = base_vorp * pos_mult * 10.0  # scaled

        score = apply_age_multiplier(score, row.get("AGE", np.nan), pos)
        base_values.append(score)

    fp["BaseValue"] = base_values

    # -------- Monotonic smoothing by overall rank ----------
    # Ensure that as overall Rank gets worse (higher number), value never goes UP.
    fp = fp.sort_values("Rank").reset_index(drop=True)
    values = fp["BaseValue"].to_numpy()
    for i in range(1, len(values)):
        if values[i] > values[i - 1]:
            values[i] = values[i - 1] * 0.997  # small downward step
    fp["BaseValueAdj"] = values

    max_val = fp["BaseValueAdj"].max()
    if max_val and max_val > 0:
        fp["ValueIndex"] = (fp["BaseValueAdj"] / max_val) * 1000.0
    else:
        fp["ValueIndex"] = 0.0

    fp = fp.rename(columns={"Player": "PlayerName"})
    fp["Player"] = fp["PlayerName"]
    return fp


# ---------- Sleeper integration (with picks + logo) ----------


@st.cache_data(show_spinner=False)
def fetch_sleeper_data(league_id: str):
    base = "https://api.sleeper.app/v1"
    league = requests.get(f"{base}/league/{league_id}", timeout=10).json()
    users = requests.get(f"{base}/league/{league_id}/users", timeout=10).json()
    rosters = requests.get(f"{base}/league/{league_id}/rosters", timeout=10).json()
    traded_picks = requests.get(f"{base}/league/{league_id}/traded_picks", timeout=10).json()

    # basic team info
    user_by_id = {}
    for u in users:
        uid = u.get("user_id")
        meta = u.get("metadata") or {}
        display = (
            meta.get("team_name")
            or meta.get("nickname")
            or u.get("display_name")
            or u.get("username")
            or f"Team {uid}"
        )
        avatar = meta.get("team_avatar") or u.get("avatar")
        user_by_id[uid] = {
            "team_name": display,
            "avatar": avatar,
        }

    teams = []
    all_player_ids = set()
    for r in rosters:
        roster_id = r.get("roster_id")
        owner_id = r.get("owner_id")
        players = r.get("players") or []
        for pid in players:
            all_player_ids.add(pid)

        settings = r.get("settings") or {}
        wins = settings.get("wins", 0) or 0
        losses = settings.get("losses", 0) or 0
        ties = settings.get("ties", 0) or 0
        fpts = settings.get("fpts", 0) or 0
        fpts_dec = settings.get("fpts_decimal", 0) or 0
        points_for = float(fpts) + float(fpts_dec) / 100.0

        meta = user_by_id.get(owner_id, {})
        teams.append(
            {
                "team_id": roster_id,
                "owner_id": owner_id,
                "team_name": meta.get("team_name", f"Team {roster_id}"),
                "avatar": meta.get("avatar"),
                "wins": wins,
                "losses": losses,
                "ties": ties,
                "points_for": points_for,
                "player_ids": players,
            }
        )

    # player metadata
    try:
        players_json = requests.get(f"{base}/players/nfl", timeout=25).json()
    except Exception:
        players_json = {}

    id_to_name = {}
    for pid in all_player_ids:
        info = players_json.get(pid) or {}
        name = info.get("full_name")
        if not name:
            first = info.get("first_name")
            last = info.get("last_name")
            if first or last:
                name = f"{first or ''} {last or ''}".strip()
        if name:
            id_to_name[pid] = name

    return league, teams, id_to_name, traded_picks


def map_rosters_to_fp(fp: pd.DataFrame, teams, id_to_name):
    """Return dict team_name -> list of FP player names on that team."""
    if not teams or not id_to_name:
        return {}

    fp_keys = fp.set_index("name_key")
    roster_map: dict[str, list[str]] = {}

    for t in teams:
        pname_list = []
        for pid in t["player_ids"]:
            raw_name = id_to_name.get(pid)
            if not raw_name:
                continue
            key = normalize_name(raw_name)

            # direct match
            if key in fp_keys.index:
                pname_list.append(fp_keys.loc[key, "PlayerName"])
            else:
                # fuzzy match on normalized name
                close = difflib.get_close_matches(key, fp_keys.index, n=1, cutoff=0.80)
                if close:
                    pname_list.append(fp_keys.loc[close[0], "PlayerName"])

        roster_map[t["team_name"]] = sorted(set(pname_list))

    return roster_map


def build_team_power_df(fp: pd.DataFrame, teams, roster_map: dict) -> pd.DataFrame:
    """Estimate team strength for pick valuation."""
    rows = []
    for t in teams:
        team_name = t["team_name"]
        names = roster_map.get(team_name, [])
        sub = fp[fp["PlayerName"].isin(names)].sort_values("ValueIndex", ascending=False)
        core = sub.head(12)["ValueIndex"].sum()
        wins = t["wins"] or 0
        losses = t["losses"] or 0
        ties = t["ties"] or 0
        games = wins + losses + ties
        win_pct = (wins + 0.5 * ties) / games if games > 0 else 0.5
        rows.append(
            {
                "team_id": t["team_id"],
                "team_name": team_name,
                "wins": wins,
                "losses": losses,
                "ties": ties,
                "points_for": t["points_for"],
                "core_value": float(core),
                "win_pct": win_pct,
            }
        )
    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    max_core = df["core_value"].max() or 1.0
    df["core_norm"] = df["core_value"] / max_core
    df["power"] = 0.6 * df["win_pct"] + 0.4 * df["core_norm"]
    # lower power => worse team => earlier pick
    df = df.sort_values("power", ascending=False).reset_index(drop=True)
    # best team index 0 -> slot = N; worst team index N-1 -> slot = 1
    n = len(df)
    df["slot"] = n - df.index
    return df


def build_pick_ownership(league, teams, traded_picks):
    """Return (team_picks_map, pick_label_meta)."""
    if not teams:
        return {}, {}

    season_now = int(league.get("season", CURRENT_SEASON))
    draft_rounds = int(league.get("draft_rounds", 4))

    # Only future drafts (no 2025 if current season is 2025)
    future_seasons = [season_now + i for i in (1, 2, 3)]

    roster_ids = [t["team_id"] for t in teams]
    # default: each roster owns its own future picks
    pick_owner = {}
    for s in future_seasons:
        for rnd in range(1, draft_rounds + 1):
            for rid in roster_ids:
                pick_owner[(s, rnd, rid)] = rid

    # apply traded picks final ownership
    for tp in traded_picks or []:
        try:
            s = int(tp.get("season"))
            rnd = int(tp.get("round"))
        except Exception:
            continue
        if s not in future_seasons:
            continue
        orig_rid = tp.get("roster_id")
        new_owner_rid = tp.get("owner_id")
        if orig_rid is None or new_owner_rid is None:
            continue
        if (s, rnd, orig_rid) not in pick_owner:
            continue
        pick_owner[(s, rnd, orig_rid)] = new_owner_rid

    # Map roster_id -> team name
    roster_to_name = {t["team_id"]: t["team_name"] for t in teams}

    team_picks = {t["team_name"]: [] for t in teams}
    pick_label_meta = {}

    for (s, rnd, orig_rid), owner_rid in pick_owner.items():
        owner_name = roster_to_name.get(owner_rid)
        orig_name = roster_to_name.get(orig_rid)
        if not owner_name or not orig_name:
            continue
        if rnd == 1:
            suf = "1st"
        elif rnd == 2:
            suf = "2nd"
        elif rnd == 3:
            suf = "3rd"
        else:
            suf = f"{rnd}th"
        label = f"{s} {suf} ({orig_name})"
        team_picks.setdefault(owner_name, []).append(label)
        pick_label_meta[label] = (s, rnd, orig_rid)

    for k in team_picks:
        team_picks[k] = sorted(team_picks[k])

    return team_picks, pick_label_meta


def pick_value_from_meta(label: str, pick_label_meta: dict, power_df: pd.DataFrame) -> float:
    """Numeric value for a pick label."""
    meta = pick_label_meta.get(label)
    if not meta:
        return 0.0
    season, rnd, orig_rid = meta

    # base by round – 1sts strong, later rounds fall off fast
    base_curve = {1: 1000.0, 2: 420.0, 3: 180.0, 4: 80.0}
    base = base_curve.get(rnd, 20.0)

    if power_df is None or power_df.empty:
        slot_mult = 1.0
    else:
        row = power_df[power_df["team_id"] == orig_rid]
        if row.empty:
            slot_mult = 1.0
        else:
            slot = float(row["slot"].iloc[0])
            n = len(power_df)
            # slot 1 (worst team) -> 1.25; slot n (best team) -> 0.75
            if n <= 1:
                slot_mult = 1.0
            else:
                slot_mult = 1.25 - 0.5 * ((slot - 1) / (n - 1))
                slot_mult = max(0.70, min(1.30, slot_mult))

    # time discount: further-out picks slightly less valuable
    year_diff = max(0, season - CURRENT_SEASON)
    time_mult = 0.95 ** year_diff

    return base * slot_mult * time_mult


def team_strength_from_roster(fp: pd.DataFrame, names: list[str]) -> float:
    if not names:
        return 0.0
    sub = fp[fp["PlayerName"].isin(names)]
    return float(sub["ValueIndex"].sum())


def describe_package(fp: pd.DataFrame, players: list[str], picks: list[str]) -> str:
    parts = []
    if players:
        sub = fp[fp["PlayerName"].isin(players)].sort_values("ValueIndex", ascending=False)
        for _, row in sub.iterrows():
            parts.append(f"{row['PlayerName']} ({row['Pos']} #{int(row['Rank'])})")
    if picks:
        parts.append("Picks: " + ", ".join(picks))
    return ", ".join(parts) if parts else "nothing"


def classify_trade(
    A_get: float,
    B_get: float,
) -> str:
    """Return one of the buckets: Perfect Fit / Reasonable / Questionable / Not Good / Call the Commissioner."""
    bigger = max(A_get, B_get, 1.0)
    gap = abs(A_get - B_get)
    pct = gap / bigger

    if pct < 0.03:
        return "Perfect Fit"
    elif pct < 0.18:
        return "Reasonable"
    elif pct < 0.30:
        return "Questionable"
    elif pct < 0.50:
        return "Not Good"
    else:
        return "Call the Commissioner if They Accept"


def analysis_blurb(
    bucket: str,
    teamA: str,
    teamB: str,
    A_get: float,
    B_get: float,
    A_desc: str,
    B_desc: str,
) -> str:
    bigger = max(A_get, B_get, 1.0)
    gap = abs(A_get - B_get)
    pct = gap / bigger * 100.0

    if A_get > B_get:
        ahead_team, behind_team = teamA, teamB
    elif B_get > A_get:
        ahead_team, behind_team = teamB, teamA
    else:
        ahead_team, behind_team = None, None

    header = f"**Analysis: {bucket}**"

    if bucket == "Perfect Fit":
        core = (
            "This looks about as even as trades get. The value gap between the sides is tiny, "
            "and both teams are moving pieces in a way that most managers would consider fair."
        )
    elif bucket == "Reasonable":
        core = (
            "This trade falls into the range of deals that are commonly accepted in real leagues. "
            "One side is ahead by a bit, but not so much that it would be shocking if both managers agree."
        )
    elif bucket == "Questionable":
        core = (
            "The value gap is noticeable. This doesn't mean it could never be accepted, "
            "but it would probably require one manager to like their side more than the market does."
        )
    elif bucket == "Not Good":
        core = (
            "This looks like a clearly lopsided deal. It's not impossible, but it would usually raise eyebrows "
            "and might lead to pushback from the league."
        )
    else:
        core = (
            "This is the kind of offer that usually leads to a group chat meltdown. "
            "If this got accepted as-is, most leagues would at least talk about vetoing it."
        )

    lines = [header, "", core, ""]
    lines.append(
        f"- **{teamA} receives:** {A_desc} (value ~{A_get:,.0f})  \n"
        f"- **{teamB} receives:** {B_desc} (value ~{B_get:,.0f})"
    )

    if ahead_team:
        lines.append(
            f"- The side for **{ahead_team}** is ahead by about **{gap:,.0f} value points**, "
            f"roughly **{pct:.1f}%** more than the other side."
        )

    lines.append(
        "- Remember: these numbers are a guide, not a verdict. Real managers might value youth, safety, or upside "
        "very differently."
    )
    return "\n".join(lines)


# ---------- Build core data ----------

fp_df = build_player_values()

league, teams, id_to_name, traded_picks = fetch_sleeper_data(LEAGUE_ID)
roster_map = map_rosters_to_fp(fp_df, teams or [], id_to_name or {})

team_power_df = build_team_power_df(fp_df, teams or [], roster_map)
team_picks_map, pick_label_meta = build_pick_ownership(league or {}, teams or [], traded_picks or [])

team_names = sorted(roster_map.keys()) if roster_map else []

league_logo_url = None
if league and league.get("avatar"):
    league_logo_url = f"https://sleepercdn.com/avatars/{league['avatar']}"

# ---------- UI: Header ----------

header_cols = st.columns([1, 6])
with header_cols[0]:
    if league_logo_url:
        st.image(league_logo_url, width=64)
with header_cols[1]:
    st.markdown("### Dynasty Bros. Trade Calculator")
    st.caption(
        "FantasyPros Dynasty Superflex PPR rankings + age + position curves + live Sleeper rosters & future picks."
    )

st.markdown("---")

tabs = st.tabs(["Trade Calculator", "Trade Finder"])


# ---------- Trade Calculator Tab ----------

with tabs[0]:
    st.subheader("Trade Calculator")

    if not team_names:
        st.warning("Could not load Sleeper rosters. Player values will still show, but team-specific tools are limited.")

    colA, colB = st.columns(2)

    with colA:
        teamA = st.selectbox("Team A", team_names, index=0 if team_names else None)
        playersA = roster_map.get(teamA, fp_df["PlayerName"].tolist())
        giveA_players = st.multiselect("Team A gives (players)", playersA, key="A_give_players")
        getA_players = st.multiselect("Team A gets (players)", fp_df["PlayerName"].tolist(), key="A_get_players")

        picksA_owned = team_picks_map.get(teamA, [])
        giveA_picks = st.multiselect("Team A gives (picks)", picksA_owned, key="A_give_picks")
        getA_picks = st.multiselect(
            "Team A gets (picks)",
            [p for p in sum(team_picks_map.values(), []) if p not in picksA_owned],
            key="A_get_picks",
        )

    with colB:
        default_idx_B = 1 if len(team_names) > 1 else 0
        teamB = st.selectbox("Team B", team_names, index=default_idx_B)
        playersB = roster_map.get(teamB, fp_df["PlayerName"].tolist())
        giveB_players = st.multiselect("Team B gives (players)", playersB, key="B_give_players")
        getB_players = st.multiselect("Team B gets (players)", fp_df["PlayerName"].tolist(), key="B_get_players")

        picksB_owned = team_picks_map.get(teamB, [])
        giveB_picks = st.multiselect("Team B gives (picks)", picksB_owned, key="B_give_picks")
        getB_picks = st.multiselect(
            "Team B gets (picks)",
            [p for p in sum(team_picks_map.values(), []) if p not in picksB_owned],
            key="B_get_picks",
        )

    def total_player_value(names: list[str]) -> float:
        if not names:
            return 0.0
        sub = fp_df[fp_df["PlayerName"].isin(names)]
        return float(sub["ValueIndex"].sum())

    def total_pick_value(labels: list[str]) -> float:
        return float(sum(pick_value_from_meta(lbl, pick_label_meta, team_power_df) for lbl in labels))

    # A / B give / get totals
    val_A_gives = total_player_value(giveA_players) + total_pick_value(giveA_picks)
    val_A_gets = total_player_value(getA_players) + total_pick_value(getA_picks)
    val_B_gives = total_player_value(giveB_players) + total_pick_value(giveB_picks)
    val_B_gets = total_player_value(getB_players) + total_pick_value(getB_picks)

    m1, m2, m3, m4 = st.columns(4)
    m1.metric(f"{teamA} gives", f"{val_A_gives:,.0f}")
    m2.metric(f"{teamA} gets", f"{val_A_gets:,.0f}")
    m3.metric(f"{teamB} gives", f"{val_B_gives:,.0f}")
    m4.metric(f"{teamB} gets", f"{val_B_gets:,.0f}")

    st.markdown("---")
    st.subheader("Analysis")

    if (
        not giveA_players
        and not giveA_picks
        and not giveB_players
        and not giveB_picks
        and not getA_players
        and not getA_picks
        and not getB_players
        and not getB_picks
    ):
        st.write("Build a trade by adding players and/or picks to each side to see an analysis.")
    else:
        bucket = classify_trade(val_A_gets, val_B_gets)
        A_desc = describe_package(fp_df, getA_players, getA_picks)
        B_desc = describe_package(fp_df, getB_players, getB_picks)
        blurb = analysis_blurb(bucket, teamA, teamB, val_A_gets, val_B_gets, A_desc, B_desc)
        st.markdown(blurb)

        st.caption(
            "Picks are valued by round, how strong the original team looks (record + core roster), "
            "and how far out the pick is (further years get a small discount)."
        )


# ---------- Trade Finder Tab ----------

with tabs[1]:
    st.subheader("Trade Finder")

    if not team_names:
        st.warning("Trade Finder needs live Sleeper rosters + picks to work.")
    else:
        tf_team = st.selectbox("Your team", team_names, key="tf_team")
        tf_players = roster_map.get(tf_team, [])
        tf_picks = team_picks_map.get(tf_team, [])

        other_teams = [t for t in team_names if t != tf_team]

        st.markdown("### 1) Acquire Position / Pick")

        col_pos1, col_pos2 = st.columns(2)
        with col_pos1:
            desired_positions = st.multiselect(
                "Positions you're interested in acquiring",
                ["QB", "RB", "WR", "TE"],
                default=["RB", "WR"],
            )
        with col_pos2:
            desired_pick_rounds = st.multiselect(
                "Pick rounds you're interested in acquiring",
                [1, 2, 3, 4],
                default=[1],
            )

        if st.button("Suggest offers to acquire these types of assets"):
            suggestions = []

            # build candidate outgoing assets from your team
            your_assets = []
            for p in tf_players:
                sub = fp_df[fp_df["PlayerName"] == p]
                if sub.empty:
                    continue
                v = float(sub["ValueIndex"].iloc[0])
                your_assets.append(("player", p, v))
            for pk in tf_picks:
                v = pick_value_from_meta(pk, pick_label_meta, team_power_df)
                if v > 0:
                    your_assets.append(("pick", pk, v))

            # limit to top 15 by value
            your_assets.sort(key=lambda x: x[2], reverse=True)
            your_assets = your_assets[:15]

            # for each other team, look for assets matching desired criteria
            for ot in other_teams:
                their_players = roster_map.get(ot, [])
                their_picks = team_picks_map.get(ot, [])

                # candidate incoming players by position
                for name in their_players:
                    row = fp_df[fp_df["PlayerName"] == name]
                    if row.empty:
                        continue
                    pos = row["Pos"].iloc[0]
                    if pos not in desired_positions:
                        continue
                    v_target = float(row["ValueIndex"].iloc[0])

                    # try your 1- and 2-asset combos
                    for i, (k1, a1, v1) in enumerate(your_assets):
                        if 0.8 * v_target <= v1 <= 1.2 * v_target:
                            suggestions.append(
                                f"{tf_team} sends **{a1}** (~{v1:,.0f}) to {ot} for **{name} ({pos})** (~{v_target:,.0f})."
                            )
                        for j in range(i + 1, len(your_assets)):
                            k2, a2, v2 = your_assets[j]
                            v_sum = v1 + v2
                            if 0.8 * v_target <= v_sum <= 1.2 * v_target:
                                suggestions.append(
                                    f"{tf_team} sends **{a1}** and **{a2}** (~{v_sum:,.0f}) to {ot} for **{name} ({pos})** (~{v_target:,.0f})."
                                )

                # candidate incoming picks by round
                for pk in their_picks:
                    meta = pick_label_meta.get(pk)
                    if not meta:
                        continue
                    _, rnd, _ = meta
                    if rnd not in desired_pick_rounds:
                        continue
                    v_target = pick_value_from_meta(pk, pick_label_meta, team_power_df)
                    for i, (k1, a1, v1) in enumerate(your_assets):
                        if 0.8 * v_target <= v1 <= 1.2 * v_target:
                            suggestions.append(
                                f"{tf_team} sends **{a1}** (~{v1:,.0f}) to {ot} for **{pk}** (~{v_target:,.0f})."
                            )
                        for j in range(i + 1, len(your_assets)):
                            k2, a2, v2 = your_assets[j]
                            v_sum = v1 + v2
                            if 0.8 * v_target <= v_sum <= 1.2 * v_target:
                                suggestions.append(
                                    f"{tf_team} sends **{a1}** and **{a2}** (~{v_sum:,.0f}) to {ot} for **{pk}** (~{v_target:,.0f})."
                                )

            suggestions = list(dict.fromkeys(suggestions))  # de-duplicate
            if not suggestions:
                st.info("I couldn't find any simple 1–2 piece offers that land in a reasonable value range.")
            else:
                st.markdown("**Top suggestion ideas:**")
                for s in suggestions[:3]:
                    st.markdown(f"- {s}")

        st.markdown("---")
        st.markdown("### 2) Trade Away")

        trade_away_players = st.multiselect(
            "Players you're open to trading away",
            tf_players,
            key="tf_away_players",
        )
        trade_away_picks = st.multiselect(
            "Picks you're open to trading away",
            tf_picks,
            key="tf_away_picks",
        )

        if st.button("Suggest what you could reasonably get back"):
            send_val = total_player_value(trade_away_players) + total_pick_value(trade_away_picks)
            if send_val <= 0:
                st.info("Select at least one player or pick with value.")
            else:
                suggestions = []
                for ot in other_teams:
                    their_players = roster_map.get(ot, [])
                    their_picks = team_picks_map.get(ot, [])

                    # 1-for-1 players
                    for name in their_players:
                        v_target = total_player_value([name])
                        if 0.8 * send_val <= v_target <= 1.2 * send_val:
                            suggestions.append(
                                f"{tf_team} sends **{', '.join(trade_away_players + trade_away_picks)}** (~{send_val:,.0f}) "
                                f"to {ot} for **{name}** (~{v_target:,.0f})."
                            )

                    # 1-for-1 picks
                    for pk in their_picks:
                        v_target = pick_value_from_meta(pk, pick_label_meta, team_power_df)
                        if 0.8 * send_val <= v_target <= 1.2 * send_val:
                            suggestions.append(
                                f"{tf_team} sends **{', '.join(trade_away_players + trade_away_picks)}** (~{send_val:,.0f}) "
                                f"to {ot} for **{pk}** (~{v_target:,.0f})."
                            )

                suggestions = list(dict.fromkeys(suggestions))
                if not suggestions:
                    st.info("I couldn't find any simple 1-for-1 ideas near that value range. You may need to ask for multiple pieces back.")
                else:
                    st.markdown("**Top suggestion ideas:**")
                    for s in suggestions[:3]:
                        st.markdown(f"- {s}")

st.markdown("---")
st.caption(
    "Player values are derived from FantasyPros Dynasty Superflex PPR rankings plus PPR scoring curves by position, "
    "with a light age adjustment. Picks are valued by round, projected finish of the original team, and how far into "
    "the future they are."
)
