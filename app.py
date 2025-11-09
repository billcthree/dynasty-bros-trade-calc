import math
import difflib
from functools import lru_cache

import pandas as pd
import numpy as np
import requests
import streamlit as st


LEAGUE_ID = "1194681871141023744"

st.set_page_config(
    page_title="Dynasty Bros. Trade Calculator",
    layout="wide",
)

# -------------------------
# Helpers
# -------------------------


def normalize_name(name: str) -> str:
    """Lowercase, strip punctuation, drop Jr/Sr/II/III etc to help match names."""
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
    df["PLAYER NAME"] = df["PLAYER NAME"].astype(str).str.strip()
    df["name_key"] = df["PLAYER NAME"].map(normalize_name)
    df["AGE"] = pd.to_numeric(df["AGE"], errors="coerce")
    return df


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
        if "#" not in df.columns or "MDN" not in df.columns:
            continue

        df = df.rename(columns={"#": "ScoreRank"})[["ScoreRank", "MDN"]]
        df = df.dropna(subset=["ScoreRank", "MDN"]).copy()
        df["ScoreRank"] = pd.to_numeric(df["ScoreRank"], errors="coerce")
        df["MDN"] = pd.to_numeric(df["MDN"], errors="coerce")
        df = df.dropna(subset=["ScoreRank", "MDN"]).sort_values("ScoreRank")
        if df.empty:
            continue

        # replacement-level rank per position (how many start in a typical league)
        repl_rank = {"QB": 24, "RB": 30, "WR": 40, "TE": 18}.get(pos, 30)
        max_rank = int(df["ScoreRank"].max())
        rr = min(repl_rank, max_rank)
        repl_mdn = float(df.loc[df["ScoreRank"] == rr, "MDN"].iloc[0])

        ranks = df["ScoreRank"].to_numpy()
        mdn = df["MDN"].to_numpy()

        def vorp_for_pos_rank(pos_rank: int) -> float:
            """Approximate VORP at this positional rank using the PPR curve."""
            if pos_rank <= ranks.min():
                m = mdn[0]
            elif pos_rank >= ranks.max():
                m = mdn[-1]
            else:
                # linear interpolate between neighbor rows
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
    """Apply a small age bump / haircut (a bit stronger for RB)."""
    if age is None or (isinstance(age, float) and np.isnan(age)):
        return base

    if age <= 23:
        mult = 1.12
    elif age <= 26:
        mult = 1.07
    elif age <= 29:
        mult = 1.00
    elif age <= 31:
        mult = 0.94
    else:
        mult = 0.88

    # Age matters more for RBs
    if pos == "RB":
        mult = 1 + (mult - 1) * 1.25

    return base * mult


def build_player_values() -> pd.DataFrame:
    """
    Build the core player table with a ValueIndex column.

    ValueIndex is based on:
      - FantasyPros Dynasty Superflex rank per position
      - PPR median scoring drop-off curves by position
      - Light age adjustments
      - A small position weight (QB > RB > WR > TE)
    """
    fp = load_fp_ranks()
    age = load_age()
    curves = load_ppr_curves()

    # merge age by normalized name
    if age is not None and not age.empty:
        fp = fp.merge(
            age[["name_key", "AGE"]],
            on="name_key",
            how="left",
        )
    else:
        fp["AGE"] = np.nan

    # positional rank (1st WR, 2nd WR, etc.) based on FantasyPros overall rank
    fp = fp.sort_values("Rank").copy()
    fp["PosRank"] = fp.groupby("Pos").cumcount() + 1

    pos_weights = {"QB": 1.25, "RB": 1.10, "WR": 1.00, "TE": 0.70}
    scores = []

    for _, row in fp.iterrows():
        pos = row["Pos"]
        pos_rank = int(row["PosRank"])

        # Core VORP from PPR curves – makes the top few guys pop
        if pos in curves:
            base_vorp = curves[pos](pos_rank)
        else:
            # Fallback: steeper curve at the very top
            base_vorp = max(0.0, 200.0 / math.sqrt(pos_rank))

        pos_mult = pos_weights.get(pos, 1.0)
        score = base_vorp * pos_mult * 10.0  # scaled to "nice" numbers

        score = apply_age_multiplier(score, row.get("AGE", np.nan), pos)
        scores.append(score)

    fp["BaseValue"] = scores

    # normalize to ~0–1000 scale, mostly for human readability
    max_val = fp["BaseValue"].max()
    if max_val and max_val > 0:
        fp["ValueIndex"] = (fp["BaseValue"] / max_val) * 1000.0
    else:
        fp["ValueIndex"] = 0.0

    # friendly columns
    fp = fp.rename(columns={"Player": "PlayerName"})
    fp["Player"] = fp["PlayerName"]  # convenience alias
    return fp


# ---------- Sleeper integration ----------


@st.cache_data(show_spinner=False)
def fetch_sleeper_league(league_id: str):
    base = "https://api.sleeper.app/v1"
    try:
        users = requests.get(f"{base}/league/{league_id}/users", timeout=10).json()
        rosters = requests.get(f"{base}/league/{league_id}/rosters", timeout=10).json()
    except Exception as e:
        st.warning(f"Could not reach Sleeper API: {e}")
        return None, None, None

    # build team metadata
    user_by_id = {}
    for u in users:
        uid = u.get("user_id")
        meta = u.get("metadata") or {}
        display = (
            meta.get("team_name")
            or meta.get("nickname")
            or u.get("display_name")
            or f"Team {uid}"
        )
        avatar = u.get("avatar")
        user_by_id[uid] = {
            "display_name": display,
            "avatar": avatar,
        }

    # pull player ids owned & record info
    all_player_ids = set()
    teams = []
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

        u = user_by_id.get(owner_id, {})
        teams.append(
            {
                "team_id": roster_id,
                "owner_id": owner_id,
                "team_name": u.get("display_name", f"Team {roster_id}"),
                "avatar": u.get("avatar"),
                "wins": wins,
                "losses": losses,
                "ties": ties,
                "points_for": points_for,
                "player_ids": players,
            }
        )

    # fetch only relevant player metadata
    try:
        players_json = requests.get(f"{base}/players/nfl", timeout=20).json()
    except Exception:
        players_json = {}

    id_to_name = {}
    for pid in all_player_ids:
        info = players_json.get(pid) or {}
        name = info.get("full_name") or info.get("first_name")
        if name and info.get("last_name"):
            name = f"{info['first_name']} {info['last_name']}"
        if not name:
            name = info.get("last_name")
        if name:
            id_to_name[pid] = name

    return teams, id_to_name, user_by_id


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
                close = difflib.get_close_matches(key, fp_keys.index, n=1, cutoff=0.8)
                if close:
                    pname_list.append(fp_keys.loc[close[0], "PlayerName"])

        roster_map[t["team_name"]] = sorted(set(pname_list))

    return roster_map


def team_strength(fp: pd.DataFrame, names: list[str]) -> float:
    """Rough team strength: sum of top 12 ValueIndex players on the roster."""
    if not names:
        return 0.0
    sub = fp[fp["PlayerName"].isin(names)]
    sub = sub.sort_values("ValueIndex", ascending=False).head(12)
    return float(sub["ValueIndex"].sum())


def total_player_value(fp: pd.DataFrame, names: list[str]) -> float:
    if not names:
        return 0.0
    sub = fp[fp["PlayerName"].isin(names)]
    return float(sub["ValueIndex"].sum())


def describe_package(fp: pd.DataFrame, names: list[str]) -> str:
    if not names:
        return "no players"
    sub = fp[fp["PlayerName"].isin(names)].sort_values("ValueIndex", ascending=False)
    parts = []
    for _, row in sub.iterrows():
        parts.append(f"{row['PlayerName']} ({row['Pos']} #{int(row['Rank'])})")
    return ", ".join(parts)


# ---------- Build data ----------

fp_df = build_player_values()
age_note = (
    "Age is included lightly: younger RBs/WRs/QBs get a small bump; "
    "older options get a slight haircut."
)

st.markdown(
    """
    <h3 style="margin-bottom:0.25rem;">Dynasty Bros. Trade Calculator</h3>
    <small>Uses FantasyPros latest <strong>Dynasty Superflex PPR</strong> rankings plus light age + position context.</small>
    """,
    unsafe_allow_html=True,
)

st.info(
    "Numbers shown are on an arbitrary **value index** scale (higher is better). "
    "Differences of ~150–200 usually separate a solid starter from a bench piece; "
    "bigger gaps indicate more one-sided trades."
)

tabs = st.tabs(["Trade Calculator", "Trade Finder"])

teams, id_to_name, users = fetch_sleeper_league(LEAGUE_ID)
roster_map = map_rosters_to_fp(fp_df, teams or [], id_to_name or {})

team_names = sorted(roster_map.keys()) if roster_map else []

# ---------- Trade Calculator ----------

with tabs[0]:
    st.subheader("Trade Calculator")

    if not team_names:
        st.warning(
            "Could not load Sleeper rosters. You can still see player values below."
        )

    colA, colB = st.columns(2)

    with colA:
        teamA = st.selectbox("Team A", team_names, index=0 if team_names else None)
        playersA = roster_map.get(teamA, fp_df["PlayerName"].tolist())
        giveA = st.multiselect("Team A gives (players)", playersA, key="A_give")
        getA = st.multiselect(
            "Team A gets (players)", fp_df["PlayerName"].tolist(), key="A_get"
        )

    with colB:
        default_idx_B = 1 if len(team_names) > 1 else 0
        teamB = st.selectbox("Team B", team_names, index=default_idx_B)
        playersB = roster_map.get(teamB, fp_df["PlayerName"].tolist())
        giveB = st.multiselect("Team B gives (players)", playersB, key="B_give")
        getB = st.multiselect(
            "Team B gets (players)", fp_df["PlayerName"].tolist(), key="B_get"
        )

    val_A_gives = total_player_value(fp_df, giveA)
    val_A_gets = total_player_value(fp_df, getA)
    val_B_gives = total_player_value(fp_df, giveB)
    val_B_gets = total_player_value(fp_df, getB)

    m1, m2, m3, m4 = st.columns(4)
    m1.metric(f"{teamA} gives", f"{val_A_gives:,.0f}")
    m2.metric(f"{teamA} gets", f"{val_A_gets:,.0f}")
    m3.metric(f"{teamB} gives", f"{val_B_gives:,.0f}")
    m4.metric(f"{teamB} gets", f"{val_B_gets:,.0f}")

    st.markdown("---")
    st.subheader("Fairness & Explanation")

    if not giveA and not getA and not giveB and not getB:
        st.write("Start by selecting players on each side to see a verdict.")
    else:
        gainA = val_A_gets - val_A_gives
        gainB = val_B_gets - val_B_gives
        total_val = max(val_A_gets + val_B_gets, 1.0)
        diff = gainA - gainB
        pct = diff / total_val

        if abs(diff) < 60:
            verdict = "This looks roughly even in pure value."
        elif diff > 0:
            verdict = (
                f"This trade likely **favors {teamA}** by about {abs(diff):.0f} value "
                f"(~{abs(pct) * 100:.1f}% of the total in the deal)."
            )
        else:
            verdict = (
                f"This trade likely **favors {teamB}** by about {abs(diff):.0f} value "
                f"(~{abs(pct) * 100:.1f}% of the total in the deal)."
            )

        st.write(verdict)

        expl_parts = []
        if giveA or getA:
            expl_parts.append(
                f"- **{teamA}** sends: {describe_package(fp_df, giveA)};\n"
                f"  receives: {describe_package(fp_df, getA)}."
            )
        if giveB or getB:
            expl_parts.append(
                f"- **{teamB}** sends: {describe_package(fp_df, giveB)};\n"
                f"  receives: {describe_package(fp_df, getB)}."
            )
        st.markdown("\n".join(expl_parts))

        st.caption(
            "Remember: this is a **guide**, not gospel. League context, manager preferences, "
            "and risk tolerance all matter. The model leans heavily on FantasyPros rank + "
            "positional scoring curves, with only a light adjustment for age."
        )

# ---------- Trade Finder ----------

with tabs[1]:
    st.subheader("Trade Finder (beta)")
    st.write(
        "Pick a team and a player you’d like to **acquire**. "
        "The tool will suggest rough packages from your roster with similar total value."
    )

    tf_col1, tf_col2 = st.columns(2)
    with tf_col1:
        tf_team = st.selectbox("Your team", team_names, key="tf_team")
        your_players = roster_map.get(tf_team, [])
    with tf_col2:
        target_player = st.selectbox(
            "Player you want to acquire", fp_df["PlayerName"].tolist(), key="tf_target"
        )

    if not your_players:
        st.warning("Could not find your roster from Sleeper. Trade Finder needs rosters to work.")
    else:
        target_val = total_player_value(fp_df, [target_player])
        st.write(
            f"Target player **{target_player}** has model value ≈ **{target_val:,.0f}**."
        )

        # candidate packages: all 1-for-1 plus 2-for-1 combos from your team
        your_sub = (
            fp_df[fp_df["PlayerName"].isin(your_players)]
            .sort_values("ValueIndex", ascending=False)
        )
        candidates = []
        names = your_sub["PlayerName"].tolist()
        vals = dict(zip(your_sub["PlayerName"], your_sub["ValueIndex"]))

        for n in names:
            candidates.append(([n], vals[n]))
        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                ns = [names[i], names[j]]
                candidates.append((ns, vals[names[i]] + vals[names[j]]))

        # score by closeness to target
        scored = []
        for pkg, v in candidates:
            scored.append((abs(v - target_val), v, pkg))
        scored.sort(key=lambda x: x[0])

        best = scored[:5]
        if not best:
            st.info("No reasonable suggestions found.")
        else:
            st.markdown("**Suggested packages you could offer:**")
            for diff_val, v, pkg in best:
                rel = (v - target_val) / target_val if target_val else 0
                direction = "slightly more" if v > target_val else "slightly less"
                st.write(
                    f"- {', '.join(pkg)} (value {v:,.0f}, {direction} than {target_player}, "
                    f"off by ~{abs(rel) * 100:.1f}%)."
                )

st.markdown("---")
st.caption(
    "Data sources: FantasyPros Dynasty Superflex PPR rankings + historical PPR scoring by position. "
    "Values are an approximate index that tries to reflect how quickly production drops off at each position."
)
