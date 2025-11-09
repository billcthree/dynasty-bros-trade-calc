import streamlit as st
import pandas as pd
import numpy as np
import requests
from functools import lru_cache

st.set_page_config(page_title="Dynasty Bros. Trade Calculator", layout="wide")

FANTASYPROS_URL = "https://www.fantasypros.com/nfl/rankings/dynasty-superflex.php"
SLEEPER_LEAGUE_ID = "1194681871141023744"
SLEEPER_API_BASE = "https://api.sleeper.app/v1"
CURRENT_SEASON = 2025  # used only to decide which future pick seasons to show
FUTURE_PICK_SEASONS = [2026, 2027, 2028]
PICK_ROUNDS = [1, 2, 3, 4]

# ---------- Helpers ----------

def canon_name(name: str) -> str:
    """Normalize names so Brian Thomas Jr. ≈ Brian Thomas, etc."""
    if not isinstance(name, str):
        return ""
    n = name.lower().strip()
    for suf in [" jr.", " sr.", " jr", " sr", " ii", " iii", " iv"]:
        if n.endswith(suf):
            n = n[: -len(suf)]
    bad = ".'`\"-"
    for ch in bad:
        n = n.replace(ch, " ")
    n = " ".join(n.split())
    return n

@st.cache_data(show_spinner=False)
def fetch_json(url: str):
    resp = requests.get(url, timeout=10)
    resp.raise_for_status()
    return resp.json()

@st.cache_data(show_spinner=False)
def fetch_text(url: str):
    resp = requests.get(url, timeout=10, headers={"User-Agent": "dynasty-bros-trade-calc"})
    resp.raise_for_status()
    return resp.text

@st.cache_data(show_spinner=False)
def load_fantasypros_table():
    """
    Try live FantasyPros. Fall back to local CSV if needed.
    Returns a DataFrame with at least: Player, Pos, Rank.
    """
    df = None
    # 1) live
    try:
        html = fetch_text(FANTASYPROS_URL)
        tables = pd.read_html(html)
        if tables:
            df = tables[0]
    except Exception:
        df = None

    # 2) local CSV fallback
    if df is None:
        try:
            df = pd.read_csv("data/player_ranks.csv")
        except Exception:
            return pd.DataFrame(columns=["Player", "Pos", "Rank"])

    cols = [str(c) for c in df.columns]
    lower = [c.lower() for c in cols]

    # find player column
    player_col = None
    for c in cols:
        if "player" in c.lower():
            player_col = c
            break
    if player_col is None and "Player" in cols:
        player_col = "Player"

    # find rank col
    rank_col = None
    for c in cols:
        lc = c.lower()
        if "rank" in lc or lc == "#" or lc == "ovr":
            rank_col = c
            break
    if rank_col is None:
        rank_col = cols[0]

    # find position col
    pos_col = None
    for c in cols:
        if "pos" in c.lower():
            pos_col = c
            break

    # if we cannot find player column, just assume CSV is already formatted
    if player_col is None or pos_col is None:
        out = df.copy()
        if "Rank" not in out.columns and rank_col in out.columns:
            out = out.rename(columns={rank_col: "Rank"})
        if "Player" not in out.columns and player_col is not None:
            out = out.rename(columns={player_col: "Player"})
        if "Pos" not in out.columns and pos_col is not None:
            out = out.rename(columns={pos_col: "Pos"})
        keep = [c for c in ["Player", "Pos", "Rank"] if c in out.columns]
        return out[keep]

    out = df[[player_col, pos_col, rank_col]].copy()
    out.columns = ["Player", "Pos", "Rank"]
    # Pos often comes like "WR1" or "WR". Strip trailing numbers
    out["Pos"] = out["Pos"].astype(str).str.extract(r"([A-Z]+)", expand=False)
    out["Rank"] = pd.to_numeric(out["Rank"], errors="coerce")
    out = out.dropna(subset=["Rank"])
    return out

@st.cache_data(show_spinner=False)
def load_age_table():
    """Optional age data (fantasyage.csv)."""
    try:
        df = pd.read_csv("data/fantasyage.csv")
    except Exception:
        return pd.DataFrame(columns=["Player", "Age"])
    if "Player" not in df.columns:
        return pd.DataFrame(columns=["Player", "Age"])
    df["Age"] = pd.to_numeric(df.get("Age"), errors="coerce")
    df["name_key"] = df["Player"].map(canon_name)
    return df[["name_key", "Age"]]

@st.cache_data(show_spinner=False)
def load_ppr_curves():
    """
    Optional: read PPR .xlsx to get how fast value falls off by position.
    If file missing, we just return None and later use hard-coded curves.
    """
    try:
        ppr = pd.read_excel("data/PPR.xlsx")
    except Exception:
        return None
    if "Pos" not in ppr.columns:
        return None
    if "Rank" not in ppr.columns:
        for c in ppr.columns:
            if "rk" in str(c).lower():
                ppr = ppr.rename(columns={c: "Rank"})
                break
    if "Rank" not in ppr.columns:
        return None
    score_col = None
    for c in ppr.columns:
        if "ppr" in str(c).lower() or "points" in str(c).lower():
            score_col = c
            break
    if score_col is None:
        return None
    ppr = ppr[["Pos", "Rank", score_col]].copy()
    ppr = ppr.dropna()
    ppr["Rank"] = pd.to_numeric(ppr["Rank"], errors="coerce")
    ppr = ppr.dropna(subset=["Rank"])
    ppr = ppr.rename(columns={score_col: "Pts"})
    curves = {}
    for pos, sub in ppr.groupby("Pos"):
        sub = sub.sort_values("Rank")
        top = sub["Pts"].iloc[0]
        if top <= 0:
            continue
        sub["Norm"] = sub["Pts"] / top
        curves[pos] = sub[["Rank", "Norm"]].reset_index(drop=True)
    return curves

@st.cache_data(show_spinner=False)
def build_player_pool():
    fp = load_fantasypros_table()
    if fp.empty:
        return pd.DataFrame()

    fp["name_key"] = fp["Player"].map(canon_name)
    fp = fp.drop_duplicates(subset=["name_key"], keep="first")

    age_df = load_age_table()
    if not age_df.empty:
        fp = fp.merge(age_df, on="name_key", how="left")
    else:
        fp["Age"] = np.nan

    fp["PosRank"] = fp.groupby("Pos")["Rank"].rank("first")

    curves = load_ppr_curves()

    def base_curve(pos, pos_rank):
        # generic fallback; top players are heavily separated
        params = {
            "QB": (525, 0.55),
            "RB": (500, 0.58),
            "WR": (480, 0.60),
            "TE": (360, 0.62),
        }
        base, exp = params.get(pos, (450, 0.60))
        return base * (pos_rank ** (-exp))

    # small extra separation for some positions
    exp_boost = {"QB": 0.0, "RB": 0.0, "WR": 0.05, "TE": 0.08}

    values = []
    for _, row in fp.iterrows():
        pos = row["Pos"]
        pr = float(row["PosRank"])
        v = base_curve(pos, pr)
        if curves is not None and pos in curves:
            curve = curves[pos]
            nearest = curve.iloc[(curve["Rank"] - pr).abs().argmin()]
            norm = float(nearest["Norm"])
            v = v * (0.7 + 0.6 * norm)
        overall = float(row["Rank"])
        if overall <= 12:
            v *= 1.30
        elif overall <= 24:
            v *= 1.18
        elif overall <= 48:
            v *= 1.08

        adj_exp = exp_boost.get(pos, 0.0)
        if adj_exp:
            v *= (pr ** (-adj_exp))

        values.append(v)

    fp["DynastyValue"] = values
    return fp

# ---------- Sleeper league data ----------

@st.cache_data(show_spinner=False)
def fetch_league_data():
    league = fetch_json(f"{SLEEPER_API_BASE}/league/{SLEEPER_LEAGUE_ID}")
    users = fetch_json(f"{SLEEPER_API_BASE}/league/{SLEEPER_LEAGUE_ID}/users")
    rosters = fetch_json(f"{SLEEPER_API_BASE}/league/{SLEEPER_LEAGUE_ID}/rosters")
    players_meta = fetch_json(f"{SLEEPER_API_BASE}/players/nfl")
    traded = fetch_json(f"{SLEEPER_API_BASE}/league/{SLEEPER_LEAGUE_ID}/traded_picks")
    return league, users, rosters, players_meta, traded

def build_team_frames(fp_players):
    league, users, rosters, players_meta, traded = fetch_league_data()

    user_map = {}
    for u in users:
        display = u.get("metadata", {}).get("team_name") or u.get("display_name") or u.get("username")
        avatar = u.get("avatar")
        user_map[u["user_id"]] = {
            "team_name": display,
            "avatar": avatar,
        }

    rows = []
    for r in rosters:
        rid = r["roster_id"]
        owner_id = r.get("owner_id")
        settings = r.get("settings", {}) or {}
        wins = settings.get("wins", 0)
        losses = settings.get("losses", 0)
        ties = settings.get("ties", 0)
        points_for = settings.get("fpts", 0) + settings.get("fpts_decimal", 0) / 100.0

        meta = user_map.get(owner_id, {"team_name": f"Team {rid}", "avatar": None})
        team_name = meta["team_name"]
        avatar = meta["avatar"]

        for pid in r.get("players", []) or []:
            p = players_meta.get(pid) or {}
            full_name = p.get("full_name") or (p.get("first_name", "") + " " + p.get("last_name", ""))
            pos = p.get("position", "")
            name_key = canon_name(full_name)
            match = fp_players.loc[fp_players["name_key"] == name_key]
            if match.empty:
                continue
            mrow = match.sort_values("Rank").iloc[0]
            rows.append(
                {
                    "sleeper_id": pid,
                    "SleeperName": full_name,
                    "Pos": pos or mrow["Pos"],
                    "TeamName": team_name,
                    "RosterID": rid,
                    "OwnerID": owner_id,
                    "avatar": avatar,
                    "FP_Player": mrow["Player"],
                    "Rank": mrow["Rank"],
                    "DynastyValue": mrow["DynastyValue"],
                    "Age": mrow.get("Age", np.nan),
                }
            )

    team_df = pd.DataFrame(rows)
    if team_df.empty:
        return team_df, pd.DataFrame(), {}, league

    team_strength = (
        team_df.groupby(["RosterID", "TeamName"])
        .agg(TotalValue=("DynastyValue", "sum"))
        .reset_index()
    )

    rec_rows = []
    for r in rosters:
        rid = r["roster_id"]
        settings = r.get("settings", {}) or {}
        wins = settings.get("wins", 0)
        losses = settings.get("losses", 0)
        ties = settings.get("ties", 0)
        points_for = settings.get("fpts", 0) + settings.get("fpts_decimal", 0) / 100.0
        rec_rows.append(
            {
                "RosterID": rid,
                "Wins": wins,
                "Losses": losses,
                "Ties": ties,
                "PointsFor": points_for,
            }
        )
    rec_df = pd.DataFrame(rec_rows)
    team_strength = team_strength.merge(rec_df, on="RosterID", how="left")

    team_strength["Power"] = (
        team_strength["TotalValue"] * 0.65
        + team_strength["PointsFor"].fillna(0) * 0.25
        + team_strength["Wins"].fillna(0) * 0.10
    )
    team_strength = team_strength.sort_values("Power", ascending=False).reset_index(drop=True)
    team_strength["LeagueRank"] = np.arange(1, len(team_strength) + 1)

    total_teams = len(team_strength)
    slot_map = {}
    for _, row in team_strength.iterrows():
        rid = row["RosterID"]
        rank = row["LeagueRank"]
        slot = total_teams - rank + 1
        slot_map[rid] = slot

    # compute current pick owners from traded_picks
    pick_owner = {}
    for season in FUTURE_PICK_SEASONS:
        for rnd in PICK_ROUNDS:
            for r in rosters:
                rid = r["roster_id"]
                pick_owner[(season, rnd, rid)] = rid

    for t in traded:
        try:
            season = int(t.get("season"))
        except Exception:
            continue
        if season not in FUTURE_PICK_SEASONS:
            continue
        rnd = t.get("round")
        if rnd not in PICK_ROUNDS:
            continue
        orig = t.get("roster_id")
        owner = t.get("owner_id")
        if orig is None or owner is None:
            continue
        pick_owner[(season, rnd, orig)] = owner

    return team_df, team_strength, pick_owner, league

# ---------- Value & pick helpers ----------

def team_pick_list(roster_id, pick_owner, team_strength):
    """Return list of tuples (season, rnd, orig_rid, label) for picks this roster currently owns."""
    picks = []
    for (season, rnd, orig_rid), owner_rid in pick_owner.items():
        if owner_rid == roster_id:
            if rnd == 1:
                suffix = "1st"
            elif rnd == 2:
                suffix = "2nd"
            elif rnd == 3:
                suffix = "3rd"
            else:
                suffix = f"{rnd}th"
            label = f"{season} {suffix}"
            picks.append((season, rnd, orig_rid, label))
    picks.sort(key=lambda x: (x[0], x[1]))
    return picks

def pick_value_numeric(season, rnd, orig_roster_id, team_strength):
    """
    Simple curve:
    - Early 1sts ~= premium stud.
    - Drops by round and by projected finish of original owner.
    """
    ts = team_strength.set_index("RosterID")
    total_teams = len(ts)
    if orig_roster_id in ts.index:
        league_rank = ts.loc[orig_roster_id, "LeagueRank"]  # 1 = strongest
        slot = total_teams - league_rank + 1  # worst team => slot 1
    else:
        slot = total_teams / 2.0

    overall_index = (rnd - 1) * total_teams + slot
    base = 420.0
    round_factor = {1: 1.0, 2: 0.55, 3: 0.30, 4: 0.15}.get(rnd, 0.1)
    decay = 0.12
    val = base * round_factor * np.exp(-decay * (overall_index - 1))
    return float(val)

def side_package_value(player_ids, pick_labels, roster_id, team_df, pick_owner, team_strength, need_weight=0.3, picks_weight=1.0):
    """Compute numeric value for a side of trade."""
    if team_df.empty:
        return 0.0, []

    team_players = team_df.set_index("sleeper_id")
    details = []

    base_val = 0.0
    need_adj_val = 0.0

    team_rows = team_df[team_df["RosterID"] == roster_id]
    pos_counts = team_rows["Pos"].value_counts().to_dict()

    def need_multiplier(pos, counts):
        target = {"QB": 2, "RB": 5, "WR": 6, "TE": 2}.get(pos, 3)
        current = counts.get(pos, 0)
        diff = current - target
        if diff <= -2:
            return 1.10
        if diff == -1:
            return 1.05
        if diff == 0:
            return 1.0
        if diff == 1:
            return 0.96
        return 0.92

    # players
    for pid in player_ids:
        if pid not in team_players.index:
            continue
        row = team_players.loc[pid]
        p_base = float(row["DynastyValue"])
        base_val += p_base
        mult = need_multiplier(row["Pos"], pos_counts)
        adj = p_base * mult
        need_adj_val += adj
        details.append(
            {
                "kind": "player",
                "name": row["SleeperName"],
                "pos": row["Pos"],
                "fp_rank": int(row["Rank"]),
                "base": p_base,
                "need_mult": mult,
                "adj": adj,
            }
        )

    # picks
    for label in pick_labels:
        label = str(label).strip()
        parts = label.split()
        if len(parts) != 2:
            continue
        season_s, round_s = parts
        try:
            season = int(season_s)
        except Exception:
            continue
        try:
            rnd = int(round_s[0])
        except Exception:
            continue
        orig_ids = [orig for (s, r, orig), owner in pick_owner.items() if s == season and r == rnd and owner == roster_id]
        if not orig_ids:
            orig = None
        else:
            orig = orig_ids[0]
        pv = pick_value_numeric(season, rnd, orig, team_strength)
        base_val += pv
        need_adj_val += pv * picks_weight
        details.append(
            {
                "kind": "pick",
                "name": label,
                "pos": "PICK",
                "fp_rank": None,
                "base": pv,
                "need_mult": picks_weight,
                "adj": pv * picks_weight,
            }
        )

    blended = (1 - need_weight) * base_val + need_weight * need_adj_val
    return blended, details

def format_value_gap(a_val, b_val, teamA_name, teamB_name):
    diff = a_val - b_val
    bigger = max(a_val, b_val, 1e-6)
    pct = abs(diff) / bigger

    if abs(diff) < bigger * 0.05:
        verdict = "This looks close to even."
    elif diff > 0:
        verdict = f"This trade likely leans toward {teamA_name}."
    else:
        verdict = f"This trade likely leans toward {teamB_name}."

    explain = f"The gap between the two sides is about **{pct*100:.1f}%** when we compare total value.\n"
    explain += "Because trades are subjective, managers might still disagree depending on their window, risk tolerance, and how they personally view the players."
    return verdict, diff, pct, explain

def summarise_roster_context(team_df, roster_id, incoming_details, outgoing_details):
    team_rows = team_df[team_df["RosterID"] == roster_id]
    if team_rows.empty:
        return "We don't have enough roster data here."

    counts = team_rows["Pos"].value_counts().to_dict()

    def pos_list(details, kind):
        return [d for d in details if d["kind"] == kind]

    inc_players = pos_list(incoming_details, "player")
    out_players = pos_list(outgoing_details, "player")

    notes = []

    for pos in ["QB", "RB", "WR", "TE"]:
        before = counts.get(pos, 0)
        delta = sum(1 for d in inc_players if d["pos"] == pos) - sum(1 for d in out_players if d["pos"] == pos)
        after = before + delta
        if before <= 1 and delta < 0:
            notes.append(f"They'd be pretty thin at **{pos}** after this deal (around {after} on the roster).")
        elif before <= 2 and pos in ["QB", "RB"] and delta < 0:
            notes.append(f"This would noticeably weaken their **{pos}** depth (down to about {after}).")
        elif before >= 4 and delta > 0 and pos in ["RB", "WR"]:
            notes.append(f"They already have plenty of **{pos}**; adding more here is probably a luxury rather than a necessity.")

    if not notes:
        notes.append("Neither side creates an obvious positional issue. Whether they accept may come down to how they feel about these specific players and their competitive window.")

    return " ".join(notes)

def build_piece_table(details):
    rows = []
    for d in details:
        if d["kind"] == "player":
            rows.append(
                {
                    "Piece": d["name"],
                    "Pos": d["pos"],
                    "FP Rank": d["fp_rank"],
                    "Base Value": round(d["base"]),
                    "Need/Pick Mult": round(d["need_mult"], 2),
                    "Adjusted Value": round(d["adj"]),
                }
            )
        else:
            rows.append(
                {
                    "Piece": d["name"],
                    "Pos": "PICK",
                    "FP Rank": "",
                    "Base Value": round(d["base"]),
                    "Need/Pick Mult": round(d["need_mult"], 2),
                    "Adjusted Value": round(d["adj"]),
                }
            )
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    return df.sort_values("Adjusted Value", ascending=False)

# ---------- UI ----------

players_fp = build_player_pool()
if players_fp.empty:
    st.error("Couldn't load FantasyPros rankings. Try again later.")
    st.stop()

team_df, team_strength, pick_owner, league = build_team_frames(players_fp)

st.markdown(
    "### Dynasty Bros. Trade Calculator\n"
    "*Powered by FantasyPros' latest **Dynasty Superflex PPR** rankings and live Sleeper league data.*"
)
st.caption(
    "Player values are built off FantasyPros dynasty superflex ranks, then adjusted by position, last-season PPR scoring curves, team needs, "
    "and projected draft pick strength. Treat this as a guide — not a guarantee."
)

with st.sidebar:
    st.subheader("How should we view this trade?")
    rank_slider = st.slider(
        "Weight on FantasyPros rank",
        0.5,
        1.5,
        1.0,
        0.05,
        help="Higher = rankings matter more. Lower = things like team needs and picks matter a bit more."
    )
    need_slider = st.slider(
        "Weight on roster needs",
        0.0,
        1.0,
        0.30,
        0.05,
        help="How much to nudge values based on how many QBs/RBs/WRs/TEs each team already has. This always stays a smaller factor than rank."
    )
    pick_slider = st.slider(
        "Weight on future picks",
        0.5,
        1.5,
        1.0,
        0.05,
        help="Higher = future picks get a bit more credit vs current players. Lower = more win-now focused."
    )

tabs = st.tabs(["Trade Calculator", "Trade Finder", "League / Data Overview"])
tab_calc, tab_find, tab_overview = tabs

team_list = (
    team_strength.sort_values("TeamName")["TeamName"].tolist()
    if not team_strength.empty
    else sorted(team_df["TeamName"].unique().tolist())
)

name_to_roster = {}
for _, row in team_strength.iterrows():
    name_to_roster[row["TeamName"]] = row["RosterID"]

# ---------- Trade Calculator ----------

with tab_calc:
    st.subheader("Trade Calculator")

    if team_df.empty or team_strength.empty:
        st.warning("Couldn't load Sleeper league data. Team-specific features will be limited.")
    else:
        colA, colB = st.columns(2)
        with colA:
            teamA = st.selectbox("Team A", team_list, index=0)
        with colB:
            teamB = st.selectbox("Team B", team_list, index=min(1, len(team_list) - 1))

        ridA = name_to_roster.get(teamA)
        ridB = name_to_roster.get(teamB)

        rosterA = team_df[team_df["RosterID"] == ridA].sort_values("SleeperName")
        rosterB = team_df[team_df["RosterID"] == ridB].sort_values("SleeperName")

        A_players = st.multiselect(
            f"{teamA} sends players",
            options=rosterA["SleeperName"].tolist(),
            default=[],
        )
        A_picks_list = team_pick_list(ridA, pick_owner, team_strength)
        A_pick_labels = [p[3] for p in A_picks_list]
        A_picks = st.multiselect(
            f"{teamA} sends picks",
            options=A_pick_labels,
            default=[],
        )

        B_players = st.multiselect(
            f"{teamB} sends players",
            options=rosterB["SleeperName"].tolist(),
            default=[],
        )
        B_picks_list = team_pick_list(ridB, pick_owner, team_strength)
        B_pick_labels = [p[3] for p in B_picks_list]
        B_picks = st.multiselect(
            f"{teamB} sends picks",
            options=B_pick_labels,
            default=[],
        )

        def ids_from_names(roster_df, names):
            sub = roster_df[roster_df["SleeperName"].isin(names)]
            return sub["sleeper_id"].tolist()

        A_player_ids = ids_from_names(rosterA, A_players)
        B_player_ids = ids_from_names(rosterB, B_players)

        A_out_val, A_out_det = side_package_value(
            A_player_ids,
            A_picks,
            ridA,
            team_df,
            pick_owner,
            team_strength,
            need_weight=need_slider,
            picks_weight=pick_slider,
        )
        A_in_val, A_in_det = side_package_value(
            B_player_ids,
            B_picks,
            ridA,
            team_df,
            pick_owner,
            team_strength,
            need_weight=need_slider,
            picks_weight=pick_slider,
        )
        B_out_val, B_out_det = side_package_value(
            B_player_ids,
            B_picks,
            ridB,
            team_df,
            pick_owner,
            team_strength,
            need_weight=need_slider,
            picks_weight=pick_slider,
        )
        B_in_val, B_in_det = side_package_value(
            A_player_ids,
            A_picks,
            ridB,
            team_df,
            pick_owner,
            team_strength,
            need_weight=need_slider,
            picks_weight=pick_slider,
        )

        # rank emphasis
        A_out_val *= rank_slider
        A_in_val *= rank_slider
        B_out_val *= rank_slider
        B_in_val *= rank_slider

        col1, col2, col3, col4 = st.columns(4)
        col1.metric(f"{teamA} sends", f"{A_out_val:,.0f}")
        col2.metric(f"{teamA} receives", f"{A_in_val:,.0f}")
        col3.metric(f"{teamB} sends", f"{B_out_val:,.0f}")
        col4.metric(f"{teamB} receives", f"{B_in_val:,.0f}")

        st.markdown("---")
        st.subheader("Fairness verdict")

        verdict, diff, pct, expl = format_value_gap(A_in_val, B_in_val, teamA, teamB)

        if abs(diff) < max(A_in_val, B_in_val, 1.0) * 0.05:
            st.info(f"Trade looks roughly balanced. {expl}")
        elif diff > 0:
            st.info(
                f"**Trade leans toward {teamA}.**\n\n"
                f"{teamA} is receiving about **{abs(diff):,.0f} more value points**, "
                f"which is roughly **{pct*100:.1f}%** of the larger side.\n\n{expl}"
            )
        else:
            st.info(
                f"**Trade leans toward {teamB}.**\n\n"
                f"{teamB} is receiving about **{abs(diff):,.0f} more value points**, "
                f"which is roughly **{pct*100:.1f}%** of the larger side.\n\n{expl}"
            )

        st.markdown("#### Roster context (why a manager might like or hesitate)")
        ctxA = summarise_roster_context(team_df, ridA, A_in_det, A_out_det)
        ctxB = summarise_roster_context(team_df, ridB, B_in_det, B_out_det)
        st.markdown(f"**{teamA} perspective:** {ctxA}")
        st.markdown(f"**{teamB} perspective:** {ctxB}")

        with st.expander("See the pieces, ranks, and values we used"):
            st.markdown(f"**What {teamA} receives**")
            dfA = build_piece_table(A_in_det)
            if not dfA.empty:
                st.table(dfA)
            else:
                st.write("No players or picks selected for this side.")
            st.markdown(f"**What {teamB} receives**")
            dfB = build_piece_table(B_in_det)
            if not dfB.empty:
                st.table(dfB)
            else:
                st.write("No players or picks selected for this side.")

# ---------- Trade Finder ----------

with tab_find:
    st.subheader("Trade Finder (beta)")

    if team_df.empty or team_strength.empty:
        st.warning("Need live Sleeper data to suggest trades. Try again later.")
    else:
        col1, col2 = st.columns(2)
        with col1:
            wanting_team = st.selectbox("Team that wants to acquire a player", team_list, key="tf_team")
        with col2:
            target_player_name = st.selectbox(
                "Player they want",
                sorted(team_df["SleeperName"].unique().tolist()),
                key="tf_target",
            )

        wanting_rid = name_to_roster.get(wanting_team)
        target_row = team_df[team_df["SleeperName"] == target_player_name]
        if target_row.empty:
            st.info("Pick a player to get started.")
        else:
            target_row = target_row.iloc[0]
            target_owner_id = target_row["RosterID"]
            target_team_name = team_strength.loc[team_strength["RosterID"] == target_owner_id, "TeamName"].iloc[0]

            st.markdown(
                f"{wanting_team} is trying to acquire **{target_player_name} ({target_row['Pos']})**, "
                f"currently on **{target_team_name}**."
            )

            offer_roster = team_df[team_df["RosterID"] == wanting_rid].sort_values("SleeperName")
            offer_picks = team_pick_list(wanting_rid, pick_owner, team_strength)
            offer_pick_labels = [p[3] for p in offer_picks]

            # target value
            target_val, _ = side_package_value(
                [target_row["sleeper_id"]],
                [],
                target_owner_id,
                team_df,
                pick_owner,
                team_strength,
                need_weight=need_slider,
                picks_weight=pick_slider,
            )
            target_val *= rank_slider

            # candidate players from offering team
            pvals = []
            for _, row in offer_roster.iterrows():
                pid = row["sleeper_id"]
                if pid == target_row["sleeper_id"]:
                    continue
                val, _ = side_package_value(
                    [pid],
                    [],
                    wanting_rid,
                    team_df,
                    pick_owner,
                    team_strength,
                    need_weight=need_slider,
                    picks_weight=pick_slider,
                )
                val *= rank_slider
                pvals.append((pid, row["SleeperName"], row["Pos"], val))
            pvals.sort(key=lambda x: x[3], reverse=True)

            suggestions = []

            # 1-for-1 offers near target value
            for pid, name, pos, val in pvals[:12]:
                gap = abs(val - target_val)
                if gap <= target_val * 0.20:
                    suggestions.append(
                        {
                            "players_out": [name],
                            "picks_out": [],
                            "value_out": val,
                            "desc": f"{wanting_team} sends **{name} ({pos})**.",
                            "gap": gap,
                        }
                    )

            # player + pick combos
            for pid, name, pos, val in pvals[:8]:
                for label in offer_pick_labels[:6]:
                    val2, _ = side_package_value(
                        [pid],
                        [label],
                        wanting_rid,
                        team_df,
                        pick_owner,
                        team_strength,
                        need_weight=need_slider,
                        picks_weight=pick_slider,
                    )
                    val2 *= rank_slider
                    gap = abs(val2 - target_val)
                    if gap <= target_val * 0.20:
                        suggestions.append(
                            {
                                "players_out": [name],
                                "picks_out": [label],
                                "value_out": val2,
                                "desc": f"{wanting_team} sends **{name} ({pos})** and **{label}**.",
                                "gap": gap,
                            }
                        )

            suggestions.sort(key=lambda x: x["gap"])
            if not suggestions:
                # if nothing is close, show a few best approximate options
                for pid, name, pos, val in pvals[:3]:
                    suggestions.append(
                        {
                            "players_out": [name],
                            "picks_out": [],
                            "value_out": val,
                            "desc": f"{wanting_team} sends **{name} ({pos})**.",
                            "gap": abs(val - target_val),
                        }
                    )
                suggestions.sort(key=lambda x: x["gap"])

            max_show = min(5, len(suggestions))
            st.markdown("##### Suggested offers")
            st.caption(
                "These are based on our value model. Real managers might ask for a bit more or less depending on their timelines and preferences."
            )

            for i in range(max_show):
                s = suggestions[i]
                st.markdown(
                    f"**Option {i+1}:** {s['desc']}  \n"
                    f"Approximate package value: **{s['value_out']:,.0f}** vs. target's **{target_val:,.0f}**."
                )

# ---------- Overview tab ----------

with tab_overview:
    st.subheader("League / Data Overview")
    if not team_strength.empty:
        st.markdown("**Team power rankings (by our value model)**")
        show_cols = ["LeagueRank", "TeamName", "TotalValue", "Wins", "Losses", "PointsFor"]
        st.dataframe(team_strength[show_cols].sort_values("LeagueRank"), use_container_width=True)

    st.markdown("**Top 50 players by our value model**")
    top_players = players_fp.sort_values("DynastyValue", ascending=False).head(50)
    st.dataframe(
        top_players[["Player", "Pos", "Rank", "Age", "DynastyValue"]],
        use_container_width=True,
    )

    st.markdown(
        "---\n"
        "_Notes:_ Values are derived from FantasyPros dynasty superflex PPR ranks, smoothed using last-season PPR scoring by position, "
        "and then lightly adjusted for roster context and projected draft pick strength. "
        "They're meant to give you a structured starting point for trade talks, not a final answer."
    )
