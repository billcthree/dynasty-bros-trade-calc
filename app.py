import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
from itertools import combinations
import random
import requests

# -------------------- Page config --------------------
st.set_page_config(
    page_title="Dynasty Bros. Trade Calculator (FantasyPros + Sleeper)",
    layout="wide"
)
st.title("Dynasty Bros. Trade Calculator (using FantasyPros Rankings + AI Generated Team Needs)")
st.caption(
    "FantasyPros dynasty ranks + live Sleeper rosters. "
    "Rank-first values with light team-need seasoning. Picks scaled by roster strength, record, and original team."
)

# ====================================================
# Helpers: name normalization, Sleeper fetch w/ picks
# ====================================================

def normalize_name(name: str) -> str:
    """
    Smooth over small name differences:
    - lower case
    - remove punctuation
    - drop 'jr', 'sr', 'ii', 'iii', 'iv', 'v'
    - drop 1-letter middle initials
    """
    if not isinstance(name, str):
        return ""
    name = name.lower().replace(".", " ").replace(",", " ").strip()
    tokens = [
        t for t in name.split()
        if t not in {"jr", "sr", "ii", "iii", "iv", "v"} and len(t) > 1
    ]
    return " ".join(tokens)


@st.cache_data(show_spinner=False)
def load_sleeper_league(league_id: str):
    """
    Fetch Sleeper league info + users + rosters + records + NFL player DB + traded picks.

    Returns:
      rosters_df: Team, Player, Pos
      records_df: Team, Wins, Losses, Ties
      picks_by_team: { current_owner_team: [pick_label, ...] }
      pick_label_to_original_team: { pick_label: original_team_name }
      future_years: list[int] of seasons we built picks for
    """
    base = f"https://api.sleeper.app/v1/league/{league_id}"

    league_info = requests.get(base, timeout=20).json()
    season = int(league_info.get("season", datetime.now().year))
    draft_rounds = int(league_info.get("draft_rounds", 4))

    # We only care about picks for drafts that haven't happened yet:
    # next 3 drafts after current season.
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
    # Base assumption: each roster keeps its own picks (future_years x 1..draft_rounds),
    # then we apply /traded_picks to move them.

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

    # Apply traded picks (Sleeper gives final state)
    for tp in traded or []:
        try:
            yr = int(tp.get("season", 0))
            rnd = int(tp.get("round", 0))
            orig_rid = tp.get("roster_id")
            new_owner_rid = tp.get("owner_id")  # roster_id that now owns the pick
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

    # Build per-team labels + original-team map
    picks_by_team = {}  # current_owner -> [labels]
    pick_label_to_original_team = {}  # label -> original_team_name

    for (yr, rnd, orig_rid), current_owner in picks_current_owner.items():
        original_team = rosterid_to_team.get(orig_rid)
        if not original_team or not current_owner:
            continue
        label = f"{yr} R{rnd} ({original_team})"
        picks_by_team.setdefault(current_owner, []).append(label)
        pick_label_to_original_team[label] = original_team

    # Sort labels per team for nicer UI
    for tm in picks_by_team:
        picks_by_team[tm] = sorted(picks_by_team[tm])

    return rosters_df, records_df, picks_by_team, pick_label_to_original_team, future_years


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
    help="Paste your Sleeper league ID. Example: 1194681871141023744"
)

st.sidebar.caption("You can still upload CSVs below to override or test things manually.")
up_players = st.sidebar.file_uploader("Player Ranks CSV (Player, Pos, Rank)", type=["csv"])
up_rosters = st.sidebar.file_uploader("Rosters CSV (Team, Player, Pos optional)", type=["csv"])

# -------- Modifiers ----------
st.sidebar.header("2) Modifiers (quick dials)")

ELITE_GAP = st.sidebar.slider(
    "Elite gap (value for Rank #1)",
    800.0, 2200.0, 1500.0, 50.0,
    help="Higher = bigger gap between elite and everyone else."
)

RANK_IMPORTANCE = st.sidebar.slider(
    "Rank importance (steepness)",
    0.015, 0.060, 0.038, 0.001,
    help="Higher = rank differences matter more (e.g., #26 >> #137)."
)

NEED_WEIGHT = st.sidebar.slider(
    "Roster needs weight",
    0.0, 0.6, 0.20, 0.05,
    help="How much roster needs nudge values. Keep low so rank is still the main driver."
)

PACKAGE_PENALTY = st.sidebar.slider(
    "2-for-1 tax (package penalty)",
    0.0, 1.0, 0.75, 0.05,
    help="Higher = quantity counts less vs quality. Prevents 3 mids beating 1 stud."
)

N_STRENGTH = st.sidebar.slider(
    "Team strength depth (top N players)",
    6, 20, 10, 1,
    help="How many top assets define team strength for pick values."
)

PICK_MAX = st.sidebar.slider(
    "Pick max (value for earliest 1st)",
    250.0, 900.0, 500.0, 25.0,
    help="Max value for the very earliest first-round pick."
)

PICK_SLOT_DECAY = st.sidebar.slider(
    "In-round drop-off",
    0.08, 0.35, 0.20, 0.01,
    help="How fast picks get worse later in the round."
)

R2_SCALE = st.sidebar.slider("Round 2 scale vs Round 1", 0.20, 0.60, 0.40, 0.01)
R3_SCALE = st.sidebar.slider("Round 3 scale vs Round 1", 0.08, 0.35, 0.20, 0.01)
R4_SCALE = st.sidebar.slider("Round 4 scale vs Round 1", 0.03, 0.25, 0.10, 0.01)

YEAR2_DISC = st.sidebar.slider("Next-year pick discount", 0.70, 0.95, 0.85, 0.01)
YEAR3_DISC = st.sidebar.slider("Two-years-out discount", 0.50, 0.90, 0.70, 0.01)

st.sidebar.header("3) Update")
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
            rosters_live, records_df, picks_by_team, pick_label_to_original, future_pick_years = load_sleeper_league(
                league_id.strip()
            )

            # Rankings from bundled CSV (FantasyPros export you maintain)
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

    # Generic picks (no traded info) – future 3 drafts
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

# ====================================================
# Core valuation helpers (with positional scarcity)
# ====================================================

players_df = players_df.copy()
players_df["Rank"] = pd.to_numeric(players_df["Rank"], errors="coerce")
players_df = players_df.dropna(subset=["Rank"]).reset_index(drop=True)

# --- positional scarcity: adjust multipliers based on how many good players exist at each position ---
TOP_CUTOFF = 100  # look at top-100 ranked players for scarcity
top_slice = players_df[players_df["Rank"] <= TOP_CUTOFF]
counts = top_slice["Pos"].value_counts()
avg_count = counts.mean() if len(counts) else 1.0

scarcity_mult = {}
for pos in ["QB", "RB", "WR", "TE"]:
    c = counts.get(pos, 1)  # number of top players at this pos
    raw = avg_count / c  # >1 if scarce, <1 if abundant
    raw_clamped = max(0.5, min(2.0, raw))
    # Map raw_clamped in [0.5, 2.0] into [0.85, 1.15]
    scarcity = 0.85 + (raw_clamped - 0.5) * (1.15 - 0.85) / (2.0 - 0.5)
    scarcity_mult[pos] = scarcity

posmult_effective = {}
for pos, base_mult in DEFAULT_TARGETS.keys() if False else DEFAULT_POSMULT.items():
    # base position multiplier * scarcity tweak
    posmult_effective[pos] = base_mult * scarcity_mult.get(pos, 1.0)

players_df["BaseValue"] = (
    ELITE_GAP * np.exp(-RANK_IMPORTANCE * (players_df["Rank"] - 1))
).round(2)
players_df["PosMult"] = players_df["Pos"].map(posmult_effective).fillna(1.0)

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
    base_mult = need_multiplier(team_counts.get(pos, 0), targets.get(pos, 0))
    # Blend toward 1 based on NEED_WEIGHT so needs are a light factor
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

# Build strengths & record-based ranking for picks (by ORIGINAL team)
team_list = sorted(rosters_df["Team"].unique().tolist())
strengths = {t: team_strength(t) for t in team_list}

records_df = records_df.set_index("Team")
def get_record(team: str):
    if team in records_df.index:
        row = records_df.loc[team]
        return float(row.get("Wins", 0)), float(row.get("Losses", 0))
    return 0.0, 0.0

# Worst teams (few wins, weaker rosters) get earliest (most valuable) picks
sorted_for_picks = sorted(
    team_list,
    key=lambda t: (get_record(t)[0], strengths[t])  # wins ascending, strength ascending
)
team_pick_slot = {t: i + 1 for i, t in enumerate(sorted_for_picks)}

def pick_factor_for_round(rnd: int) -> float:
    return {1: 1.0, 2: R2_SCALE, 3: R3_SCALE, 4: R4_SCALE}.get(rnd, 0.0)

def pick_base_value(slot: int, rnd: int) -> float:
    round_mult = pick_factor_for_round(rnd)
    maxslot = max(1, len(team_list))
    slot = min(max(1, int(slot)), maxslot)
    slot_val = np.exp(-PICK_SLOT_DECAY * (slot - 1))  # earlier slot => larger value
    return PICK_MAX * round_mult * slot_val

def pick_value(original_team: str, year: int, rnd: int) -> float:
    current_year = datetime.now().year
    diff = year - current_year
    if diff <= 0:
        year_mult = 1.0
    elif diff == 1:
        year_mult = YEAR2_DISC
    else:
        year_mult = YEAR3_DISC ** diff

    slot = team_pick_slot.get(original_team, len(team_list))
    base = pick_base_value(slot, rnd)
    return round(base * year_mult, 1)

def build_pick_labels_for_team(team: str):
    # In live mode, this uses picks_by_team (includes trades, only future drafts).
    # In CSV mode, picks_by_team was built generically above.
    return picks_by_team.get(team, [])

def parse_pick_label(label: str):
    try:
        parts = label.split()
        yr = int(parts[0])
        rnd = int(parts[1].replace("R", ""))
        # team in parentheses is ORIGINAL team (whose record/strength matters)
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
        base = float(r["BaseValue"].iloc[0]) * float(r["PosMult"].iloc[0])
        mult = apply_need(pos, counts)
        val = base * mult
        vals.append(val)
        details.append((n, val, mult, pos))
    return package_sum(vals), details

def sum_picks_value(pick_labels):
    vals = []
    details = []
    for lbl in pick_labels:
        v = label_value(lbl)
        vals.append(v)
        details.append((lbl, v))
    return package_sum(vals), details


# ====================================================
# UI layout: tabs
# ====================================================

tab_trade, tab_finder = st.tabs(["💼 Trade Calculator", "🔍 Trade Finder"])

# ----------------------------------------------------
# TAB 1: TRADE CALCULATOR
# ----------------------------------------------------
with tab_trade:
    st.subheader("Build a Trade (two boxes)")

    colA, colB = st.columns(2)
    with colA:
        teamA = st.selectbox("Team A (left side)", team_list, index=0 if team_list else None)
    with colB:
        teamB_choices = [t for t in team_list if t != (teamA if team_list else None)]
        teamB = st.selectbox("Team B (right side)", teamB_choices, index=0 if teamB_choices else None)

    a_players_list = roster_players(teamA) if teamA else []
    b_players_list = roster_players(teamB) if teamB else []

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
            build_pick_labels_for_team(teamA) if teamA else [],
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
            build_pick_labels_for_team(teamB) if teamB else [],
            key="b_send_picks",
        )

    if not manual or recalc:
        # Value each side RECEIVES
        A_get_players, A_get_p_det = sum_players_value(b_send_players, teamA)
        A_get_picks,   A_get_pk_det = sum_picks_value(b_send_picks)
        A_get_total = A_get_players + A_get_picks

        B_get_players, B_get_p_det = sum_players_value(a_send_players, teamB)
        B_get_picks,   B_get_pk_det = sum_picks_value(a_send_picks)
        B_get_total = B_get_players + B_get_picks

        m1, m2 = st.columns(2)
        m1.metric(f"{teamA} receives (rank-first value)", f"{A_get_total:,.0f}")
        m2.metric(f"{teamB} receives (rank-first value)", f"{B_get_total:,.0f}")

        # Roster-context needs (focus on real needs, not surplus spam)
        def needs_summary(team, incoming_details):
            before = team_pos_counts(team)
            inc_counts = {"QB": 0, "RB": 0, "WR": 0, "TE": 0}
            for _, _, _, pos in incoming_details:
                inc_counts[pos] = inc_counts.get(pos, 0) + 1

            messages = []
            for p in ["QB", "RB", "WR", "TE"]:
                tgt = int(targets.get(p, 0))
                current = before.get(p, 0)
                incoming = inc_counts.get(p, 0)
                if incoming > 0 and current < tgt:
                    if current == 0 and p == "QB":
                        messages.append("really needs a starting QB, so QB help carries extra weight here.")
                    else:
                        messages.append(f"could use more {p}s, so adding {p} here fits a roster need.")
            return messages

        A_needs = needs_summary(teamA, A_get_p_det)
        B_needs = needs_summary(teamB, B_get_p_det)

        st.markdown("---")
        st.subheader("Fairness Verdict")

        diff = A_get_total - B_get_total
        larger = max(A_get_total, B_get_total, 1.0)
        pct = diff / larger

        def grade(pct_diff: float) -> str:
            ad = abs(pct_diff)
            if ad < 0.05:
                return "Fair for both sides."
            elif ad < 0.12:
                return "Slight edge to the winner."
            else:
                return "Clear win for the winner."

        THRESH = 0.03
        if abs(pct) < THRESH:
            msg = (
                f"This looks like a **fair trade** – "
                f"{teamA} and {teamB} receive very similar value by rankings."
            )
            st.success(msg)
        elif pct > 0:
            msg = f"**Trade favors {teamA}** by {diff:,.0f} (~{pct:.1%}). {grade(pct)}"
            if A_needs:
                msg += " " + f"{teamA} " + " ".join(A_needs)
            if B_needs:
                msg += " " + f"{teamB} " + " ".join(B_needs)
            st.info(msg)
        else:
            msg = f"**Trade favors {teamB}** by {abs(diff):,.0f} (~{abs(pct):.1%}). {grade(pct)}"
            if B_needs:
                msg += " " + f"{teamB} " + " ".join(B_needs)
            if A_needs:
                msg += " " + f"{teamA} " + " ".join(A_needs)
            st.info(msg)

        with st.expander("Details (who adds what value)"):
            st.write(f"**What {teamA} receives**")
            if len(A_get_p_det):
                st.table(pd.DataFrame(A_get_p_det, columns=["Player", "Value", "Need Mult", "Pos"]))
            if len(A_get_pk_det):
                st.table(pd.DataFrame(A_get_pk_det, columns=["Pick", "Value"]))

            st.write(f"**What {teamB} receives**")
            if len(B_get_p_det):
                st.table(pd.DataFrame(B_get_p_det, columns=["Player", "Value", "Need Mult", "Pos"]))
            if len(B_get_pk_det):
                st.table(pd.DataFrame(B_get_pk_det, columns=["Pick", "Value"]))


# ----------------------------------------------------
# TAB 2: TRADE FINDER
# ----------------------------------------------------
with tab_finder:
    st.subheader("Trade Finder — suggest packages for a target")

    tf_from = st.selectbox("Your team (offering)", team_list, key="tf_from")
    tf_to_choices = [t for t in team_list if t != tf_from]
    tf_to = st.selectbox("Trade partner", tf_to_choices, key="tf_to")

    target_list = roster_players(tf_to)
    tf_target = st.selectbox(
        f"Target player on {tf_to}",
        target_list,
        format_func=lambda x: label_map.get(x, x),
    )

    def player_value_for_team(player_name, team_for_need):
        r = players_df.loc[players_df["Player"] == player_name]
        if r.empty:
            return 0.0, 1.0
        pos = r["Pos"].iloc[0]
        base = float(r["BaseValue"].iloc[0]) * float(r["PosMult"].iloc[0])
        mult = apply_need(pos, team_pos_counts(team_for_need))
        return base * mult, mult

    target_val, target_mult = player_value_for_team(tf_target, tf_to)
    st.caption(
        f"Estimated value of {label_map.get(tf_target, tf_target)} **to {tf_to}**: "
        f"~{target_val:,.0f} (need multiplier: {target_mult:.2f}). "
        "Aim for ~90–110% of this in offers."
    )

    your_names = roster_players(tf_from)
    cand = players_df[players_df["Player"].isin(your_names)].copy()

    def val_to_partner(name):
        v, _ = player_value_for_team(name, tf_to)
        return v

    from_counts = team_pos_counts(tf_from)
    def surplus_score(pos):
        return max(0, from_counts.get(pos, 0) - targets.get(pos, 0))

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

    st.markdown("#### Suggested Packages")
    if not uniq:
        st.write(
            "No clean suggestions found in the 90–110% range. "
            "Try lowering the 2-for-1 tax or rank importance, or click **Suggest another trade**."
        )
    else:
        for i, (assets, v) in enumerate(uniq[:5], start=1):
            pct = v / target_val if target_val > 0 else 0
            st.write(f"**Suggestion {i}** — value to {tf_to}: {v:,.0f} (~{pct:.0%} of target)")
            st.write("- " + "\n- ".join(assets))
