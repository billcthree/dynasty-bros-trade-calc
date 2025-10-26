
import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
from itertools import combinations
import random

st.set_page_config(page_title="Dynasty Bros. Trade Calculator (using FantasyPros Rankings + AI Generated Team Needs)", layout="wide")
st.title("Dynasty Bros. Trade Calculator (using FantasyPros Rankings + AI Generated Team Needs)")
st.caption("Rank-first values with light team-need seasoning. Two-box entry. Picks scale by team strength. Trade Finder suggests multiple packages.")

@st.cache_data
def load_csv(path):
    return pd.read_csv(path)

def try_load(folder, name):
    try:
        return load_csv(f"{folder}/{name}")
    except Exception:
        return None

players = try_load("data","player_ranks.csv")
rosters = try_load("data","rosters.csv")
settings = try_load("data","settings.csv")

st.sidebar.header("1) Data (upload to override)")
up_players = st.sidebar.file_uploader("Player Ranks CSV (Player, Pos, Rank)", type=["csv"])
up_rosters = st.sidebar.file_uploader("Rosters CSV (Team, Player)", type=["csv"])
up_settings = st.sidebar.file_uploader("Settings CSV (Targets + PosMultipliers)", type=["csv"])
if up_players is not None: players = pd.read_csv(up_players)
if up_rosters is not None: rosters = pd.read_csv(up_rosters)
if up_settings is not None: settings = pd.read_csv(up_settings)

st.sidebar.header("1a) (Optional) Load rosters from Sleeper")
sleeper_league_id = st.sidebar.text_input("Sleeper League ID (paste to import rosters)", value="", help="Example: 1194681871141023744")
if sleeper_league_id and st.sidebar.button("Fetch rosters/users from Sleeper"):
    try:
        import requests
        users = requests.get(f"https://api.sleeper.app/v1/league/{sleeper_league_id}/users", timeout=15).json()
        rosts = requests.get(f"https://api.sleeper.app/v1/league/{sleeper_league_id}/rosters", timeout=15).json()
        id_to_name = {u.get("user_id"): (u.get("metadata",{}).get("team_name") or u.get("display_name") or f"Team {i+1}") for i,u in enumerate(users)}
        rows = []
        for r in rosts:
            team_label = id_to_name.get(r.get("owner_id"), f"Team {r.get('roster_id','?')}")
            for pid in (r.get("players") or []):
                rows.append({"Team": team_label, "Player": str(pid)})
        if rows:
            rosters = pd.DataFrame(rows)
            st.success(f"Loaded {len(rosters)} roster rows from Sleeper. (Note: player IDs are not mapped to names unless your player_ranks.csv uses Sleeper IDs.)")
        else:
            st.warning("Fetched data, but no player rows present. You may need to map Sleeper player IDs to names to merge with your ranks file.")
    except Exception as e:
        st.error(f"Sleeper fetch failed: {e}")

required_player_cols = {"Player","Pos","Rank"}
if players is None or not required_player_cols.issubset(set(players.columns)):
    st.error("Please provide player_ranks.csv with columns: Player, Pos, Rank")
    st.stop()
if rosters is None or not {"Team","Player"}.issubset(set(rosters.columns)):
    st.error("Please provide rosters.csv with columns: Team, Player")
    st.stop()

players["Pos"] = players["Pos"].astype(str).str.upper().str.strip()
players["Rank"] = pd.to_numeric(players["Rank"], errors="coerce")
players = players.dropna(subset=["Rank"]).reset_index(drop=True)
rosters["Team"] = rosters["Team"].astype(str).str.strip()
rosters["Player"] = rosters["Player"].astype(str).str.strip()

def s_lookup(key, default):
    if settings is None or "Key" not in settings.columns: return default
    row = settings.loc[settings["Key"]==key]
    if row.empty: return default
    try:
        return float(row["Value"].iloc[0])
    except Exception:
        return default

targets = {"QB": int(s_lookup("Target_QB",3)), "RB": int(s_lookup("Target_RB",5)),
           "WR": int(s_lookup("Target_WR",7)), "TE": int(s_lookup("Target_TE",2))}
posmult = {"QB": s_lookup("PosMult_QB",1.12), "RB": s_lookup("PosMult_RB",1.00),
           "WR": s_lookup("PosMult_WR",1.00), "TE": s_lookup("PosMult_TE",0.95)}

st.sidebar.header("2) Modifiers (quick dials)")
BASE = st.sidebar.slider("Elite gap (ceiling for Rank #1)", 800.0, 2200.0, 1500.0, 50.0,
                         help="Bigger = larger gap between elite and everyone else.")
K = st.sidebar.slider("Rank importance (steepness)", 0.015, 0.060, 0.035, 0.001,
                      help="Bigger = rank differences matter more (e.g., #26 >> #137).")
NEED_WEIGHT = st.sidebar.slider("Roster-needs weight", 0.0, 1.0, 0.30, 0.05,
                                help="How much roster needs nudge values (keep modest so rank stays king).")
PACKAGE_GAMMA = st.sidebar.slider("2-for-1 tax (package penalty)", 0.0, 1.0, 0.70, 0.05,
                                  help="Bigger = quantity counts less vs quality. Prevents 3 mids beating 1 stud.")
N_STRENGTH = st.sidebar.slider("Pick strength depth (Top-N assets)", 6, 20, 10, 1,
                               help="How many top assets define a team’s strength for pick values.")
PICK_CEIL = st.sidebar.slider("Pick max (R1.01 ceiling)", 400.0, 1500.0, 750.0, 25.0,
                              help="Max value for the very best first-round pick.")
SLOT_K = st.sidebar.slider("In-round drop-off", 0.05, 0.30, 0.15, 0.01,
                           help="How fast picks get worse later in the round.")
R2_mult = st.sidebar.slider("Round 2 scale", 0.25, 0.80, 0.50, 0.01, help="Round 2 base vs Round 1.")
R3_mult = st.sidebar.slider("Round 3 scale", 0.10, 0.50, 0.25, 0.01, help="Round 3 base vs Round 1.")
R4_mult = st.sidebar.slider("Round 4 scale", 0.05, 0.30, 0.12, 0.01, help="Round 4 base vs Round 1.")
YEAR_DECAY_2 = st.sidebar.slider("Next-year discount", 0.80, 1.00, 0.92, 0.01,
                                 help="Discount for next-year picks (contenders value later picks less).")
YEAR_DECAY_3 = st.sidebar.slider("Two-year discount", 0.70, 1.00, 0.92, 0.01,
                                 help="Discount for picks two years out.")

st.sidebar.header("3) Update")
manual = st.sidebar.checkbox("Manual mode (use button to recalc)", value=False)
recalc = True
if manual:
    recalc = st.sidebar.button("Recalculate now")

players = players.copy()
players["BaseValue"] = (BASE * np.exp(-K * (players["Rank"] - 1))).round(2)
players["PosMult"] = players["Pos"].map(posmult).fillna(1.0)

def team_pos_counts(team):
    names = set(rosters.loc[rosters["Team"]==team, "Player"].tolist())
    sub = players[players["Player"].isin(names)]
    return {p:int((sub["Pos"]==p).sum()) for p in ["QB","RB","WR","TE"]}

def need_multiplier(count, target):
    diff = count - target
    if diff <= -2: return 1.12
    if diff == -1: return 1.06
    if diff == 0:  return 1.00
    if diff == 1:  return 0.95
    return 0.88

def apply_need(pos, team_counts):
    base_mult = need_multiplier(team_counts.get(pos,0), targets.get(pos,0))
    return 1.0 + NEED_WEIGHT * (base_mult - 1.0)

def roster_players(team):
    names = rosters.loc[rosters["Team"]==team, "Player"].tolist()
    names = [n for n in names if n in set(players["Player"])]
    names.sort(key=lambda x: x.lower())
    return names

label_map = {row["Player"]: f'{row["Player"]} ({row["Pos"]})' for _, row in players.iterrows()}

def team_strength(team):
    names = rosters.loc[rosters["Team"]==team, "Player"].tolist()
    sub = players[players["Player"].isin(names)].copy()
    vals = (sub["BaseValue"] * sub["PosMult"]).sort_values(ascending=False).head(N_STRENGTH)
    return float(vals.sum()) if len(vals) else 0.0

teams = sorted(rosters["Team"].unique().tolist())
strengths = {t: team_strength(t) for t in teams}
sorted_teams = sorted(teams, key=lambda t: strengths[t], reverse=True)
team_expected_slot = {t: i+1 for i,t in enumerate(sorted_teams)}  # 1=best team (late)

def pick_base_value(slot, rnd):
    round_mult = {1:1.00, 2:R2_mult, 3:R3_mult, 4:R4_mult}[rnd]
    maxslot = max(1, len(teams))
    slot = min(max(1, int(slot)), maxslot)
    slot_val = np.exp(-SLOT_K * (slot - 1))
    return PICK_CEIL * round_mult * slot_val

def pick_value(team, year, rnd):
    years = [datetime.now().year + i for i in [1,2,3]]
    decay = {years[0]:1.00, years[1]:YEAR_DECAY_2, years[2]:YEAR_DECAY_3}
    base = pick_base_value(team_expected_slot.get(team, len(teams)), rnd)
    return round(base * decay.get(year, YEAR_DECAY_2**(year-years[0])), 1)

def build_pick_labels_for_team(team):
    labels = []
    years = [datetime.now().year + i for i in [1,2,3]]
    for yr in years:
        for r in [1,2,3,4]:
            labels.append(f"{yr} R{r} ({team})")
    return labels

def parse_pick_label(label):
    try:
        parts = label.split()
        yr = int(parts[0]); rnd = int(parts[1].replace("R",""))
        team = label[label.find("(")+1:label.find(")")]
        return yr, rnd, team
    except Exception:
        return None, None, None

def label_value(lbl):
    yr, rnd, tm = parse_pick_label(lbl)
    if yr is None: return 0.0
    return pick_value(tm, yr, rnd)

def package_sum(values):
    if not values: return 0.0
    values = sorted(values, reverse=True)
    total = 0.0
    for i, v in enumerate(values, start=1):
        weight = (i ** (-PACKAGE_GAMMA)) if PACKAGE_GAMMA > 0 else 1.0
        total += v * weight
    return total

def player_value_for_team(player_name, team_for_need):
    r = players.loc[players["Player"]==player_name]
    if r.empty: return 0.0, 1.0
    pos = r["Pos"].iloc[0]
    base = float(r["BaseValue"].iloc[0]) * float(r["PosMult"].iloc[0])
    counts = team_pos_counts(team_for_need)
    need_mult = apply_need(pos, counts)
    return base * need_mult, need_mult

def sum_players_value(player_list, team_for_need):
    vals = []; details = []
    counts = team_pos_counts(team_for_need)
    for n in player_list:
        r = players.loc[players["Player"]==n]
        if r.empty: continue
        pos = r["Pos"].iloc[0]
        base = float(r["BaseValue"].iloc[0]) * float(r["PosMult"].iloc[0])
        mult = apply_need(pos, counts)
        val = base * mult
        vals.append(val)
        details.append((n, val, mult, pos))
    return package_sum(vals), details

def sum_picks_value(pick_labels):
    vals = []; details = []
    for lbl in pick_labels:
        v = label_value(lbl)
        vals.append(v); details.append((lbl, v))
    return package_sum(vals), details

st.header("Build a Trade (two boxes)")
colA, colB = st.columns(2)
with colA:
    teamA = st.selectbox("Team A", teams, index=0 if teams else None)
with colB:
    teamB = st.selectbox("Team B", [t for t in teams if t != (teamA if teams else None)], index=0 if len(teams)>1 else None)

a_players_list = roster_players(teamA) if teamA else []
b_players_list = roster_players(teamB) if teamB else []

left, right = st.columns(2)
with left:
    st.subheader("Team A sends → Team B")
    a_send_players = st.multiselect("Players", a_players_list, format_func=lambda x: label_map.get(x,x), key="a_send_players")
    a_send_picks = st.multiselect("Picks", build_pick_labels_for_team(teamA) if teamA else [], key="a_send_picks")
with right:
    st.subheader("Team B sends → Team A")
    b_send_players = st.multiselect("Players", b_players_list, format_func=lambda x: label_map.get(x,x), key="b_send_players")
    b_send_picks = st.multiselect("Picks", build_pick_labels_for_team(teamB) if teamB else [], key="b_send_picks")

if not manual or recalc:
    A_get_players, A_get_p_det = sum_players_value(b_send_players, teamA)
    A_get_picks,   A_get_pk_det = sum_picks_value(b_send_picks)
    A_get_total = A_get_players + A_get_picks

    B_get_players, B_get_p_det = sum_players_value(a_send_players, teamB)
    B_get_picks,   B_get_pk_det = sum_picks_value(a_send_picks)
    B_get_total = B_get_players + B_get_picks

    m1,m2 = st.columns(2)
    m1.metric("Team A RECEIVES (rank-weighted)", f"{A_get_total:,.0f}")
    m2.metric("Team B RECEIVES (rank-weighted)", f"{B_get_total:,.0f}")

    def roster_context(team, incoming_details):
        before = team_pos_counts(team)
        inc_counts = {"QB":0,"RB":0,"WR":0,"TE":0}
        for name, val, mult, pos in incoming_details:
            inc_counts[pos] = inc_counts.get(pos,0) + 1
        after = {p: before.get(p,0) + inc_counts.get(p,0) for p in ["QB","RB","WR","TE"]}
        tips = []
        for p in ["QB","RB","WR","TE"]:
            tgt = int(targets.get(p,0)); b = before.get(p,0)
            if b > tgt and inc_counts.get(p,0)>0:
                tips.append(f"already has a **surplus at {p}** (+{b - tgt}); extra {p} returns less value.")
            elif b < tgt and inc_counts.get(p,0)>0:
                tips.append(f"is **short at {p}** ({tgt-b} under target); incoming {p} fits a need.")
        return tips

    A_tips = roster_context(teamA, A_get_p_det)
    B_tips = roster_context(teamB, B_get_p_det)

    st.markdown("---")
    st.subheader("Fairness Verdict")
    diff = A_get_total - B_get_total
    larger = max(A_get_total, B_get_total, 1.0)
    pct = diff / larger

    THRESH = 0.03
    if abs(pct) < THRESH:
        st.success("Even trade by rank-weighted value (within ~3%).")
    elif pct > 0:
        st.info(f"**Favors Team A** by {diff:,.0f} (~{pct:.1%}). " +
                (" ".join([f"For A: {t}" for t in A_tips]) if A_tips else "") +
                (" " + " ".join([f"For B: {t}" for t in B_tips]) if B_tips else ""))
    else:
        st.info(f"**Favors Team B** by {abs(diff):,.0f} (~{abs(pct):.1%}). " +
                (" ".join([f"For A: {t}" for t in A_tips]) if A_tips else "") +
                (" " + " ".join([f"For B: {t}" for t in B_tips]) if B_tips else ""))

    with st.expander("Why (details)"):
        st.write("**What Team A receives**")
        if len(A_get_p_det): st.table(pd.DataFrame(A_get_p_det, columns=["Player","Value","Need Mult","Pos"]))
        if len(A_get_pk_det): st.table(pd.DataFrame(A_get_pk_det, columns=["Pick","Value"]))
        st.write("**What Team B receives**")
        if len(B_get_p_det): st.table(pd.DataFrame(B_get_p_det, columns=["Player","Value","Need Mult","Pos"]))
        if len(B_get_pk_det): st.table(pd.DataFrame(B_get_pk_det, columns=["Pick","Value"]))

st.markdown("---")
st.header("Trade Finder — multiple suggestions")

tf_from = st.selectbox("Your team (offering)", teams, key="tf_from")
tf_to   = st.selectbox("Trade partner", [t for t in teams if t != tf_from], key="tf_to")
tf_target = st.selectbox("Target from partner", roster_players(tf_to), format_func=lambda x: f"{x} ({players.loc[players['Player']==x,'Pos'].iloc[0]})" if (players['Player']==x).any() else x)

def player_value_for_team(player_name, team_for_need):
    r = players.loc[players["Player"]==player_name]
    if r.empty: return 0.0, 1.0
    pos = r["Pos"].iloc[0]
    base = float(r["BaseValue"].iloc[0]) * float(r["PosMult"].iloc[0])
    mult = apply_need(pos, team_pos_counts(team_for_need))
    return base * mult, mult

target_val, target_mult = player_value_for_team(tf_target, tf_to)
st.caption(f"Target value for {tf_to}: ~{target_val:,.0f} (need mult: {target_mult:.2f})")

your_names = roster_players(tf_from)
cand = players[players["Player"].isin(your_names)].copy()

def val_to_partner(name):
    v, _ = player_value_for_team(name, tf_to); return v
from_counts = team_pos_counts(tf_from)
def surplus_score(pos): return max(0, from_counts.get(pos,0) - targets.get(pos,0))

cand["ValueToPartner"] = cand["Player"].apply(val_to_partner)
cand["Surplus"] = cand["Pos"].apply(surplus_score)
cand["Score"] = cand["ValueToPartner"] * (1 + 0.2*cand["Surplus"])
cand = cand.sort_values(["Score","ValueToPartner"], ascending=[False, False]).reset_index(drop=True)

TOPN = min(12, len(cand))
LOW, HIGH = 0.90, 1.10

def pick_labels_for(team):
    return build_pick_labels_for_team(team)

def package_value(asset_names):
    vals = [val_to_partner(n) for n in asset_names]
    return package_sum(vals)

def bridge_with_pick(team, current_val, target):
    need = max(0.0, target - current_val)
    if need <= 0: return None, 0.0
    best_lbl, best_delta = None, 0.0
    for lbl in pick_labels_for(team):
        v = label_value(lbl)
        if v >= need*0.6 and (best_lbl is None or abs(v-need) < abs(best_delta-need)):
            best_lbl, best_delta = lbl, v
    if best_lbl is None:
        for lbl in pick_labels_for(team):
            v = label_value(lbl)
            if v > best_delta: best_lbl, best_delta = lbl, v
    return best_lbl, best_delta

if "tf_seed" not in st.session_state:
    st.session_state["tf_seed"] = 0
if st.button("Suggest another trade"):
    st.session_state["tf_seed"] += 1
random.seed(st.session_state["tf_seed"])

suggestions = []
base_list = cand.head(TOPN)["Player"].tolist()
random.shuffle(base_list)

for size in [1,2,3]:
    for combo in combinations(base_list, size):
        val = package_value(combo)
        assets = list(combo)
        pkg_val = val
        if pkg_val < target_val*LOW:
            lbl, dv = bridge_with_pick(tf_from, pkg_val, target_val)
            if lbl:
                assets = assets + [lbl]
                pkg_val = package_sum([val] + [dv])
        if target_val*LOW <= pkg_val <= target_val*HIGH:
            suggestions.append((assets, pkg_val))

seen = set(); uniq_suggestions = []
for assets, v in suggestions:
    key = tuple(sorted(assets))
    if key in seen: continue
    seen.add(key); uniq_suggestions.append((assets, v))

uniq_suggestions.sort(key=lambda x: (len(x[0]), abs(x[1]-target_val), -x[1]))

st.subheader("Suggested Packages")
if not uniq_suggestions:
    st.write("No good suggestions found. Try lowering the 2-for-1 tax or increasing rank importance, or click **Suggest another trade**.")
else:
    for i, (assets, v) in enumerate(uniq_suggestions[:5], start=1):
        st.write(f"**Suggestion {i}** — value to partner: {v:,.0f} (~{v/target_val:.0%} of target)")
        st.write("- " + "\n- ".join(assets))

