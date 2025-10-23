import pandas as pd
import numpy as np
from xgboost import XGBRegressor, XGBClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error, accuracy_score, classification_report
import math
import joblib
from pathlib import Path

# ---------------- CONFIG ----------------
BASE = Path(r"C:\Users\admin\OneDrive - National University of Singapore\Football Dataset")
MATCHES_PATH = BASE / "matches_with_elo_new.csv"
PLAYERS_PATH = BASE / "merged_player_data_new.csv"   # user said merged_player.csv earlier
MATCHES_TEST_PATH = BASE / "matches_test.csv"
OUT_PRED_PATH = "matches_test_predicted_with_player_elo.csv"

# Elo / K settings (position-aware)
BASE_ELO = 1500.0
K_POS = {"GK": 8.0, "DEF": 10.0, "MID": 12.0, "ATT": 14.0}  # sensitivity by pos group
TEAM_K = 18.0  # team-level Elo K for reference (not strictly used but available)

# Position grouping mapping (user positions)
position_groups = {
    "GK": ["GK"],
    "DEF": ["LB", "CB", "RB", "WB"],
    "MID": ["CM", "DM", "LM", "RM", "AM"],
    "ATT": ["LW", "RW", "FW"]
}

# ---------------- Helpers ----------------
def map_pos_group(pos):
    if pd.isna(pos):
        return "OTHER"
    p = str(pos).upper().strip()
    # ensure we strip whitespace and consider first token if weird
    p = p.split("/")[0].split("-")[0]
    for g, lst in position_groups.items():
        if p in lst:
            return g
    # fallback: some players have "RW,AM" etc; they should be exploded earlier
    return "OTHER"

def safe_min_to_float(x):
    # convert minutes string like "90" or "90'" or empty to float
    try:
        if pd.isna(x):
            return 0.0
        s = str(x).replace("'", "").strip()
        if s == "":
            return 0.0
        return float(s)
    except:
        return 0.0

# Performance scoring (leak-free). Uses only observed count stats that are available after match.
def calculate_player_score(row):
    # uses goals, assists, shots on target, tackles, interceptions, blocks, progressive passes, SCA/GCA, cards, minutes
    posg = row.get("PosGroup", "OTHER")
    mins = safe_min_to_float(row.get("Min", 0))
    # get numeric safely
    def n(k): 
        return float(row.get(k, 0) if not pd.isna(row.get(k, 0)) else 0.0)
    score = 0.0
    if posg == "ATT":
        score += 4.0 * n("Gls")
        score += 2.0 * n("Ast")
        score += 0.3 * n("SoT")
        score += 0.15 * n("Sh")
        score += 0.2 * n("SCA")
        score += 0.25 * n("GCA")
        score -= 0.5 * n("CrdY") + 2.0 * n("CrdR")
    elif posg == "MID":
        score += 2.0 * n("Ast")
        score += 0.02 * n("PrgP")  # progressive passes distance
        score += 0.2 * n("Carries") if "Carries" in row else 0.0
        score += 0.25 * n("SCA") + 0.2 * n("GCA")
        score += 0.3 * n("Tkl") + 0.3 * n("Int")
        score -= 0.3 * n("CrdY") + 1.5 * n("CrdR")
    elif posg == "DEF":
        score += 0.8 * n("Tkl") + 0.6 * n("Int") + 0.6 * n("Blocks")
        score += 0.05 * n("Touches")
        score -= 0.3 * n("CrdY") + 1.5 * n("CrdR") + 3.0 * n("OG") if "OG" in row else 0.0
    elif posg == "GK":
        # we may not have saves directly; use SoT conceded vs GA if available, else touches and clean sheet proxies
        # Use touches somewhat, cards penalty
        score += 0.02 * n("Touches")
        score -= 3.0 * n("GA") if "GA" in row else 0.0
        score -= 2.0 * n("CrdY") + 8.0 * n("CrdR")
    else:
        # default blend
        score += 1.0 * n("Gls") + 0.8 * n("Ast") + 0.2 * n("SoT")
        score += 0.2 * n("Tkl") + 0.2 * n("Int")
    # normalize by minutes
    if mins > 0:
        score = score * (mins / 90.0)
    else:
        score = 0.0
    # small bonus for team win / penalty for loss will be added via team result influence when updating
    return float(score)

# safe expected probability given rating difference
def elo_expected(r_a, r_b):
    return 1.0 / (1.0 + 10 ** ((r_b - r_a) / 400.0))

# ---------------- Load data ----------------
print("Loading data...")
matches = pd.read_csv(MATCHES_PATH, encoding="utf-8")
players = pd.read_csv(PLAYERS_PATH, encoding="utf-8")

# ensure date parsing
matches["Date"] = pd.to_datetime(matches["Date"], errors="coerce")
players["Date"] = pd.to_datetime(players["Date"], errors="coerce")

# Standardize column names for matches: ensure Team (home), Opponent (away), GF, GA exist
# The user's dataset uses "Team" and "Opponent" commonly. Try to support variants.
for col in ["Team", "team", "Squad"]:
    if col in matches.columns:
        matches.rename(columns={col: "Team"}, inplace=True)
        break
for col in ["Opponent", "Opp", "opp"]:
    if col in matches.columns:
        matches.rename(columns={col: "Opponent"}, inplace=True)
        break

# GF/GA variants
if "GF" not in matches.columns and "FTHG" in matches.columns:
    matches.rename(columns={"FTHG": "GF", "FTAG": "GA", "HomeTeam": "Team", "AwayTeam": "Opponent"}, inplace=True)

# Filter Premier League if present
if "Comp" in matches.columns:
    matches = matches[matches["Comp"].str.contains("Premier League", na=False) | (matches["Comp"] == "Premier League")]

# Preprocess players: explode multi-pos, compute PosGroup and Min numeric
players["Pos"] = players["Pos"].astype(str).str.replace(" ", "")
players = players.assign(Pos=players["Pos"].str.split(",")).explode("Pos")
players["PosGroup"] = players["Pos"].apply(map_pos_group)
players["Min"] = players["Min"].apply(safe_min_to_float)

# For speed, create index by (Team, Date) for quick lookup of players that played that match for that team
players_index = players.set_index(["Team", "Date"])

# Initialize player Elo dictionary and last_seen mapping
player_elo = {}   # player -> elo
player_last_seen = {}  # player -> last date they played (for filtering)

# If players have a column "Player" name
if "Player" not in players.columns:
    raise RuntimeError("Players CSV must contain 'Player' column.")

# Initialize all known players to BASE_ELO
unique_players = players["Player"].unique()
for p in unique_players:
    player_elo[p] = BASE_ELO
    player_last_seen[p] = pd.Timestamp("1970-01-01")

# Helper: compute team positional elos *before* a given match_date
def compute_team_pos_elos(team, match_date, players_df):
    """
    For a team and a match_date, compute the mean player_elo among players who played for that team
    in matches strictly before match_date, grouped by PosGroup. If no players in group, fallback to BASE_ELO.
    """
    # find all player rows for this team with Date < match_date and Min>0
    df = players_df[(players_df["Team"] == team) & (players_df["Date"] < match_date) & (players_df["Min"] > 0)]
    if df.empty:
        # fallback: use overall team average (BASE_ELO)
        return {"elo_gk": BASE_ELO, "elo_def": BASE_ELO, "elo_mid": BASE_ELO, "elo_att": BASE_ELO}
    # We want unique players and their last known PosGroup before match_date
    players_last = (df.sort_values(["Player", "Date"])
                      .groupby("Player", as_index=False)
                      .last()[["Player", "PosGroup"]])
    # get elos
    grouped = players_last.groupby("PosGroup")["Player"].apply(list).to_dict()
    def mean_elo_for_group(group_players):
        vals = [player_elo.get(p, BASE_ELO) for p in group_players if p in player_elo]
        if len(vals) == 0:
            return BASE_ELO
        return float(np.mean(vals))
    elo_gk = mean_elo_for_group(grouped.get("GK", []))
    elo_def = mean_elo_for_group(grouped.get("DEF", []))
    elo_mid = mean_elo_for_group(grouped.get("MID", []))
    elo_att = mean_elo_for_group(grouped.get("ATT", []))
    # If any are absent (no players of that group), fallback to team mean
    team_mean = np.mean([v for v in [elo_gk, elo_def, elo_mid, elo_att] if v is not None])
    if math.isnan(team_mean):
        team_mean = BASE_ELO
    # fill if weird
    return {
        "elo_gk": elo_gk if not math.isnan(elo_gk) else team_mean,
        "elo_def": elo_def if not math.isnan(elo_def) else team_mean,
        "elo_mid": elo_mid if not math.isnan(elo_mid) else team_mean,
        "elo_att": elo_att if not math.isnan(elo_att) else team_mean
    }

# Helper: update player_elos after a match using that match's player rows (players DataFrame)
def update_player_elos_for_match(team, opponent, match_date, result_for_team, players_df):
    """
    result_for_team: 1.0 win, 0.5 draw, 0.0 loss for the given team
    For each player who played for 'team' at 'match_date', compute player_score and update their elo.
    The update rule is heuristic:
      new_elo = old_elo + K_pos * ( (result_for_team - 0.5) + (score_norm) )
    where score_norm is (player_score / 10.0) normalized; tuning is required.
    """
    key = (team, match_date)
    if key not in players_index.index:
        return
    try:
        rows = players_index.loc[key]
    except KeyError:
        # no players recorded
        return
    # rows may be a DataFrame or Series for single match
    if isinstance(rows, pd.Series):
        rows = rows.to_frame().T
    # compute player_score for each and update
    for _, prow in rows.iterrows():
        player = prow["Player"]
        posg = prow.get("PosGroup", map_pos_group(prow.get("Pos", "")))
        score = calculate_player_score(prow)
        # normalize score to roughly [-3, +8] typical depending on metrics; divide by 10
        score_norm = score / 10.0
        K = K_POS.get(posg, 10.0)
        # team result component (centered at -0.5..+0.5)
        team_comp = (result_for_team - 0.5)
        # total delta
        delta = K * (team_comp + 0.5 * score_norm)
        # update
        old = player_elo.get(player, BASE_ELO)
        player_elo[player] = float(old + delta)
        # record last seen date
        player_last_seen[player] = max(player_last_seen.get(player, pd.Timestamp("1970-01-01")), match_date)

# ---------------- Build training features by iterating matches chronologically ----------------
print("Building match-by-match features (pre-match snapshots and post-match player elo updates)...")
matches_sorted = matches.sort_values("Date").reset_index(drop=True)

feature_rows = []

# We'll compute rolling form points per team as well; keep a dict with last N results
from collections import deque, defaultdict
team_recent_points = defaultdict(lambda: deque(maxlen=5))  # last 5 match points (3/1/0)

for idx, m in matches_sorted.iterrows():
    date = m["Date"]
    # Skip incomplete matches (no GF/GA)
    try:
        gh = float(m.get("GF", np.nan))
        ga = float(m.get("GA", np.nan))
    except:
        gh, ga = np.nan, np.nan
    if pd.isna(date):
        continue

    home = m.get("Team")  # user's column naming
    away = m.get("Opponent")
    if pd.isna(home) or pd.isna(away):
        # try alternative columns
        home = m.get("Squad", home)
        away = m.get("Opponent", away)
    # compute pre-match team positional elos (only using matches before 'date')
    home_pos_elos = compute_team_pos_elos(home, date, players)
    away_pos_elos = compute_team_pos_elos(away, date, players)

    # compute summary features
    elo_gk_diff = home_pos_elos["elo_gk"] - away_pos_elos["elo_gk"]
    elo_def_diff = home_pos_elos["elo_def"] - away_pos_elos["elo_def"]
    elo_mid_diff = home_pos_elos["elo_mid"] - away_pos_elos["elo_mid"]
    elo_att_diff = home_pos_elos["elo_att"] - away_pos_elos["elo_att"]

    # team_strength_diff or other columns may exist in matches; fallback compute simple proxy:
    # team_strength = sum of positional elos (home) - away
    team_strength_home = sum(home_pos_elos.values())
    team_strength_away = sum(away_pos_elos.values())
    team_strength_diff = team_strength_home - team_strength_away

    # recent form (mean points last 5) for home and away
    recent_home = np.mean(team_recent_points[home]) if len(team_recent_points[home])>0 else 0.0
    recent_away = np.mean(team_recent_points[away]) if len(team_recent_points[away])>0 else 0.0
    form_diff = recent_home - recent_away

    # venue: if matches rows have Venue and "Home" is for the Team, then venue_home = 1 else 1 by convention
    venue = str(m.get("Venue", "Home"))
    venue_home = 1 if venue.lower().startswith("home") else 0

    # Build feature row (these are pre-match!)
    row_features = {
        "Date": date,
        "Home": home,
        "Away": away,
        "elo_gk_home": home_pos_elos["elo_gk"],
        "elo_def_home": home_pos_elos["elo_def"],
        "elo_mid_home": home_pos_elos["elo_mid"],
        "elo_att_home": home_pos_elos["elo_att"],
        "elo_gk_away": away_pos_elos["elo_gk"],
        "elo_def_away": away_pos_elos["elo_def"],
        "elo_mid_away": away_pos_elos["elo_mid"],
        "elo_att_away": away_pos_elos["elo_att"],
        "elo_gk_diff": elo_gk_diff,
        "elo_def_diff": elo_def_diff,
        "elo_mid_diff": elo_mid_diff,
        "elo_att_diff": elo_att_diff,
        "team_strength_diff": team_strength_diff,
        "recent_form_home": recent_home,
        "recent_form_away": recent_away,
        "form_diff": form_diff,
        "venue_home": venue_home,
        # target labels (if present)
        "GF": gh,
        "GA": ga
    }
    feature_rows.append(row_features)

    # --- Now update player_elos using this match's actual player performances (post-match) ---
    # need to determine match outcome per team (1/0.5/0)
    if not pd.isna(gh) and not pd.isna(ga):
        if gh > ga:
            res_home, res_away = 1.0, 0.0
            points_home, points_away = 3, 0
        elif gh == ga:
            res_home, res_away = 0.5, 0.5
            points_home, points_away = 1, 1
        else:
            res_home, res_away = 0.0, 1.0
            points_home, points_away = 0, 3
        # update team recent points
        team_recent_points[home].append(points_home)
        team_recent_points[away].append(points_away)
        # update player elos for both teams using their players rows for this match
        update_player_elos_for_match(home, away, date, res_home, players)
        update_player_elos_for_match(away, home, date, res_away, players)
    else:
        # match not played (no score) -> don't update
        pass

# Convert to DataFrame
features_df = pd.DataFrame(feature_rows)
# Drop rows w/o targets for training (unplayed matches)
train_df = features_df.dropna(subset=["GF","GA"]).reset_index(drop=True)
print("Built", len(train_df), "training matches with pre-match features.")

from sklearn.metrics import mean_squared_error

# ---------------- Train models ----------------
# Features for models
feat_cols = [
    "elo_gk_home","elo_def_home","elo_mid_home","elo_att_home",
    "elo_gk_away","elo_def_away","elo_mid_away","elo_att_away",
    "elo_gk_diff","elo_def_diff","elo_mid_diff","elo_att_diff",
    "team_strength_diff","form_diff","venue_home","recent_form_home","recent_form_away"
]

# Ensure numeric and fill na
train_X = train_df[feat_cols].fillna(BASE_ELO)
train_y_home = train_df["GF"].astype(float).fillna(0)
train_y_away = train_df["GA"].astype(float).fillna(0)

# For classification outcome label based on GF/GA (0=away win,1=draw,2=home win)
def label_from_scores(gf, ga):
    if gf > ga: return 2
    if gf == ga: return 1
    return 0
train_df["label"] = train_df.apply(lambda r: label_from_scores(r["GF"], r["GA"]), axis=1)
train_y_label = train_df["label"]

print("Training regressors for home/away goals and classifier for match result...")

# Regressor for home goals
model_home = XGBRegressor(n_estimators=400, learning_rate=0.05, max_depth=5, random_state=42)
model_home.fit(train_X, train_y_home)

# Regressor for away goals
model_away = XGBRegressor(n_estimators=400, learning_rate=0.05, max_depth=5, random_state=42)
model_away.fit(train_X, train_y_away)

# Classifier for match result
model_cls = XGBClassifier(n_estimators=400, learning_rate=0.05, max_depth=5, use_label_encoder=False, eval_metric="mlogloss", random_state=42)
model_cls.fit(train_X, train_y_label)

# Optional: quick in-sample metrics
pred_home_in = model_home.predict(train_X)
pred_away_in = model_away.predict(train_X)
pred_label_in = model_cls.predict(train_X)
print("In-sample MAE home goals:", mean_absolute_error(train_y_home, pred_home_in))
print("In-sample MAE away goals:", mean_absolute_error(train_y_away, pred_away_in))
print("In-sample MSE home goals:", mean_squared_error(train_y_home, pred_home_in))
print("In-sample MSE away goals:", mean_squared_error(train_y_away, pred_away_in))

print("In-sample accuracy (result):", accuracy_score(train_y_label, pred_label_in))
print(classification_report(train_y_label, pred_label_in, target_names=["AwayWin","Draw","HomeWin"]))

# ---------------- Predict on upcoming matches (matches_test.csv) ----------------
if True:
    print("Predicting upcoming matches from matches_test.csv ...")
    matches_test = pd.read_csv(MATCHES_TEST_PATH)
    matches_test = matches_test[matches_test["Venue"] == "Home"]
    matches_test["Date"] = pd.to_datetime(matches_test["Date"], errors="coerce")
    # normalize column names
    if "Team" not in matches_test.columns:
        for col in ["team","Squad","HomeTeam"]:
            if col in matches_test.columns:
                matches_test.rename(columns={col:"Team"}, inplace=True)
                break
    if "Opponent" not in matches_test.columns:
        for col in ["Opponent","Opp","AwayTeam"]:
            if col in matches_test.columns:
                matches_test.rename(columns={col:"Opponent"}, inplace=True)
                break

    preds_rows = []
    # compute features for each upcoming match using latest player_elo snapshot (which was updated above)
    for _, rm in matches_test.iterrows():
        mdate = rm["Date"]
        home = rm["Team"]
        away = rm["Opponent"]
        if pd.isna(mdate) or pd.isna(home) or pd.isna(away):
            continue
        # compute team pos elos using pre-match players (players with Date < match_date)
        home_pos = compute_team_pos_elos(home, mdate, players)
        away_pos = compute_team_pos_elos(away, mdate, players)
        feat = {
            "Date": mdate, "Home": home, "Away": away,
            "elo_gk_home": home_pos["elo_gk"], "elo_def_home": home_pos["elo_def"],
            "elo_mid_home": home_pos["elo_mid"], "elo_att_home": home_pos["elo_att"],
            "elo_gk_away": away_pos["elo_gk"], "elo_def_away": away_pos["elo_def"],
            "elo_mid_away": away_pos["elo_mid"], "elo_att_away": away_pos["elo_att"],
        }
        # derived
        feat["elo_gk_diff"] = feat["elo_gk_home"] - feat["elo_gk_away"]
        feat["elo_def_diff"] = feat["elo_def_home"] - feat["elo_def_away"]
        feat["elo_mid_diff"] = feat["elo_mid_home"] - feat["elo_mid_away"]
        feat["elo_att_diff"] = feat["elo_att_home"] - feat["elo_att_away"]
        feat["team_strength_diff"] = (feat["elo_gk_home"]+feat["elo_def_home"]+feat["elo_mid_home"]+feat["elo_att_home"]) - (feat["elo_gk_away"]+feat["elo_def_away"]+feat["elo_mid_away"]+feat["elo_att_away"])
        # recent form: use team_recent_points (we've maintained it for training matches) — compute mean or 0
        feat["recent_form_home"] = np.mean(team_recent_points[home]) if len(team_recent_points[home])>0 else 0.0
        feat["recent_form_away"] = np.mean(team_recent_points[away]) if len(team_recent_points[away])>0 else 0.0
        feat["form_diff"] = feat["recent_form_home"] - feat["recent_form_away"]
        venue = str(rm.get("Venue", "Home"))
        feat["venue_home"] = 1 if str(venue).lower().startswith("home") else 0

        # construct model input vector
        Xvec = pd.DataFrame([feat])[feat_cols].fillna(BASE_ELO)
        # predictions
        
        res_map = {0:"AwayWin",1:"Draw",2:"HomeWin"}
        lambda_h = model_home.predict(Xvec)[0]
        lambda_a = model_away.predict(Xvec)[0]

        pred_result = "HomeWin" if int(max(0, round(lambda_h)))>int(max(0, round(lambda_a))) else ("AwayWin" if int(max(0, round(lambda_a)))>int(max(0, round(lambda_h))) else "Draw")

        # round and clip
        preds_rows.append({
            "Date": mdate, "Home": home, "Away": away,
            "pred_home_goals": int(max(0, round(lambda_h))), "pred_away_goals": int(max(0, round(lambda_a))),
            "pred_goal_diff": int(max(0, round(lambda_h))) - int(max(0, round(lambda_a))), "pred_result": pred_result
        })

    preds_df = pd.DataFrame(preds_rows)
    preds_df.to_csv(OUT_PRED_PATH, index=False)
    print(f"Saved predictions for upcoming matches to: {OUT_PRED_PATH}")
    print(preds_df.head())
else:
    print("No matches_test.csv found; training completed and in-sample metrics printed above.")

# Optionally save models
joblib.dump(model_home, "model_home_goals_xgb.joblib")
joblib.dump(model_away, "model_away_goals_xgb.joblib")
joblib.dump(model_cls,"model_result_xgb.joblib")
print("Saved models to disk.")