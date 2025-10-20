# import pandas as pd
# import numpy as np

# # --- Paths ---
# matches_path = r"C:\Users\admin\OneDrive - National University of Singapore\Football Dataset\matches.csv"
# player_features_path = r"C:\Users\admin\OneDrive - National University of Singapore\Football Dataset\aggregated_player_features_with_inactivity.csv"

# # --- Load data ---
# matches = pd.read_csv(matches_path)
# player_features = pd.read_csv(player_features_path)

# # --- Keep only Premier League matches ---
# matches = matches[matches["Comp"] == "Premier League"].copy()

# # --- Ensure proper datatypes ---
# matches["Date"] = pd.to_datetime(matches["Date"], errors="coerce")
# player_features["Date"] = pd.to_datetime(player_features["Date"], errors="coerce")

# # --- Merge home and away team features ---
# matches = matches.merge(
#     player_features.add_suffix("_home"),
#     left_on=["Team", "Date"],
#     right_on=["Squad_home", "Date_home"],
#     how="left"
# )
# matches = matches.merge(
#     player_features.add_suffix("_away"),
#     left_on=["Opponent", "Date"],
#     right_on=["Squad_away", "Date_away"],
#     how="left"
# )

# # --- Encode match result (target) ---
# matches["result_label"] = matches["Result"].map({"W": 2, "D": 1, "L": 0})

# # --- Feature: Goal differences ---
# matches["goal_diff"] = matches["GF"] - matches["GA"]
# matches["xG_diff"] = matches["xG"] - matches["xGA"]

# # --- Feature: Possession difference ---
# matches["Poss_diff"] = matches["Poss"].astype(float) - matches.get("Poss_away", matches["Poss"]).astype(float)

# # --- Feature: Team strength metrics ---
# # Simple example — average of xG, xAG, SoT, and Tkl + Int
# for side in ["home", "away"]:
#     matches[f"team_strength_{side}"] = (
#         matches[f"xG_{side}"].fillna(0) * 0.4 +
#         matches[f"xAG_{side}"].fillna(0) * 0.2 +
#         matches[f"SoT_{side}"].fillna(0) * 0.2 +
#         matches[f"Tkl_{side}"].fillna(0) * 0.1 +
#         matches[f"Int_{side}"].fillna(0) * 0.1
#     )

# # --- Feature: Team strength difference ---
# matches["team_strength_diff"] = matches["team_strength_home"] - matches["team_strength_away"]

# # --- Rolling form feature ---
# matches = matches.sort_values(["Team", "Date"])
# matches["recent_form"] = (
#     matches.groupby("Team")["result_label"]
#     .transform(lambda x: x.rolling(5, min_periods=1).mean())
# )

# # --- Aggregate team form into home/away context ---
# home_form = matches[["Team", "Date", "recent_form"]].rename(
#     columns={"Team": "home_team", "recent_form": "recent_form_home"}
# )
# away_form = matches[["Team", "Date", "recent_form"]].rename(
#     columns={"Team": "away_team", "recent_form": "recent_form_away"}
# )

# matches = matches.merge(
#     home_form, left_on=["Team", "Date"], right_on=["home_team", "Date"], how="left"
# )
# matches = matches.merge(
#     away_form, left_on=["Opponent", "Date"], right_on=["away_team", "Date"], how="left"
# )

# # --- Recent form difference ---
# matches["form_diff"] = matches["recent_form_home"].fillna(0) - matches["recent_form_away"].fillna(0)

# # --- Fill NaN values for numeric columns ---
# numeric_cols = matches.select_dtypes(include=[np.number]).columns
# matches[numeric_cols] = matches[numeric_cols].fillna(0)

# # --- Drop redundant columns ---
# drop_cols = [col for col in matches.columns if col.startswith(("Squad_", "Date_", "home_team", "away_team"))]
# matches = matches.drop(columns=drop_cols, errors="ignore")

# # --- Save engineered features ---
# out_path = r"C:\Users\admin\OneDrive - National University of Singapore\Football Dataset\matches_features.csv"
# matches.to_csv(out_path, index=False)

# print("✅ Feature engineering complete!")
# print(f"🧩 Total matches processed: {len(matches)}")
# print(f"🏟️ Features available: {len(numeric_cols)} numeric columns")

# import pandas as pd
# import numpy as np

# # --- Load your feature engineered matches data ---
# matches = pd.read_csv(
#     r"C:\Users\admin\OneDrive - National University of Singapore\Football Dataset\matches_features.csv"
# )
# matches["Date"] = pd.to_datetime(matches["Date"])
# matches = matches.sort_values("Date").reset_index(drop=True)

# # --- Initialize Elo parameters ---
# BASE_ELO = 1500
# K_FACTOR = 20

# # --- Dictionary to hold live ratings ---
# team_elos = {}

# # --- Lists to store evolving Elo values per match ---
# elo_home_list = []
# elo_away_list = []

# for _, row in matches.iterrows():
#     home = row["Team"]
#     away = row["Opponent"]

#     # Get current Elo ratings (default = BASE_ELO)
#     elo_home = team_elos.get(home, BASE_ELO)
#     elo_away = team_elos.get(away, BASE_ELO)

#     # Expected outcomes
#     expected_home = 1 / (1 + 10 ** ((elo_away - elo_home) / 400))
#     expected_away = 1 - expected_home

#     # Actual outcomes
#     result = row["result_label"]
#     if result == 1:  # home win
#         score_home, score_away = 1, 0
#     elif result == 0:  # draw
#         score_home, score_away = 0.5, 0.5
#     else:  # away win
#         score_home, score_away = 0, 1

#     # Elo update
#     new_elo_home = elo_home + K_FACTOR * (score_home - expected_home)
#     new_elo_away = elo_away + K_FACTOR * (score_away - expected_away)

#     # Save updated ratings
#     team_elos[home] = new_elo_home
#     team_elos[away] = new_elo_away

#     # Store pre-match Elo (for feature use)
#     elo_home_list.append(elo_home)
#     elo_away_list.append(elo_away)

# # --- Add Elo columns to dataframe ---
# matches["elo_home_before"] = elo_home_list
# matches["elo_away_before"] = elo_away_list
# matches["elo_diff"] = matches["elo_home_before"] - matches["elo_away_before"]

# # --- Save enhanced dataset ---
# matches.to_csv(
#     r"C:\Users\admin\OneDrive - National University of Singapore\Football Dataset\matches_with_elo.csv",
#     index=False
# )

# print("✅ Elo ratings added successfully!")
# print(f"🏟️ Teams rated: {len(team_elos)}")
# print(f"📈 Elo range: {min(team_elos.values()):.1f} - {max(team_elos.values()):.1f}")

import pandas as pd
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

matches = pd.read_csv(
    r"C:\Users\admin\OneDrive - National University of Singapore\Football Dataset\matches_with_elo.csv"
)

#feature_cols = [col for col in matches.columns if col.endswith("_home") or col.endswith("_away") or "_diff" in col]

#feature_cols = ["xG_diff", "Poss_diff","goal_diff", "team_strength_diff", "form_diff", "elo_diff"]

feature_cols = ["team_strength_diff", "form_diff", "elo_diff"]

X = matches[feature_cols]
y = matches["result_label"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

model = XGBClassifier(
    objective="multi:softprob",
    num_class=3,
    eval_metric="mlogloss",
    use_label_encoder=False,
    n_estimators=300,
    learning_rate=0.05,
    max_depth=4,
    random_state=42
)
model.fit(X_train, y_train)

preds = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, preds))
print("Confusion Matrix:\n", confusion_matrix(y_test, preds))

# Get last known team state from matches.csv
team_latest = matches.sort_values("Date").groupby("Team").tail(1)

# Select only relevant columns
team_features = team_latest[
    ["Team", "elo_home_before", "team_strength_home", "recent_form_home"]
].rename(columns={
    "elo_home_before": "elo_latest",
    "team_strength_home": "strength_latest",
    "recent_form_home": "form_latest"
})


matches_test = matches = pd.read_csv(
    r"C:\Users\admin\OneDrive - National University of Singapore\Football Dataset\matches_test.csv"
)

matches_test["Date"] = pd.to_datetime(matches_test["Date"], errors="coerce")

# Merge for home team
matches_test = matches_test.merge(
    team_features.add_suffix("_home"),
    left_on="Team", right_on="Team_home", how="left"
)

# Merge for away team
matches_test = matches_test.merge(
    team_features.add_suffix("_away"),
    left_on="Opponent", right_on="Team_away", how="left"
)

matches_test["elo_diff"] = matches_test["elo_latest_home"] - matches_test["elo_latest_away"]
matches_test["team_strength_diff"] = matches_test["strength_latest_home"] - matches_test["strength_latest_away"]
matches_test["form_diff"] = matches_test["form_latest_home"] - matches_test["form_latest_away"]

# If you don’t have xG_diff or Poss_diff yet, set defaults to 0
matches_test["xG_diff"] = 0
matches_test["Poss_diff"] = 0
matches_test["goal_diff"] = 0

for col in feature_cols:
    if col not in matches_test.columns:
        # Create missing columns with 0 as default (neutral)
        matches_test[col] = 0

X_test = matches_test[feature_cols]

# Predict
preds = model.predict(X_test)
pred_probs = model.predict_proba(X_test)

matches_test["predicted_result"] = preds
matches_test["prob_loss"] = pred_probs[:, 0]
matches_test["prob_draw"] = pred_probs[:, 1]
matches_test["prob_win"] = pred_probs[:, 2]

result_map = {0: "Loss", 1: "Draw", 2: "Win"}
matches_test["predicted_result_label"] = matches_test["predicted_result"].map(result_map)

matches_test.to_csv("matches_test_predicted.csv", index=False)