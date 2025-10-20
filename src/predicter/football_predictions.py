import os,json
import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, accuracy_score


base_path = r"C:\Users\admin\OneDrive - National University of Singapore\Football Dataset\2425_NEW"

all_players = []

for team_name in os.listdir(base_path):
    team_folder = os.path.join(base_path, team_name)
    if not os.path.isdir(team_folder):
        continue

    for player_name in os.listdir(team_folder):
        player_folder = os.path.join(team_folder, player_name)
        summary_file = os.path.join(player_folder, "summary.json")

        if os.path.isfile(summary_file):
            try:
                with open(summary_file, "r", encoding="utf-8") as f:
                    data = json.load(f)

                # Some files might be dicts, some lists
                if isinstance(data, dict):
                    df = pd.DataFrame([data])
                else:
                    df = pd.DataFrame(data)

                df["Team"] = team_name
                df["Player"] = player_name

                # if df.iloc[-1]['Squad'] == df.iloc[-1]['Team']:
                #     pass
                # else:
                #     df = pd.DataFrame()
                all_players.append(df)
            except Exception as e:
                print(f"⚠️ Error reading {summary_file}: {e}")

player_matches_df = pd.concat(all_players, ignore_index=True)
print("✅ Player rows loaded:", len(player_matches_df))

# Clean missing values
player_matches_df.replace(["", "Match Report", "-", "—"], np.nan, inplace=True)

# Fix Date
player_matches_df["Date"] = pd.to_datetime(player_matches_df["Date"], errors="coerce")
player_matches_df = player_matches_df.sort_values(['Player', 'Date'])

player_matches_df['Prev_Squad'] = player_matches_df.groupby('Player')['Squad'].shift(1)
player_matches_df['Transferred'] = player_matches_df['Squad'] != player_matches_df['Prev_Squad']

transfers = player_matches_df[
    player_matches_df['Transferred'] &
    player_matches_df['Prev_Squad'].notna()
][['Player', 'Prev_Squad', 'Squad', 'Date']]
transfers = transfers.rename(columns={'Prev_Squad': 'From', 'Squad': 'To','Date': 'Date_transferred'})

player_matches_df = player_matches_df.merge(
    transfers[['Player', 'From', 'To', 'Date_transferred']],
    on='Player',
    how='left'
)

player_matches_df['Valid_Team'] = np.where(
    (player_matches_df['Date'] >= player_matches_df['Date_transferred']) &
    (player_matches_df['Player'] == player_matches_df['Player']),
    player_matches_df['To'],
    player_matches_df['Squad']
)


# Convert numeric columns
numeric_cols = ["Min","Gls","Ast","xG","npxG","xAG","SCA","GCA","Cmp","Att","Cmp%","PrgP","Carries","PrgC","Succ"]
for col in numeric_cols:
    if col in player_matches_df.columns:
        player_matches_df[col] = pd.to_numeric(player_matches_df[col], errors="coerce")


# Filter only Premier League matches (in case there are others)
player_matches_df = player_matches_df[player_matches_df["Comp"].str.contains("Premier League", na=False)]

player_matches_df.head(3)

agg_funcs = {
    "Min": "sum",
    "Gls": "sum",
    "Ast": "sum",
    "xG": "sum",
    "xAG": "sum",
    "SCA": "sum",
    "GCA": "sum",
    "PrgP": "sum",
    "Carries": "sum",
    "PrgC": "sum"
}

team_player_summary = (
    player_matches_df.groupby(["Date", "Team"])
    .agg(agg_funcs)
    .reset_index()
    .rename(columns={c: f"player_{c}_sum" for c in agg_funcs.keys()})
)

# Load all team matchlogs.csv files
all_matches = []
for team_name in os.listdir(base_path):
    team_folder = os.path.join(base_path, team_name)
    match_file = os.path.join(team_folder, "matchlogs_for_2024-25.csv")
    if os.path.isfile(match_file):
        df = pd.read_csv(match_file)
        df["Team"] = team_name
        all_matches.append(df)

team_matches_df = pd.concat(all_matches, ignore_index=True)
team_matches_df["Date"] = pd.to_datetime(team_matches_df["Date"], errors="coerce")

# Merge team data with aggregated player data
full_team_df = pd.merge(team_matches_df, team_player_summary, on=["Date", "Team"], how="left")
print("✅ Combined dataset shape:", full_team_df.shape)
full_team_df.head(3)

def result_label(row):
    res = str(row["Result"]).strip().upper()
    if res.startswith("W"):
        return 2  # Home win
    elif res.startswith("L"):
        return 0  # Home loss (away win)
    elif res.startswith("D"):
        return 1  # Draw
    else:
        return np.nan

full_team_df["result_label"] = full_team_df.apply(result_label, axis=1)
full_team_df = full_team_df.dropna(subset=["result_label"]).reset_index(drop=True)
full_team_df["result_label"] = full_team_df["result_label"].astype(int)

home_df = full_team_df[full_team_df["Venue"].str.contains("Home", na=False)].copy()
away_df = full_team_df[full_team_df["Venue"].str.contains("Away", na=False)].copy()

def make_match_id(row):
    teams = sorted([row["Team"], row["Opponent"]])
    date_str = row["Date"].strftime("%Y-%m-%d")
    return f"{teams[0]}__{teams[1]}__{date_str}"

home_df["match_id"] = home_df.apply(make_match_id, axis=1)
away_df["match_id"] = away_df.apply(make_match_id, axis=1)

matches = pd.merge(
    home_df,
    away_df,
    on="match_id",
    suffixes=("_home","_away"),
    how="inner"
)

feature_cols = []

# Rolling/total features
for col in ["GF","GA","xG","xGA","Poss"]:
    if f"{col}_home" in matches.columns and f"{col}_away" in matches.columns:
        feature_cols += [f"{col}_home", f"{col}_away"]

# Player aggregates
for col in matches.columns:
    if "player_" in col and ("_home" in col or "_away" in col):
        feature_cols.append(col)

numeric_cols = ["GF_home","GF_away","GA_home","GA_away"]

for col in numeric_cols:
    matches[col] = pd.to_numeric(matches[col], errors="coerce")  # Non-numeric becomes NaN

# Fill missing numeric values (optional, fill 0)
matches[numeric_cols] = matches[numeric_cols].fillna(0)


X = matches[feature_cols].fillna(0)
y = matches["result_label_home"]  # 0=Away win,1=Draw,2=Home win

split_date = matches["Date_home"].quantile(0.8)
X_train = X[matches["Date_home"] <= split_date]
y_train = y[matches["Date_home"] <= split_date]
X_test = X[matches["Date_home"] > split_date]
y_test = y[matches["Date_home"] > split_date]

model = XGBClassifier(
    objective="multi:softprob",
    num_class=3,
    n_estimators=200,
    learning_rate=0.05,
    max_depth=4,
    use_label_encoder=False,
    eval_metric="mlogloss",
    random_state=42
)

model.fit(X_train, y_train)
pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, pred))
print(classification_report(y_test, pred, target_names=["AwayWin","Draw","HomeWin"]))
