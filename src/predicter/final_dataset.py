import os, json
import pandas as pd

base_dir = r"C:\Users\admin\OneDrive - National University of Singapore\Football Dataset"
seasons = ["2425_NEW", "2526_NEW"]

all_players = []

for season in seasons:
    season_path = os.path.join(base_dir, season)
    if not os.path.isdir(season_path):
        continue

    for team in os.listdir(season_path):
        team_path = os.path.join(season_path, team)
        if not os.path.isdir(team_path):
            continue

        for player in os.listdir(team_path):
            player_path = os.path.join(team_path, player)
            summary_file = os.path.join(player_path, "summary.json")
            if os.path.exists(summary_file):
                try:
                    with open(summary_file, "r", encoding="utf-8") as f:
                        data = json.load(f)
                    df = pd.DataFrame(data)
                    df["Player"] = player
                    df["Team"] = team
                    df["Season"] = season  # tag season
                    all_players.append(df)
                except Exception as e:
                    print(f"Error reading {summary_file}: {e}")

player_matches_df = pd.concat(all_players, ignore_index=True)

# --- Convert and sort by date ---
player_matches_df['Date'] = pd.to_datetime(player_matches_df['Date'])
player_matches_df = player_matches_df.sort_values(['Player', 'Date']).reset_index(drop=True)

player_matches_df.to_csv(os.path.join(base_dir, "merged_player_data.csv"), index=False)

# --- Identify previous squad and date for each player ---
player_matches_df['Prev_Squad'] = player_matches_df.groupby('Player')['Squad'].shift(1)
player_matches_df["Prev_Date"] = player_matches_df.groupby("Player")["Date"].shift(1)

# --- Detect transfers ---
player_matches_df['Transferred'] = player_matches_df['Squad'] != player_matches_df['Prev_Squad']

# --- Detect inactivity (e.g., > 60 days since last appearance) ---
INACTIVITY_THRESHOLD = 60
player_matches_df["Days_Since_Last_Match"] = (player_matches_df["Date"] - player_matches_df["Prev_Date"]).dt.days
player_matches_df["Inactive"] = player_matches_df["Days_Since_Last_Match"] > INACTIVITY_THRESHOLD


transfers = player_matches_df[
    player_matches_df['Transferred'] & player_matches_df['Prev_Squad'].notna()
][['Player', 'Prev_Squad', 'Squad', 'Date']].rename(columns={'Prev_Squad': 'From', 'Squad': 'To'})

player_matches_df = player_matches_df[player_matches_df['Comp'] == 'Premier League']

# --- Clean temporary columns ---
players_df = player_matches_df.drop(columns=["Prev_Squad", "Transferred", "Prev_Date"], errors="ignore")


# --- Adjust squads for transfers (same as before) ---
for _, row in transfers.iterrows():
    player = row["Player"]
    old_team = row["Prev_Squad"]
    new_team = row["Squad"]
    transfer_date = row["Date"]

    # Before transfer: ensure old team
    mask_before = (
        (player_matches_df["Player"] == player)
        & (player_matches_df["Date"] < transfer_date)
        & (player_matches_df["Squad"] != old_team)
    )
    player_matches_df.loc[mask_before, "Squad"] = old_team

    # After transfer: ensure new team
    mask_after = (
        (player_matches_df["Player"] == player)
        & (player_matches_df["Date"] >= transfer_date)
        & (player_matches_df["Squad"] != new_team)
    )
    player_matches_df.loc[mask_after, "Squad"] = new_team

# --- Mark inactive records ---
player_matches_df["Active"] = ~player_matches_df["Inactive"]

# --- Clean temporary columns ---
players_df = player_matches_df.drop(columns=["Prev_Squad", "Transferred", "Prev_Date"], errors="ignore")

# --- Aggregate features by team/date (only active players) ---
agg_features = {
    "xG": "mean",
    "xAG": "mean",
    "Touches": "mean",
    "Tkl": "sum",
    "Int": "sum",
    "Sh": "sum",
    "SoT": "sum",
    "CrdY": "sum",
    "CrdR": "sum"
}

active_players_df = players_df[players_df["Active"] == True]

team_features = (
    active_players_df.groupby(["Squad", "Date"])
    .agg(agg_features)
    .reset_index()
)

# --- Save outputs ---
team_features.to_csv(
    r"C:\Users\admin\OneDrive - National University of Singapore\Football Dataset\aggregated_player_features_with_inactivity.csv",
    index=False
)

players_df.to_csv(
    r"C:\Users\admin\OneDrive - National University of Singapore\Football Dataset\players_with_inactivity_flags.csv",
    index=False
)

print("✅ Transfer + Inactivity aware player data processed successfully!")
print(f"📅 Total unique dates: {team_features['Date'].nunique()}")
print(f"🧩 Active players aggregated: {active_players_df['Player'].nunique()}")
print(f"🚫 Inactive players flagged: {players_df['Inactive'].sum()}")

all_matches = []

for season in seasons:
    season_path = os.path.join(base_dir, season)
    for team in os.listdir(season_path):
        if season == "2425_NEW":
            file = f"matchlogs_for_2024-25.csv"
        else:
            file = f"matchlogs_for.csv"
        match_file = os.path.join(season_path, team, file)
        if os.path.exists(match_file):
            df = pd.read_csv(match_file)
            df["Team"] = team
            df["Season"] = season
            all_matches.append(df)

team_matches_df = pd.concat(all_matches, ignore_index=True)

def is_played(result):
    if pd.isna(result):
        return False
    result_str = str(result)
    # Match if it contains digits and a dash between them
    return True

team_matches_df = team_matches_df[team_matches_df["Comp"].str.strip().str.lower() == "premier league"].copy()

team_matches_df['Played'] = team_matches_df['Result'].apply(is_played)

matches_played = team_matches_df[team_matches_df['Played']].copy()
matches_upcoming = team_matches_df[~team_matches_df['Played']].copy()

played_path = r"C:\Users\admin\OneDrive - National University of Singapore\Football Dataset\matches.csv"
upcoming_path = r"C:\Users\admin\OneDrive - National University of Singapore\Football Dataset\matches_test.csv"

matches_played.drop(columns=['Played'], inplace=True)
matches_upcoming.drop(columns=['Played'], inplace=True)

matches_played.to_csv(played_path, index=False)
matches_upcoming.to_csv(upcoming_path, index=False)

print(f"✅ Played matches saved: {len(matches_played)}")
print(f"🧩 Upcoming matches saved: {len(matches_upcoming)}")



