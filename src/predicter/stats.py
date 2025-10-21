import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
from math import pi

# --------------------------------
# 🔹 Generate Fake Player Data
# --------------------------------
def generate_fake_player_data():
    np.random.seed(42)

    players = [
        "Haaland", "Salah", "De Bruyne", "Saka", "Fernandes",
        "Van Dijk", "Rashford", "Foden", "Rice", "Son",
        "Alisson", "Ederson", "Odegaard", "Silva", "Walker",
        "Martinez", "Casemiro", "Robertson", "Kane", "Palmer"
    ]
    positions = [
        "ST", "RW", "CM", "RW", "CAM",
        "CB", "LW", "LW", "CDM", "LW",
        "GK", "GK", "CAM", "CM", "RB",
        "GK", "CDM", "LB", "ST", "RW"
    ]
    teams = [
        "Man City", "Liverpool", "Man City", "Arsenal", "Man United",
        "Liverpool", "Man United", "Man City", "Arsenal", "Tottenham",
        "Liverpool", "Man City", "Arsenal", "Man City", "Man City",
        "Man United", "Man United", "Liverpool", "Tottenham", "Chelsea"
    ]

    df = pd.DataFrame({
        "Player": players,
        "Team": teams,
        "Position": positions,
        "Goals": np.random.randint(0, 25, len(players)),
        "Assists": np.random.randint(0, 15, len(players)),
        "ShotsOnTarget": np.random.randint(10, 60, len(players)),
        "PassAccuracy": np.random.randint(70, 95, len(players)),
        "Tackles": np.random.randint(0, 60, len(players)),
        "Interceptions": np.random.randint(0, 50, len(players)),
        "Saves": np.random.randint(0, 100, len(players)),  # for goalkeepers
    })

    return df


# --------------------------------
# 🔹 Compute Performance Score
# --------------------------------
def compute_performance_score(row):
    pos = row["Position"]

    if pos in ["ST", "RW", "LW", "CAM"]:  # Attackers
        return row["Goals"] * 4 + row["Assists"] * 3 + row["ShotsOnTarget"] * 0.6 + row["PassAccuracy"] * 0.2
    elif pos in ["CM", "CDM"]:  # Midfielders
        return row["Goals"] * 2 + row["Assists"] * 2 + row["PassAccuracy"] * 0.3 + row["Tackles"] * 0.5
    elif pos in ["CB", "LB", "RB"]:  # Defenders
        return row["Tackles"] * 0.8 + row["Interceptions"] * 0.7 + row["PassAccuracy"] * 0.3
    elif pos == "GK":  # Goalkeepers
        return row["Saves"] * 0.7 + (100 - row["Goals"]) * 0.2
    else:
        return row["Goals"] + row["Assists"]


# --------------------------------
# 🔹 Get Best Players (by Position)
# --------------------------------
def get_best_players(players_df=None, position_filter="All"):
    if players_df is None:
        players_df = generate_fake_player_data()

    players_df["PerformanceScore"] = players_df.apply(compute_performance_score, axis=1)

    if position_filter != "All":
        filtered_df = players_df[players_df["Position"].str.startswith(position_filter)]
    else:
        filtered_df = players_df

    top_players = filtered_df.sort_values(by="PerformanceScore", ascending=False).head(5)
    return top_players[["Player", "Team", "Position", "Goals", "Assists", "PassAccuracy", "PerformanceScore"]]


# --------------------------------
# 🔹 Player Comparison Visualization
# --------------------------------
def compare_players(players_df, p1, p2):
    metrics = ["Goals", "Assists", "ShotsOnTarget", "PassAccuracy", "Tackles", "Interceptions"]
    p1_stats = players_df[players_df["Player"] == p1][metrics].iloc[0]
    p2_stats = players_df[players_df["Player"] == p2][metrics].iloc[0]

    comparison = pd.DataFrame({
        "Metric": metrics,
        p1: p1_stats.values,
        p2: p2_stats.values
    })

    st.subheader("📊 Comparison Table")
    st.dataframe(comparison)

    # --- Bar Chart ---
    st.subheader("📈 Bar Chart Comparison")
    fig, ax = plt.subplots()
    comparison.plot(x="Metric", kind="bar", ax=ax, rot=0)
    ax.set_ylabel("Value")
    ax.set_title(f"{p1} vs {p2}")
    st.pyplot(fig)

    # --- Radar Chart ---
    st.subheader("🕸️ Radar Chart Comparison")

    categories = metrics
    N = len(categories)
    values1 = p1_stats.values.tolist()
    values2 = p2_stats.values.tolist()

    values1 += values1[:1]
    values2 += values2[:1]
    angles = [n / float(N) * 2 * pi for n in range(N)]
    angles += angles[:1]

    fig, ax = plt.subplots(subplot_kw=dict(polar=True))
    ax.set_theta_offset(pi / 2)
    ax.set_theta_direction(-1)
    plt.xticks(angles[:-1], categories)

    ax.plot(angles, values1, linewidth=2, linestyle='solid', label=p1)
    ax.fill(angles, values1, alpha=0.25)
    ax.plot(angles, values2, linewidth=2, linestyle='solid', label=p2)
    ax.fill(angles, values2, alpha=0.25)

    plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
    st.pyplot(fig)
