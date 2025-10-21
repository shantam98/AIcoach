import streamlit as st
import pandas as pd
import numpy as np

from predicter.predict import predict_score
from predicter.recommend import get_team_recommendations
from predicter.stats import generate_fake_player_data, get_best_players, compare_players

# -----------------------------
# ⚙️ Streamlit Setup
# -----------------------------
st.set_page_config(
    page_title="Football Manager Dashboard",
    page_icon="⚽",
    layout="wide"
)

# -----------------------------
# 📍 Sidebar Navigation
# -----------------------------
st.sidebar.title("⚽ Football Manager App")
page = st.sidebar.radio("Navigate", ["🏠 Home", "📈 Team Insights", "⭐ Player Stats"])

# -----------------------------
# 📊 Sample Team Data
# -----------------------------
teams = ["Arsenal", "Liverpool", "Man City", "Chelsea", "Man United", "Tottenham"]

# -----------------------------
# 1️⃣ HOME PAGE
# -----------------------------
if page == "🏠 Home":
    st.title("🏟️ Match Predictions & Recommendations")

    team = st.selectbox("Select your team", teams)
    st.markdown(f"### Upcoming Matches for **{team}**")

    # Dummy upcoming matches
    matches = pd.DataFrame({
        "Opponent": np.random.choice([t for t in teams if t != team], 3),
        "Date": pd.date_range(start="2025-10-25", periods=3, freq="7D")
    })

    for _, row in matches.iterrows():
        pred_score = predict_score(team, row["Opponent"])
        with st.expander(f"{team} vs {row['Opponent']} — Predicted Score: {pred_score}"):
            recs = get_team_recommendations(team, row["Opponent"])
            st.markdown("#### Match Insights & Recommendations")
            for r in recs:
                st.write(f"- {r}")

# -----------------------------
# 2️⃣ TEAM INSIGHTS PAGE
# -----------------------------
elif page == "📈 Team Insights":
    st.title("📊 Team Performance Insights")

    team = st.selectbox("Select Team", teams)
    st.markdown(f"### {team} Season Overview")

    recs = get_team_recommendations(team)
    st.markdown("#### 🔍 Recommendations:")
    for r in recs:
        st.write(f"- {r}")

    # Dummy performance metrics
    st.markdown("#### ⚙️ Key Metrics")
    metrics_df = pd.DataFrame({
        "Metric": ["Possession %", "Pass Accuracy", "Shots on Target", "Goals per Match"],
        "Value": np.random.randint(55, 90, 4)
    })
    st.bar_chart(metrics_df.set_index("Metric"))

# -----------------------------
# 3️⃣ PLAYER STATS PAGE
# -----------------------------
elif page == "⭐ Player Stats":
    st.title("⭐ Best Players of the Season")

    df = generate_fake_player_data()

    position = st.selectbox("Filter by Position", ["All", "GK", "DEF", "MID", "ATT"])

    # Position mapping
    pos_map = {
        "All": "All",
        "GK": ["GK"],
        "DEF": ["CB", "LB", "RB"],
        "MID": ["CM", "CDM", "CAM"],
        "ATT": ["ST", "LW", "RW"]
    }

    if position == "All":
        top_players = get_best_players(df, "All")
    else:
        mask = df["Position"].isin(pos_map[position])
        top_players = get_best_players(df[mask], position)

    st.markdown("### 🏅 Top Players")
    st.dataframe(top_players, hide_index=True, use_container_width=True)

    st.divider()
    st.markdown("### ⚔️ Compare Players")

    default_best = top_players.iloc[0]["Player"]
    player1 = st.selectbox("Select Player 1", [default_best] + df["Player"].tolist())
    player2 = st.selectbox("Select Player 2", df["Player"].tolist(), index=2)

    if player1 == player2:
        st.warning("Please choose two different players to compare.")
    else:
        st.markdown(f"#### Comparing **{player1}** 🆚 **{player2}**")
        compare_players(df, player1, player2)

# -----------------------------
# 📄 Footer
# -----------------------------
st.sidebar.markdown("---")
st.sidebar.caption("Created by Football Manager AI 🧠⚽")
