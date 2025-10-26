from turtle import color
import streamlit as st
import pandas as pd
import numpy as np
import base64

from PIL import Image
from datetime import datetime, timedelta

# -------------------------------------------------
# 🖼️ Function to Add Background Image
# -------------------------------------------------

def add_bg_from_local(image_file):
    with open(image_file, "rb") as f:
        data = f.read()
    encoded = base64.b64encode(data).decode()
    page_bg = page_bg = f"""
    <style>
    [data-testid="stAppViewContainer"] {{
        background-image: url("data:image/png;base64,{encoded}");
        background-size: cover;
        background-position: center;
        background-repeat: no-repeat;
        background-attachment: fixed;
    }}
    
    [data-testid="stHeader"] {{
        background: rgba(0, 0, 0, 0.8);
    }}
    
    [data-testid="stSidebar"] {{
        background-color: rgba(0, 0, 0, 0.8);
    }}
    
    /* Dark containers for text elements */
    .stDataFrame, .stMarkdown, .stRadio, .stExpander {{
        background-color: rgba(0, 0, 0, 0.8);
        border-radius: 10px;
        padding: 10px;
        color: #ffffff;
    }}
    
    /* Make sure all text in these containers is white */
    .stDataFrame *, .stMarkdown *, .stRadio *, .stExpander * {{
        color: #ffffff !important;
    }}
    
    /* Expander header styling */
    .stExpander > div > div > button {{
        background-color: rgba(0, 0, 0, 0.8) !important;
        color: #ffffff !important;
        border-radius: 10px;
    }}
    
    /* Expander content area */
    .stExpander > div > div > div {{
        background-color: rgba(0, 0, 0, 0.8) !important;
        border-radius: 0 0 10px 10px;
    }}
    
    /* Selectbox container - light background */
    .stSelectbox {{
        background-color: transparent;
        border-radius: 10px;
        padding: 5px;
    }}
    
    /* Selectbox label - dark text */
    .stSelectbox > label {{
        color: #111111 !important;
        font-weight: 500;
    }}
    
    /* Selectbox input field - light background, dark text */
    div[data-baseweb="select"] > div {{
        background-color: #f5f5f5 !important;
        color: #111111 !important;
        border: 1px solid #000000;
    }}
    
    /* Selected value text in dropdown */
    div[data-baseweb="select"] * {{
        color: #111111 !important;
    }}
    
    /* Dropdown menu popover - white background with black border */
    div[data-baseweb="popover"] {{
        background-color: #ffffff !important;
        border: 2px solid #000000 !important;
        border-radius: 5px;
    }}
    
    /* Inner popover content */
    div[data-baseweb="popover"] > div {{
        background-color: #ffffff !important;
        border-radius: 5px;
    }}
    
    /* Dropdown menu items - white background, dark text */
    ul[role="listbox"] {{
        background-color: #ffffff !important;
        border: none !important;
    }}
    
    ul[role="listbox"] li {{
        background-color: #ffffff !important;
        color: #111111 !important;
    }}
    
    /* Dropdown menu items on hover */
    ul[role="listbox"] li:hover {{
        background-color: #e0e0e0 !important;
        color: #111111 !important;
    }}
    
    /* All text inside dropdown menu */
    div[data-baseweb="popover"] * {{
        color: #111111 !important;
        background-color: transparent !important;
    }}
    
    /* Override any nested divs that might have dark backgrounds */
    div[data-baseweb="popover"] div {{
        background-color: #ffffff !important;
    }}
    
    /* Button styling */
    .stButton > button {{
        background-color: #f5f5f5;
        color: #111111;
        border-radius: 10px;
        border: 1px solid #cccccc;
        font-weight: 500;
    }}
    
    .stButton > button:hover {{
        background-color: #e0e0e0;
        border: 1px solid #999999;
    }}
    
    /* Input fields */
    .stTextInput > div > div > input,
    .stNumberInput > div > div > input,
    .stTextArea > div > div > textarea {{
        background-color: rgba(0, 0, 0, 0.8) !important;
        color: #ffffff !important;
        border: 1px solid #cccccc;
        border-radius: 5px;
    }}
    
    /* Labels */
    .stTextInput > label,
    .stNumberInput > label,
    .stTextArea > label {{
        color: #111111 !important;
        font-weight: 500;
    }}
    
    /* Radio button text */
    .stRadio > label {{
        color: #111111 !important;
        font-weight: 500;
    }}
    
    .stRadio > div > label > div {{
        color: #ffffff !important;
    }}
    
    /* Ensure main text content is dark for visibility */
    h1, h2, h3, h4, h5, h6, p {{
        color: #111111 !important;
    }}
    
    /* For elements that need dark background, keep text white */
    [data-testid="stSidebar"] * {{
        color: #ffffff !important;
    }}
    </style>
    """
    st.markdown(page_bg, unsafe_allow_html=True)

# -----------------------------
# ⚙️ Streamlit Setup
# -----------------------------
st.set_page_config(
    page_title="Football Manager Dashboard",
    page_icon="⚽",
    layout="wide"
)

# ✅ Add background image
add_bg_from_local("../data/EPL-white.jpg")

# -----------------------------
# 📍 Sidebar Navigation
# -----------------------------
st.sidebar.title("⚽ Football IQ App")
page = st.sidebar.radio("Navigate", ["🏠 Home", "📈 Team Stats"])

# -----------------------------
# 📊 Sample Team Data
# -----------------------------

# Get current date
current_date = datetime.now()

# -----------------------------
# Helper Functions
# -----------------------------

def load_prediction_data(fpath = "../data/matches_test_predicted_with_player_elo.csv"):
    df = pd.read_csv(fpath)
    # convert data column to datetime
    df['Date'] = pd.to_datetime(df['Date'])
    return df

def load_recommendation_data(fpath = "../data/llm_generation_recommendations.csv"):
    df = pd.read_csv(fpath)
    return df

predictdf = load_prediction_data()
recommenddf = load_recommendation_data()

teams = sorted(predictdf['Home'].unique().tolist())

def predict_score(team1, team2, date,homeaway, df = predictdf):
    """Providing score prediction"""
    if homeaway == "Home":
        team_home = team1
        team_away = team2
    else:
        team_home = team2
        team_away = team1
        # Filter df for the match
    print(f"Predicting score for: {team_home} vs {team_away} on {date} ({homeaway})")
    datastr = date.strftime('%Y-%m-%d')
    match = df[(df['Home'] == team_home) & (df['Away'] == team_away) & (df['Date'] == datastr)]
    homegoals = match['pred_home_goals'].values[0]
    awaygoals = match['pred_away_goals'].values[0]
    return f"{homegoals}-{awaygoals}"

def get_team_recommendations(team, opponent, date, homeaway, df=recommenddf):
    """Get recoommendations for a specific team and opponent"""
    if homeaway == "Home":
        team_col = "team"
        opponent_col = "opponent_team"
    else:
        team_col = "opponent_team"
        opponent_col = "team"
    match = df[(df[team_col] == team) & (df[opponent_col] == opponent) & (df['date'] == date.strftime('%Y-%m-%d'))]
    if match.empty:
        return []
    player_recs = match['player_recommendations'].values[0]
    opponent = match['opponent_recommendations'].values[0]
    return  [player_recs] + [opponent]

def generate_team_player_data(team, fpath = "../data/player_recommendation_summary/"):
    """Get player data for a specific team"""
    df = pd.read_csv(fpath + f"{team}_player_recommendation_summary.csv")

    return df

def load_formation_image(fpath = "../data/"):
    """Load best players formation image for the selected team"""
    image_path = f"{fpath}dream11.png"  # or whatever naming convention you use
    
    try:
        img = Image.open(image_path)
        return img, None
    except FileNotFoundError:
        return None, f"⚠️ Formation image not found for {team}. Expected file: {image_path}"
    except Exception as e:
        return None, f"❌ Error loading image: {str(e)}"


def load_cluster_image(team, fpath = "../data/team_ClusterLabel/"):
    """Load cluster visualization image for the selected team"""
    image_path = f"{fpath}{team}_ClusterLabel.png"  # or f"clusters/{team}.jpg" if in a subfolder
    
    try:
        img = Image.open(image_path)
        return img, None
    except FileNotFoundError:
        return None, f"⚠️ Cluster visualization image not found for {team}. Expected file: {image_path}"
    except Exception as e:
        return None, f"❌ Error loading image: {str(e)}"

def get_upcoming_matches(team, start_date, df=predictdf, days=14):
    """Get matches for the next 'days' period from df including today"""
    end_date = start_date + timedelta(days=days)
    # print('start_date:', start_date.date())
    # print('end_date:', end_date.date())
    team_matches = df[((df['Home'] == team) | (df['Away'] == team)) & (df['Date'] >= start_date.strftime('%Y-%m-%d')) & (df['Date'] <= end_date.strftime('%Y-%m-%d'))]
    # Generate random dates within the next two weeks
    match_dates = team_matches['Date'].dt.date.tolist()
    opponents = []
    homeaway = []
    for _, row in team_matches.iterrows():
        if row['Home'] == team:
            opponents.append(row['Away'])
            homeaway.append("Home")
        else:
            opponents.append(row['Home'])
            homeaway.append("Away")

    # Sort dates
    match_dates.sort()
    
    # Create matches dataframe
    matches = pd.DataFrame({
        "Opponent": opponents,
        "Date": match_dates,
        "homeaway": homeaway
    })
    
    return matches

# -----------------------------
# 1️⃣ HOME PAGE
# -----------------------------
if page == "🏠 Home":
    # Display current date at the top
    st.markdown(f"###  Welcome to Football IQ")
    st.markdown(f"📅 **{current_date.strftime('%A, %B %d, %Y')}**")
    
    st.title("🏟️ Match Predictions & Insights")
    team = st.selectbox("Select your team", teams)
    
    # Calculate end date (2 weeks from now)
    end_date = current_date + timedelta(days=14)
    
    st.markdown(f"### Upcoming Matches for **{team}**")
    st.markdown(f"*Showing matches from {current_date.strftime('%b %d')} to {end_date.strftime('%b %d, %Y')}*")
    
    # Generate upcoming matches for the next 2 weeks
    matches = get_upcoming_matches(team, current_date, days=14)
    
    if len(matches) == 0:
        st.info(f"No matches scheduled for {team} in the next two weeks.")
    else:
        for _, row in matches.iterrows():
            pred_score = predict_score(team, row["Opponent"], row["Date"], row['homeaway'])
            # Calculate days until match
            days_until = (row["Date"] - current_date.date()).days
            
            if days_until == 0:
                date_display = "Today"
            elif days_until == 1:
                date_display = "Tomorrow"
            else:
                date_display = f"In {days_until} days"
            
            match_date_str = row["Date"].strftime('%A, %b %d, %Y')
            
            with st.expander(f"⚽ {team} vs {row['Opponent']} — {match_date_str} ({date_display}) — Predicted: {pred_score}"):
                recs = get_team_recommendations(team, row["Opponent"], row["Date"], row['homeaway'])
                st.markdown("#### 🎯 Match Insights & Recommendations")
                for r in recs:
                    st.write(f"- {r}")
                
                # Additional match info
                st.markdown(f"**Days until match:** {days_until}")

elif page == "📈 Team Stats":
    # Display current date at the top
    st.markdown(f"### 📅 Today's Date: **{current_date.strftime('%A, %B %d, %Y')}**")
    
    st.title("📊 Team Performance & Player Analysis")
    
    team = st.selectbox("Select Team", teams)
    
    st.markdown(f"### {team} - Seasonal Analysis")

    # Generate player data for the team
    df = generate_team_player_data(team)
    
    # # -----------------------------
    # # Section 1: Player Improvement Recommendations
    # # -----------------------------
    # st.markdown("---")
    st.markdown("## 🎯 Player Development Recommendations")
    # st.markdown("*Personalized training drills and improvement areas for each player*")
    
    df['training_recommendations']=df['training_recommendations'].apply(lambda x: x.replace('*', '') if isinstance(x, str) else x)
    # # Display recommendations table
    display_df = df[['name', 'pred_rating', 'training_recommendations']].copy()
    st.dataframe(display_df, hide_index=True, use_container_width=True, height=400)
    
    # -----------------------------
    # Section 2: Clustering Analysis
    # -----------------------------
    st.markdown("## 🔬 Player Clustering Analysis")
    st.markdown("*Players grouped by performance characteristics and playing style*")
    
    
    # Load and display cluster image
    cluster_img, error_msg = load_cluster_image(team)
    if cluster_img:
    # Resize image to 80% of original size
        original_width, original_height = cluster_img.size
        new_width = int(original_width * 0.6)
        new_height = int(original_height * 0.6)
        resized_img = cluster_img.resize((new_width, new_height))

    if resized_img:
        st.image(resized_img, caption=f"{team} Player Clustering Visualization", use_container_width=True)
    else:
        st.error(error_msg)
        st.info("💡 **Tip**: Make sure your cluster images are named as `[TeamName]_clusters.jpg` and are in the same directory as your app, or update the `load_cluster_image()` function with the correct path.")


    st.markdown("## ⭐ Best X Formation")
    st.markdown("*Top performing players in optimal formation*")
    
    # Load and display formation image
    formation_img, error_msg_formation = load_formation_image()
    
    if formation_img:
        # Resize image to 80% of original size
        original_width, original_height = formation_img.size
        new_width = int(original_width * 0.5)
        new_height = int(original_height * 0.5)
        resized_formation_img = formation_img.resize((new_width, new_height), Image.LANCZOS)
        
        st.image(resized_formation_img, caption=f"Best X Formation", use_container_width=False)
    else:
        st.error(error_msg_formation)
        st.info("💡 **Tip**: Make sure your formation images are named as `[TeamName]_formation.jpg` and are in the same directory as your app.")