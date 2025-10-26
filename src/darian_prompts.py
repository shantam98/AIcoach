from langchain.prompts import PromptTemplate

prompt1 = PromptTemplate(
    input_variables=["player_data"],
    template="""You are a football performance analyst specializing in player performance optimization.
You will be given data for a player, including:

- Player name
- Player rating
- Player's top 3 impactful features contributing negatively to the his performance.

Your task:
1. Analyze how these negatively impacting features may be limiting the team's goal-scoring performance.  
2. Provide 3 **training recommendations** to **improve these features **.  
3. Recommendations should be **specific**, **tactically relevant**, and **position-aware** (e.g., forwards → finishing & positioning, midfielders → creativity & progression, defenders → support in buildup).

---

### Player Data:
{player_data}

---

### Output Format:
### training Recommendations
1. [Recommendation  – concise, actionable, with reasoning]

""")
