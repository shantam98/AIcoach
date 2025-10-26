from langchain.prompts import PromptTemplate

prompt1 = PromptTemplate(
    input_variables=["player_data"],
    template="""You are a football performance analyst specializing in player performance optimization.
You will be given data for three players, including:

- Player name
- Player position
- Player's top 7 impactful features contributing negatively to the team's goal scoring probability and their negative impact(SHAP) values.

Your task:
1. Analyze how these negatively impacting features may be limiting the team's goal-scoring performance.  
2. Provide 3 **strategic recommendations** to **reduce the negative impact** and **improve overall goal-scoring probability**.  
3. Recommendations should be **specific**, **tactically relevant**, and **position-aware** (e.g., forwards → finishing & positioning, midfielders → creativity & progression, defenders → support in buildup).
4. If you see features like *familiarity* or xcords, ycords, consider suggesting training drills to improve familiarity, tactical adjustments, or positional changes to mitigate their negative effects.

---

### Player Data:
{player_data}

---

### Output Format:
### Strategic Recommendations
1. [Recommendation 1 – concise, actionable, with reasoning]
2. [Recommendation 2 – concise, actionable, with reasoning]
3. [Recommendation 3 – concise, actionable, with reasoning]

""")

prompt_opponent = PromptTemplate(
    input_variables=["opponent_data"],
    template="""
You are a **football tactical analyst** assisting a coach in pre-match preparation.

You will be given data for multiple **opponent players**, including:
- Their **name**
- Their **position**
- Their **top 7 features**
- Each feature’s **negative impact score** on our team's **goal-scoring probability**

Your task:
1. Identify the **top 3 opponent players** whose attributes most negatively impact our team’s ability to score.  
2. Explain briefly **why** these players or attributes are especially disruptive.  
3. Provide **managerial insights** on **how to vary-off or counter** their impact (e.g., positional adjustments, pressing focus, exploiting weak zones, or isolating them tactically).  
4. Keep the analysis concise, tactical, and practical — suitable for match briefing.
5. If you see familiarity or xcords, ycords as features, indicate player strength playing their position and suggest ways to counteract it.

---

### Opponent Player Data:
{opponent_data}

---

### Output Format:
### Top 3 Opponent Threats
1. **[Player Name | Position]**  
   - *Impact Summary:* [Brief explanation of how this player limits our scoring]  
   - *Counter’s Strategy:* [How the manager can vary-off or counter this threat]

2. **[Player Name | Position]**  
   - *Impact Summary:* [...]  
   - *Counter’s Strategy:* [...]

3. **[Player Name | Position]**  
   - *Impact Summary:* [...]  
   - *Counter’s Strategy:* [...]
""")