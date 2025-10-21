def get_team_recommendations(team, opponent=None):
    if opponent:
        return [
            f"{team} should exploit {opponent}'s weak left flank.",
            f"Focus on high pressing against {opponent}.",
        ]
    else:
        return [
            "Improve finishing inside the box.",
            "Increase passing accuracy in midfield.",
            "Rotate defenders to reduce fatigue.",
        ]
