# ai_agent/recommendations.py
"""
Recommendation engine for follow-up actions.
"""
def get_recommendations(diagnosis: str) -> str:
    if diagnosis == 'Flu':
        return 'Rest, hydration, and consult a doctor if symptoms persist.'
    if diagnosis == 'Possible Heart Issue':
        return 'Seek immediate medical attention.'
    return 'Consult a healthcare professional for further evaluation.'
