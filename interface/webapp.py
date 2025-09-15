# interface/webapp.py
"""
Flask web application for Medical Diagnostics AI Agent.
"""
from flask import Flask, render_template, request
from ai_agent.engine import MedicalAIAgent
from ai_agent.rules import rule_based_diagnosis
from ai_agent.recommendations import get_recommendations
import os

app = Flask(__name__)

# Load model if available
MODEL_PATH = os.path.join('models', 'medical_model.pkl')
agent = MedicalAIAgent(MODEL_PATH) if os.path.exists(MODEL_PATH) else None

SYMPTOMS = [
    'fever', 'cough', 'chest_pain', 'shortness_of_breath', 'rash',
    'headache', 'nausea', 'sensitivity_to_light', 'abdominal_pain',
    'diarrhea', 'joint_pain', 'swelling'
]

@app.route('/', methods=['GET', 'POST'])
def index():
    diagnosis = None
    recommendation = None
    selected = {s: 0 for s in SYMPTOMS}
    if request.method == 'POST':
        for s in SYMPTOMS:
            selected[s] = int(request.form.get(s, 0))
        # Rule-based diagnosis
        diagnosis = rule_based_diagnosis(selected)
        # ML-based diagnosis if model loaded
        if agent:
            try:
                ml_diag = agent.predict(selected)
                diagnosis = f"{diagnosis} / {ml_diag}"
            except Exception:
                pass
        recommendation = get_recommendations(diagnosis)
    return render_template('index.html', symptoms=SYMPTOMS, selected=selected, diagnosis=diagnosis, recommendation=recommendation)

if __name__ == '__main__':
    app.run(debug=True)
