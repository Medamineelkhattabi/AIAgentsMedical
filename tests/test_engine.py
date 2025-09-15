# tests/test_engine.py
import csv
from ai_agent.engine import MedicalAIAgent

def test_train_and_predict():
    # Load sample data
    with open('data/sample_patients.csv') as f:
        reader = csv.DictReader(f)
        X, y = [], []
        for row in reader:
            X.append([
                int(row['fever']),
                int(row['cough']),
                int(row['chest_pain']),
                int(row['shortness_of_breath'])
            ])
            y.append(row['diagnosis'])
    agent = MedicalAIAgent()
    agent.train(X, y)
    pred = agent.predict({'fever':1,'cough':1,'chest_pain':0,'shortness_of_breath':0})
    assert pred == 'Flu'
