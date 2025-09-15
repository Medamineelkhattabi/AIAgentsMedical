# interface/cli.py
"""
Command-line interface for the Medical AI Agent.
"""
import argparse
import csv
from ai_agent.engine import MedicalAIAgent
from ai_agent.rules import rule_based_diagnosis
from ai_agent.recommendations import get_recommendations

def main():
    parser = argparse.ArgumentParser(description='Medical Diagnostics AI Agent')
    parser.add_argument('--fever', type=int, choices=[0,1], help='Fever (1=yes, 0=no)')
    parser.add_argument('--cough', type=int, choices=[0,1], help='Cough (1=yes, 0=no)')
    parser.add_argument('--chest_pain', type=int, choices=[0,1], help='Chest pain (1=yes, 0=no)')
    parser.add_argument('--shortness_of_breath', type=int, choices=[0,1], help='Shortness of breath (1=yes, 0=no)')
    parser.add_argument('--model', type=str, help='Path to trained model', default=None)
    args = parser.parse_args()

    symptoms = {
        'fever': args.fever,
        'cough': args.cough,
        'chest_pain': args.chest_pain,
        'shortness_of_breath': args.shortness_of_breath
    }

    # Rule-based diagnosis
    diagnosis = rule_based_diagnosis(symptoms)
    print(f'Rule-based diagnosis: {diagnosis}')

    # ML-based diagnosis (if model provided)
    if args.model:
        agent = MedicalAIAgent(args.model)
        ml_diag = agent.predict(symptoms)
        print(f'ML-based diagnosis: {ml_diag}')
        diagnosis = ml_diag

    # Recommendations
    print('Recommendation:', get_recommendations(diagnosis))

if __name__ == '__main__':
    main()
