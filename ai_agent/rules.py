# ai_agent/rules.py
"""
Rule-based logic for medical diagnostics.
"""
def rule_based_diagnosis(symptoms: dict) -> str:
    # Expanded rules for more real-world diseases
    if symptoms.get('fever') and symptoms.get('cough') and symptoms.get('shortness_of_breath'):
        return 'COVID-19'
    if symptoms.get('fever') and symptoms.get('rash'):
        return 'Measles'
    if symptoms.get('headache') and symptoms.get('nausea') and symptoms.get('sensitivity_to_light'):
        return 'Migraine'
    if symptoms.get('chest_pain') and symptoms.get('shortness_of_breath'):
        return 'Possible Heart Issue'
    if symptoms.get('abdominal_pain') and symptoms.get('diarrhea'):
        return 'Gastroenteritis'
    if symptoms.get('joint_pain') and symptoms.get('swelling'):
        return 'Arthritis'
    if symptoms.get('fever') and symptoms.get('cough'):
        return 'Flu'
    if symptoms.get('fever') and symptoms.get('headache') and symptoms.get('nausea'):
        return 'Dengue'
    if symptoms.get('fever') and symptoms.get('joint_pain') and symptoms.get('rash'):
        return 'Chikungunya'
    if symptoms.get('abdominal_pain') and symptoms.get('nausea') and symptoms.get('diarrhea'):
        return 'Food Poisoning'
    if symptoms.get('headache') and symptoms.get('fever') and symptoms.get('stiff_neck'):
        return 'Meningitis'
    if symptoms.get('cough') and symptoms.get('chest_pain') and symptoms.get('fever'):
        return 'Pneumonia'
    if symptoms.get('fever') and symptoms.get('sore_throat'):
        return 'Strep Throat'
    return 'Unknown'
