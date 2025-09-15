# ai_agent/triage_system.py
"""
Advanced triage and risk stratification system for emergency and clinical settings.
"""
import numpy as np
from typing import Dict, Any, List, Tuple
from datetime import datetime, timedelta
from enum import Enum

class TriageLevel(Enum):
    RESUSCITATION = 1  # Immediate
    EMERGENT = 2       # Within 15 minutes
    URGENT = 3         # Within 30 minutes
    LESS_URGENT = 4    # Within 60 minutes
    NON_URGENT = 5     # Within 120 minutes

class VitalSignsAnalyzer:
    """Analyze vital signs for triage decisions."""
    
    def __init__(self):
        self.normal_ranges = {
            'systolic_bp': (90, 140),
            'diastolic_bp': (60, 90),
            'heart_rate': (60, 100),
            'respiratory_rate': (12, 20),
            'temperature': (36.1, 37.2),
            'oxygen_saturation': (95, 100)
        }
    
    def assess_vitals(self, vitals: Dict[str, float]) -> Dict[str, Any]:
        """Assess vital signs and return severity scores."""
        scores = {}
        alerts = []
        
        for vital, value in vitals.items():
            if vital in self.normal_ranges:
                min_val, max_val = self.normal_ranges[vital]
                
                if value < min_val * 0.8 or value > max_val * 1.2:
                    scores[vital] = 3  # Critical
                    alerts.append(f"CRITICAL: {vital} = {value}")
                elif value < min_val or value > max_val:
                    scores[vital] = 2  # Abnormal
                    alerts.append(f"ABNORMAL: {vital} = {value}")
                else:
                    scores[vital] = 0  # Normal
        
        return {
            'scores': scores,
            'alerts': alerts,
            'total_score': sum(scores.values()),
            'severity': self._calculate_severity(sum(scores.values()))
        }
    
    def _calculate_severity(self, total_score: int) -> str:
        if total_score >= 9:
            return 'critical'
        elif total_score >= 6:
            return 'high'
        elif total_score >= 3:
            return 'moderate'
        else:
            return 'low'

class TriageSystem:
    """Advanced triage system with multi-factor assessment."""
    
    def __init__(self):
        self.vitals_analyzer = VitalSignsAnalyzer()
        self.pain_scale_weights = {
            0: 0, 1: 0.1, 2: 0.2, 3: 0.3, 4: 0.4,
            5: 0.5, 6: 0.6, 7: 0.7, 8: 0.8, 9: 0.9, 10: 1.0
        }
        
    def perform_triage(self, patient_data: Dict[str, Any]) -> Dict[str, Any]:
        """Perform comprehensive triage assessment."""
        
        # Assess vital signs
        vitals_assessment = self.vitals_analyzer.assess_vitals(
            patient_data.get('vitals', {})
        )
        
        # Assess chief complaint severity
        complaint_score = self._assess_chief_complaint(
            patient_data.get('chief_complaint', '')
        )
        
        # Assess pain level
        pain_score = self._assess_pain(patient_data.get('pain_level', 0))
        
        # Assess age and comorbidities
        demographic_score = self._assess_demographics(patient_data)
        
        # Calculate overall triage score
        triage_score = self._calculate_triage_score(
            vitals_assessment['total_score'],
            complaint_score,
            pain_score,
            demographic_score
        )
        
        # Determine triage level
        triage_level = self._determine_triage_level(triage_score)
        
        # Generate recommendations
        recommendations = self._generate_triage_recommendations(
            triage_level, vitals_assessment, patient_data
        )
        
        return {
            'triage_level': triage_level.name,
            'triage_score': triage_score,
            'estimated_wait_time': self._estimate_wait_time(triage_level),
            'vitals_assessment': vitals_assessment,
            'recommendations': recommendations,
            'alerts': vitals_assessment['alerts'],
            'timestamp': datetime.now().isoformat()
        }
    
    def _assess_chief_complaint(self, complaint: str) -> int:
        """Assess severity based on chief complaint."""
        high_acuity_keywords = [
            'chest pain', 'difficulty breathing', 'unconscious',
            'severe bleeding', 'stroke', 'heart attack', 'seizure'
        ]
        
        moderate_acuity_keywords = [
            'abdominal pain', 'fever', 'vomiting', 'headache',
            'injury', 'fall', 'allergic reaction'
        ]
        
        complaint_lower = complaint.lower()
        
        for keyword in high_acuity_keywords:
            if keyword in complaint_lower:
                return 3
        
        for keyword in moderate_acuity_keywords:
            if keyword in complaint_lower:
                return 2
        
        return 1
    
    def _assess_pain(self, pain_level: int) -> float:
        """Assess pain contribution to triage score."""
        return self.pain_scale_weights.get(pain_level, 0) * 2
    
    def _assess_demographics(self, patient_data: Dict) -> int:
        """Assess demographic risk factors."""
        score = 0
        age = patient_data.get('age', 0)
        
        # Age-based scoring
        if age >= 65 or age <= 2:
            score += 1
        
        # Comorbidities
        comorbidities = patient_data.get('comorbidities', [])
        high_risk_conditions = [
            'diabetes', 'heart_disease', 'copd', 'cancer',
            'immunocompromised', 'kidney_disease'
        ]
        
        for condition in comorbidities:
            if condition.lower() in high_risk_conditions:
                score += 1
        
        return min(score, 3)  # Cap at 3
    
    def _calculate_triage_score(self, vitals_score: int, complaint_score: int,
                               pain_score: float, demographic_score: int) -> float:
        """Calculate weighted triage score."""
        weights = {
            'vitals': 0.4,
            'complaint': 0.3,
            'pain': 0.2,
            'demographics': 0.1
        }
        
        return (
            vitals_score * weights['vitals'] +
            complaint_score * weights['complaint'] +
            pain_score * weights['pain'] +
            demographic_score * weights['demographics']
        )
    
    def _determine_triage_level(self, score: float) -> TriageLevel:
        """Determine triage level based on score."""
        if score >= 4.0:
            return TriageLevel.RESUSCITATION
        elif score >= 3.0:
            return TriageLevel.EMERGENT
        elif score >= 2.0:
            return TriageLevel.URGENT
        elif score >= 1.0:
            return TriageLevel.LESS_URGENT
        else:
            return TriageLevel.NON_URGENT
    
    def _estimate_wait_time(self, triage_level: TriageLevel) -> str:
        """Estimate wait time based on triage level."""
        wait_times = {
            TriageLevel.RESUSCITATION: "Immediate",
            TriageLevel.EMERGENT: "0-15 minutes",
            TriageLevel.URGENT: "15-30 minutes",
            TriageLevel.LESS_URGENT: "30-60 minutes",
            TriageLevel.NON_URGENT: "1-2 hours"
        }
        return wait_times[triage_level]
    
    def _generate_triage_recommendations(self, triage_level: TriageLevel,
                                       vitals_assessment: Dict,
                                       patient_data: Dict) -> List[str]:
        """Generate specific recommendations based on triage assessment."""
        recommendations = []
        
        if triage_level in [TriageLevel.RESUSCITATION, TriageLevel.EMERGENT]:
            recommendations.extend([
                "Immediate physician evaluation required",
                "Continuous monitoring",
                "IV access if not already established"
            ])
        
        if vitals_assessment['severity'] in ['critical', 'high']:
            recommendations.append("Frequent vital sign monitoring")
        
        if any('CRITICAL' in alert for alert in vitals_assessment['alerts']):
            recommendations.append("Consider ICU consultation")
        
        # Pain management
        pain_level = patient_data.get('pain_level', 0)
        if pain_level >= 7:
            recommendations.append("Pain management intervention needed")
        
        return recommendations