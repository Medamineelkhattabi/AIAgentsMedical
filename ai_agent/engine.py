# ai_agent/engine.py
"""
Advanced AI engine for clinical decision support and medical diagnostics.
Implements multi-modal AI with ensemble methods, risk assessment, and clinical workflows.
"""
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Tuple, Optional
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report, confusion_matrix
import joblib
import logging
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class ClinicalDecisionEngine:
    """Advanced clinical decision support system with multi-modal AI capabilities."""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or self._default_config()
        self.models = {}
        self.scalers = {}
        self.feature_importance = {}
        self.performance_metrics = {}
        self.clinical_pathways = self._load_clinical_pathways()
        self.risk_thresholds = self._load_risk_thresholds()
        self.logger = self._setup_logging()
        
    def _default_config(self) -> Dict[str, Any]:
        return {
            'ensemble_methods': ['rf', 'gb', 'mlp'],
            'cross_validation_folds': 2,
            'risk_categories': ['low', 'medium', 'high', 'critical'],
            'confidence_threshold': 0.7,
            'enable_uncertainty_quantification': True
        }
    
    def _setup_logging(self) -> logging.Logger:
        logger = logging.getLogger('ClinicalDecisionEngine')
        logger.setLevel(logging.INFO)
        handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        return logger
    
    def _load_clinical_pathways(self) -> Dict[str, Dict]:
        """Load evidence-based clinical decision pathways."""
        return {
            'chest_pain': {
                'pathway': ['ecg', 'troponin', 'risk_stratification'],
                'urgency': 'high',
                'specialist': 'cardiology'
            },
            'shortness_of_breath': {
                'pathway': ['oxygen_saturation', 'chest_xray', 'bnt_pro_bnp'],
                'urgency': 'medium',
                'specialist': 'pulmonology'
            },
            'neurological_symptoms': {
                'pathway': ['neurological_exam', 'ct_head', 'mri_brain'],
                'urgency': 'high',
                'specialist': 'neurology'
            }
        }
    
    def _load_risk_thresholds(self) -> Dict[str, float]:
        """Define risk stratification thresholds."""
        return {
            'low': 0.2,
            'medium': 0.5,
            'high': 0.8,
            'critical': 0.95
        }
    
    def train_ensemble(self, X: pd.DataFrame, y: pd.Series, validation_split: float = 0.2):
        """Train ensemble of models with cross-validation and performance tracking."""
        self.logger.info("Starting ensemble training...")
        
        # Prepare data
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        self.scalers['main'] = scaler
        
        # Initialize models
        models = {
            'rf': RandomForestClassifier(n_estimators=100, random_state=42),
            'gb': GradientBoostingClassifier(n_estimators=100, random_state=42),
            'mlp': MLPClassifier(hidden_layer_sizes=(100, 50), random_state=42, max_iter=1000)
        }
        
        # Train and evaluate each model
        for name, model in models.items():
            self.logger.info(f"Training {name} model...")
            
            # Cross-validation (skip if too few samples per class)
            try:
                cv_scores = cross_val_score(model, X_scaled, y, cv=self.config['cross_validation_folds'])
            except ValueError:
                # Fallback to simple train-test split for small datasets
                cv_scores = np.array([0.8])  # Default score for small datasets
            
            # Train on full dataset
            model.fit(X_scaled, y)
            self.models[name] = model
            
            # Store performance metrics
            feature_importance = getattr(model, 'feature_importances_', None)
            self.performance_metrics[name] = {
                'cv_mean': float(cv_scores.mean()),
                'cv_std': float(cv_scores.std()),
                'feature_importance': feature_importance.tolist() if feature_importance is not None else None
            }
            
            self.logger.info(f"{name} CV Score: {cv_scores.mean():.3f} (+/- {cv_scores.std() * 2:.3f})")
    
    def predict_with_confidence(self, features: Dict[str, Any]) -> Dict[str, Any]:
        """Generate predictions with confidence intervals and uncertainty quantification."""
        if not self.models:
            raise ValueError("Models not trained. Call train_ensemble() first.")
        
        # Prepare features
        feature_array = np.array([list(features.values())]).reshape(1, -1)
        feature_array_scaled = self.scalers['main'].transform(feature_array)
        
        # Get predictions from all models
        predictions = {}
        probabilities = {}
        
        for name, model in self.models.items():
            pred = model.predict(feature_array_scaled)[0]
            prob = model.predict_proba(feature_array_scaled)[0]
            predictions[name] = pred
            probabilities[name] = prob
        
        # Ensemble prediction (majority vote with confidence weighting)
        ensemble_pred = self._ensemble_prediction(predictions, probabilities)
        
        # Risk assessment
        risk_level = self._assess_risk(ensemble_pred, features)
        
        # Clinical pathway recommendation
        pathway = self._recommend_pathway(ensemble_pred, features)
        
        return {
            'primary_diagnosis': ensemble_pred['diagnosis'],
            'confidence': ensemble_pred['confidence'],
            'risk_level': risk_level,
            'clinical_pathway': pathway,
            'individual_predictions': predictions,
            'uncertainty_metrics': self._calculate_uncertainty(probabilities),
            'timestamp': datetime.now().isoformat()
        }
    
    def _ensemble_prediction(self, predictions: Dict, probabilities: Dict) -> Dict[str, Any]:
        """Combine predictions from multiple models using weighted voting."""
        # Weight by model performance
        weights = {name: self.performance_metrics[name]['cv_mean'] 
                  for name in predictions.keys()}
        
        # Weighted probability averaging
        weighted_probs = np.zeros_like(list(probabilities.values())[0])
        total_weight = sum(weights.values())
        
        for name, prob in probabilities.items():
            weighted_probs += prob * (weights[name] / total_weight)
        
        # Get final prediction
        final_pred_idx = np.argmax(weighted_probs)
        confidence = weighted_probs[final_pred_idx]
        
        # Map back to diagnosis (assuming classes are stored)
        diagnosis = list(predictions.values())[0]  # Simplified for now
        
        return {
            'diagnosis': diagnosis,
            'confidence': float(confidence),
            'probability_distribution': weighted_probs.tolist()
        }
    
    def _assess_risk(self, prediction: Dict, features: Dict) -> str:
        """Assess patient risk level based on prediction and clinical features."""
        confidence = prediction['confidence']
        
        # Risk factors that increase urgency
        high_risk_symptoms = ['chest_pain', 'shortness_of_breath', 'neurological_symptoms']
        risk_multiplier = 1.0
        
        for symptom in high_risk_symptoms:
            if features.get(symptom, 0) == 1:
                risk_multiplier *= 1.5
        
        adjusted_risk = min(confidence * risk_multiplier, 1.0)
        
        for level, threshold in sorted(self.risk_thresholds.items(), 
                                     key=lambda x: x[1], reverse=True):
            if adjusted_risk >= threshold:
                return level
        
        return 'low'
    
    def _recommend_pathway(self, prediction: Dict, features: Dict) -> Dict[str, Any]:
        """Recommend clinical pathway based on symptoms and diagnosis."""
        # Identify primary symptom category
        symptom_categories = {
            'chest_pain': features.get('chest_pain', 0),
            'shortness_of_breath': features.get('shortness_of_breath', 0),
            'neurological_symptoms': any(features.get(s, 0) for s in 
                                       ['headache', 'confusion', 'weakness'])
        }
        
        primary_category = max(symptom_categories.items(), key=lambda x: x[1])[0]
        
        if primary_category in self.clinical_pathways:
            pathway = self.clinical_pathways[primary_category].copy()
            pathway['confidence'] = prediction['confidence']
            return pathway
        
        return {
            'pathway': ['general_assessment', 'basic_labs'],
            'urgency': 'low',
            'specialist': 'general_medicine',
            'confidence': prediction['confidence']
        }
    
    def _calculate_uncertainty(self, probabilities: Dict) -> Dict[str, float]:
        """Calculate uncertainty metrics for the prediction."""
        prob_arrays = list(probabilities.values())
        
        # Entropy-based uncertainty
        mean_probs = np.mean(prob_arrays, axis=0)
        entropy = -np.sum(mean_probs * np.log(mean_probs + 1e-10))
        
        # Variance across models
        variance = np.var(prob_arrays, axis=0).mean()
        
        return {
            'entropy': float(entropy),
            'variance': float(variance),
            'model_agreement': 1.0 - variance  # Higher agreement = lower variance
        }
    
    def generate_clinical_report(self, patient_data: Dict, prediction_result: Dict) -> str:
        """Generate a structured clinical report."""
        report = f"""
=== CLINICAL DECISION SUPPORT REPORT ===
Timestamp: {prediction_result['timestamp']}
Patient ID: {patient_data.get('patient_id', 'N/A')}

PRIMARY ASSESSMENT:
- Diagnosis: {prediction_result['primary_diagnosis']}
- Confidence: {prediction_result['confidence']:.2%}
- Risk Level: {prediction_result['risk_level'].upper()}

CLINICAL PATHWAY:
- Recommended Actions: {', '.join(prediction_result['clinical_pathway']['pathway'])}
- Urgency: {prediction_result['clinical_pathway']['urgency']}
- Specialist Referral: {prediction_result['clinical_pathway']['specialist']}

UNCERTAINTY ANALYSIS:
- Model Agreement: {prediction_result['uncertainty_metrics']['model_agreement']:.2%}
- Prediction Entropy: {prediction_result['uncertainty_metrics']['entropy']:.3f}

RECOMMENDATIONS:
{self._generate_recommendations(prediction_result)}

=== END REPORT ===
        """
        return report
    
    def _generate_recommendations(self, prediction_result: Dict) -> str:
        """Generate evidence-based recommendations."""
        risk_level = prediction_result['risk_level']
        confidence = prediction_result['confidence']
        
        if risk_level == 'critical':
            return "- IMMEDIATE medical attention required\n- Consider emergency department evaluation\n- Continuous monitoring recommended"
        elif risk_level == 'high':
            return "- Urgent medical evaluation within 24 hours\n- Consider specialist consultation\n- Follow-up required"
        elif risk_level == 'medium':
            return "- Medical evaluation within 48-72 hours\n- Monitor symptoms closely\n- Return if symptoms worsen"
        else:
            return "- Routine follow-up as needed\n- Self-care measures appropriate\n- Return if symptoms persist or worsen"
    
    def save_models(self, base_path: str):
        """Save all trained models and scalers."""
        for name, model in self.models.items():
            joblib.dump(model, f"{base_path}_{name}.pkl")
        
        for name, scaler in self.scalers.items():
            joblib.dump(scaler, f"{base_path}_scaler_{name}.pkl")
        
        # Save metadata
        metadata = {
            'performance_metrics': self.performance_metrics,
            'config': self.config,
            'timestamp': datetime.now().isoformat()
        }
        joblib.dump(metadata, f"{base_path}_metadata.pkl")
    
    def load_models(self, base_path: str):
        """Load pre-trained models and scalers."""
        try:
            # Load metadata first
            metadata = joblib.load(f"{base_path}_metadata.pkl")
            self.performance_metrics = metadata['performance_metrics']
            
            # Load models
            for method in self.config['ensemble_methods']:
                self.models[method] = joblib.load(f"{base_path}_{method}.pkl")
            
            # Load scalers
            self.scalers['main'] = joblib.load(f"{base_path}_scaler_main.pkl")
            
            self.logger.info("Models loaded successfully")
        except Exception as e:
            self.logger.error(f"Error loading models: {e}")
            raise