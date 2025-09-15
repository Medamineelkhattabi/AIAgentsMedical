# interface/advanced_webapp.py
"""
Advanced web application with real-time monitoring, EHR integration, and clinical workflows.
"""
from flask import Flask, render_template, request, jsonify, session
from flask_socketio import SocketIO, emit
import json
import pandas as pd
from datetime import datetime
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ai_agent.engine import ClinicalDecisionEngine
from ai_agent.triage_system import TriageSystem
from ai_agent.ehr_integration import EHRIntegrationManager
from ai_agent.monitoring import ModelPerformanceMonitor, ClinicalOutcomeTracker

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-secret-key-here'
socketio = SocketIO(app, cors_allowed_origins="*")

# Initialize systems
clinical_engine = ClinicalDecisionEngine()
triage_system = TriageSystem()
performance_monitor = ModelPerformanceMonitor()
outcome_tracker = ClinicalOutcomeTracker()

# EHR integration (configure as needed)
ehr_config = {
    'fhir_enabled': False,  # Set to True when FHIR server is available
    'fhir_base_url': 'http://localhost:8080/fhir',
    'fhir_auth_token': None
}
ehr_manager = EHRIntegrationManager(ehr_config)

@app.route('/')
def index():
    """Main dashboard."""
    return render_template('advanced_dashboard.html')

@app.route('/api/diagnose', methods=['POST'])
def diagnose():
    """Advanced diagnosis endpoint with full clinical workflow."""
    try:
        data = request.json
        patient_id = data.get('patient_id', 'anonymous')
        
        # Extract symptoms and vitals
        symptoms = data.get('symptoms', {})
        vitals = data.get('vitals', {})
        patient_info = data.get('patient_info', {})
        
        # Ensure all expected features are present with default values
        expected_features = [
            'age', 'gender', 'fever', 'cough', 'chest_pain', 'shortness_of_breath',
            'headache', 'nausea', 'fatigue', 'joint_pain', 'rash', 'abdominal_pain',
            'diarrhea', 'vomiting', 'sore_throat', 'runny_nose', 'muscle_aches',
            'dizziness', 'weight_loss', 'night_sweats'
        ]
        
        # Create complete feature set
        complete_features = {}
        for feature in expected_features:
            if feature == 'age':
                complete_features[feature] = patient_info.get('age', 0)
            elif feature == 'gender':
                complete_features[feature] = 1  # Default to male
            else:
                complete_features[feature] = symptoms.get(feature, 0)
        
        # Perform triage assessment
        triage_data = {
            'vitals': vitals,
            'chief_complaint': data.get('chief_complaint', ''),
            'pain_level': data.get('pain_level', 0),
            'age': patient_info.get('age', 0),
            'comorbidities': patient_info.get('comorbidities', [])
        }
        
        triage_result = triage_system.perform_triage(triage_data)
        
        # Clinical decision support
        if clinical_engine.models:
            clinical_result = clinical_engine.predict_with_confidence(complete_features)
        else:
            # Fallback to rule-based if models not trained
            from ai_agent.rules import rule_based_diagnosis
            diagnosis = rule_based_diagnosis(symptoms)
            clinical_result = {
                'primary_diagnosis': diagnosis,
                'confidence': 0.8,
                'risk_level': 'medium',
                'clinical_pathway': {'pathway': ['general_assessment'], 'urgency': 'medium'},
                'timestamp': datetime.now().isoformat()
            }
        
        # Generate clinical report
        if clinical_engine.models:
            clinical_report = clinical_engine.generate_clinical_report(
                patient_info, clinical_result
            )
        else:
            clinical_report = f"Diagnosis: {clinical_result['primary_diagnosis']}"
        
        # Log prediction for monitoring
        from ai_agent.monitoring import PredictionLog
        prediction_log = PredictionLog(
            timestamp=datetime.now().isoformat(),
            patient_id=patient_id,
            model_version="v1.0",
            input_features=complete_features,
            prediction=clinical_result['primary_diagnosis'],
            confidence=clinical_result['confidence'],
            risk_level=clinical_result['risk_level']
        )
        performance_monitor.log_prediction(prediction_log)
        
        # Emit real-time update
        socketio.emit('new_diagnosis', {
            'patient_id': patient_id,
            'diagnosis': clinical_result['primary_diagnosis'],
            'triage_level': triage_result['triage_level'],
            'timestamp': datetime.now().isoformat()
        })
        
        return jsonify({
            'success': True,
            'clinical_assessment': clinical_result,
            'triage_assessment': triage_result,
            'clinical_report': clinical_report,
            'recommendations': _generate_integrated_recommendations(
                clinical_result, triage_result
            )
        })
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/train_models', methods=['POST'])
def train_models():
    """Train AI models with uploaded data."""
    try:
        # Load sample data for demonstration
        data_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'sample_patients.csv')
        
        if os.path.exists(data_path):
            df = pd.read_csv(data_path, comment='#')
            
            # Prepare features and labels
            feature_columns = [col for col in df.columns if col != 'diagnosis']
            X = df[feature_columns]
            y = df['diagnosis']
            
            # Train ensemble
            clinical_engine.train_ensemble(X, y)
            
            # Save models
            models_dir = os.path.join(os.path.dirname(__file__), '..', 'models')
            os.makedirs(models_dir, exist_ok=True)
            clinical_engine.save_models(os.path.join(models_dir, 'clinical_model'))
            
            return jsonify({
                'success': True,
                'message': 'Models trained successfully',
                'performance_metrics': clinical_engine.performance_metrics
            })
        else:
            return jsonify({
                'success': False,
                'error': 'Training data not found'
            }), 400
            
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/performance_metrics')
def get_performance_metrics():
    """Get model performance metrics."""
    try:
        metrics = performance_monitor.calculate_performance_metrics()
        drift_analysis = performance_monitor.detect_model_drift()
        
        return jsonify({
            'success': True,
            'metrics': {
                'accuracy': metrics.accuracy,
                'precision': metrics.precision,
                'recall': metrics.recall,
                'f1_score': metrics.f1_score,
                'prediction_count': metrics.prediction_count,
                'timestamp': metrics.timestamp
            },
            'drift_analysis': drift_analysis
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/clinical_outcomes')
def get_clinical_outcomes():
    """Get clinical outcomes report."""
    try:
        days_back = request.args.get('days', 30, type=int)
        impact_report = outcome_tracker.generate_impact_report(days_back)
        
        return jsonify({
            'success': True,
            'impact_report': impact_report
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/update_outcome', methods=['POST'])
def update_outcome():
    """Update actual outcome for a prediction."""
    try:
        data = request.json
        patient_id = data['patient_id']
        timestamp = data['timestamp']
        actual_outcome = data['actual_outcome']
        
        # Update in monitoring system
        performance_monitor.update_outcome(patient_id, timestamp, actual_outcome)
        
        # Record in outcomes tracker
        outcome_tracker.record_outcome(
            patient_id=patient_id,
            prediction_timestamp=timestamp,
            ai_diagnosis=data.get('ai_diagnosis', ''),
            ai_confidence=data.get('ai_confidence', 0.0),
            ai_risk_level=data.get('ai_risk_level', 'medium'),
            actual_diagnosis=actual_outcome,
            time_to_diagnosis_hours=data.get('time_to_diagnosis_hours'),
            treatment_effectiveness=data.get('treatment_effectiveness'),
            patient_satisfaction=data.get('patient_satisfaction'),
            readmission_30_days=data.get('readmission_30_days')
        )
        
        return jsonify({'success': True, 'message': 'Outcome updated successfully'})
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/ehr_sync/<patient_id>')
def sync_ehr_data(patient_id):
    """Sync patient data from EHR."""
    try:
        if not ehr_config['fhir_enabled']:
            return jsonify({
                'success': False,
                'error': 'EHR integration not configured'
            }), 400
        
        patient_data = ehr_manager.sync_patient_data(patient_id)
        
        if 'error' in patient_data:
            return jsonify({'success': False, 'error': patient_data['error']}), 400
        
        return jsonify({
            'success': True,
            'patient_data': patient_data
        })
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@socketio.on('connect')
def handle_connect():
    """Handle client connection."""
    emit('connected', {'message': 'Connected to clinical decision support system'})

@socketio.on('request_dashboard_update')
def handle_dashboard_update():
    """Send dashboard updates to client."""
    try:
        # Get recent metrics
        metrics = performance_monitor.calculate_performance_metrics(days_back=7)
        
        emit('dashboard_update', {
            'metrics': {
                'accuracy': metrics.accuracy,
                'prediction_count': metrics.prediction_count,
                'timestamp': metrics.timestamp
            },
            'timestamp': datetime.now().isoformat()
        })
    except Exception as e:
        emit('error', {'message': str(e)})

def _generate_integrated_recommendations(clinical_result, triage_result):
    """Generate integrated recommendations from clinical and triage assessments."""
    recommendations = []
    
    # Triage-based recommendations
    if triage_result['triage_level'] in ['RESUSCITATION', 'EMERGENT']:
        recommendations.append("IMMEDIATE medical attention required")
        recommendations.append("Activate emergency protocols")
    
    # Clinical pathway recommendations
    if 'clinical_pathway' in clinical_result:
        pathway = clinical_result['clinical_pathway']
        recommendations.extend([
            f"Follow {pathway.get('specialist', 'general')} pathway",
            f"Urgency level: {pathway.get('urgency', 'medium')}"
        ])
    
    # Confidence-based recommendations
    if clinical_result['confidence'] < 0.7:
        recommendations.append("Consider additional diagnostic tests")
        recommendations.append("Seek specialist consultation")
    
    return recommendations

if __name__ == '__main__':
    # Create templates directory and basic template if not exists
    templates_dir = os.path.join(os.path.dirname(__file__), 'templates')
    os.makedirs(templates_dir, exist_ok=True)
    
    socketio.run(app, debug=True, host='0.0.0.0', port=5000)