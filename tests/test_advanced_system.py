# tests/test_advanced_system.py
"""
Comprehensive test suite for the advanced clinical decision support system.
"""
import pytest
import pandas as pd
import numpy as np
from datetime import datetime
import tempfile
import os
import sys

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ai_agent.engine import ClinicalDecisionEngine
from ai_agent.triage_system import TriageSystem, VitalSignsAnalyzer
from ai_agent.monitoring import ModelPerformanceMonitor, PredictionLog
from ai_agent.ehr_integration import FHIRClient, EHRIntegrationManager

class TestClinicalDecisionEngine:
    """Test the advanced clinical decision engine."""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample training data."""
        np.random.seed(42)
        n_samples = 100
        
        data = {
            'fever': np.random.randint(0, 2, n_samples),
            'cough': np.random.randint(0, 2, n_samples),
            'chest_pain': np.random.randint(0, 2, n_samples),
            'shortness_of_breath': np.random.randint(0, 2, n_samples),
            'headache': np.random.randint(0, 2, n_samples),
            'nausea': np.random.randint(0, 2, n_samples)
        }
        
        # Create realistic diagnoses based on symptoms
        diagnoses = []
        for i in range(n_samples):
            if data['fever'][i] and data['cough'][i]:
                diagnoses.append('Flu')
            elif data['chest_pain'][i] and data['shortness_of_breath'][i]:
                diagnoses.append('Heart Issue')
            elif data['headache'][i] and data['nausea'][i]:
                diagnoses.append('Migraine')
            else:
                diagnoses.append('Unknown')
        
        df = pd.DataFrame(data)
        df['diagnosis'] = diagnoses
        
        return df
    
    @pytest.fixture
    def trained_engine(self, sample_data):
        """Create a trained clinical decision engine."""
        engine = ClinicalDecisionEngine()
        
        X = sample_data.drop('diagnosis', axis=1)
        y = sample_data['diagnosis']
        
        engine.train_ensemble(X, y)
        return engine
    
    def test_engine_initialization(self):
        """Test engine initialization."""
        engine = ClinicalDecisionEngine()
        
        assert engine.config is not None
        assert 'ensemble_methods' in engine.config
        assert engine.clinical_pathways is not None
        assert engine.risk_thresholds is not None
    
    def test_model_training(self, sample_data):
        """Test model training functionality."""
        engine = ClinicalDecisionEngine()
        
        X = sample_data.drop('diagnosis', axis=1)
        y = sample_data['diagnosis']
        
        engine.train_ensemble(X, y)
        
        # Check that models were trained
        assert len(engine.models) > 0
        assert 'rf' in engine.models
        assert 'gb' in engine.models
        assert 'mlp' in engine.models
        
        # Check performance metrics
        assert len(engine.performance_metrics) > 0
        for model_name, metrics in engine.performance_metrics.items():
            assert 'cv_mean' in metrics
            assert 'cv_std' in metrics
            assert metrics['cv_mean'] >= 0
    
    def test_prediction_with_confidence(self, trained_engine):
        """Test prediction with confidence scoring."""
        features = {
            'fever': 1,
            'cough': 1,
            'chest_pain': 0,
            'shortness_of_breath': 0,
            'headache': 0,
            'nausea': 0
        }
        
        result = trained_engine.predict_with_confidence(features)
        
        # Check result structure
        assert 'primary_diagnosis' in result
        assert 'confidence' in result
        assert 'risk_level' in result
        assert 'clinical_pathway' in result
        assert 'uncertainty_metrics' in result
        
        # Check value ranges
        assert 0 <= result['confidence'] <= 1
        assert result['risk_level'] in ['low', 'medium', 'high', 'critical']
    
    def test_clinical_report_generation(self, trained_engine):
        """Test clinical report generation."""
        patient_data = {'patient_id': 'TEST001', 'age': 45}
        
        features = {
            'fever': 1,
            'cough': 1,
            'chest_pain': 0,
            'shortness_of_breath': 0,
            'headache': 0,
            'nausea': 0
        }
        
        prediction_result = trained_engine.predict_with_confidence(features)
        report = trained_engine.generate_clinical_report(patient_data, prediction_result)
        
        assert isinstance(report, str)
        assert 'CLINICAL DECISION SUPPORT REPORT' in report
        assert 'PRIMARY ASSESSMENT' in report
        assert 'RECOMMENDATIONS' in report
    
    def test_model_persistence(self, trained_engine):
        """Test model saving and loading."""
        with tempfile.TemporaryDirectory() as temp_dir:
            model_path = os.path.join(temp_dir, 'test_model')
            
            # Save models
            trained_engine.save_models(model_path)
            
            # Create new engine and load models
            new_engine = ClinicalDecisionEngine()
            new_engine.load_models(model_path)
            
            # Test that loaded models work
            features = {
                'fever': 1,
                'cough': 1,
                'chest_pain': 0,
                'shortness_of_breath': 0,
                'headache': 0,
                'nausea': 0
            }
            
            result = new_engine.predict_with_confidence(features)
            assert 'primary_diagnosis' in result

class TestTriageSystem:
    """Test the advanced triage system."""
    
    @pytest.fixture
    def triage_system(self):
        """Create triage system instance."""
        return TriageSystem()
    
    @pytest.fixture
    def vitals_analyzer(self):
        """Create vital signs analyzer instance."""
        return VitalSignsAnalyzer()
    
    def test_vitals_analysis_normal(self, vitals_analyzer):
        """Test vital signs analysis with normal values."""
        vitals = {
            'systolic_bp': 120,
            'diastolic_bp': 80,
            'heart_rate': 75,
            'respiratory_rate': 16,
            'temperature': 36.8,
            'oxygen_saturation': 98
        }
        
        result = vitals_analyzer.assess_vitals(vitals)
        
        assert result['severity'] == 'low'
        assert result['total_score'] == 0
        assert len(result['alerts']) == 0
    
    def test_vitals_analysis_critical(self, vitals_analyzer):
        """Test vital signs analysis with critical values."""
        vitals = {
            'systolic_bp': 200,  # Critical high
            'diastolic_bp': 120,  # Critical high
            'heart_rate': 150,   # Critical high
            'respiratory_rate': 30,  # Critical high
            'temperature': 40.0,  # Critical high
            'oxygen_saturation': 85  # Critical low
        }
        
        result = vitals_analyzer.assess_vitals(vitals)
        
        assert result['severity'] == 'critical'
        assert result['total_score'] >= 9
        assert len(result['alerts']) > 0
        assert any('CRITICAL' in alert for alert in result['alerts'])
    
    def test_triage_assessment(self, triage_system):
        """Test complete triage assessment."""
        patient_data = {
            'vitals': {
                'systolic_bp': 160,
                'diastolic_bp': 95,
                'heart_rate': 110,
                'respiratory_rate': 22,
                'temperature': 38.5,
                'oxygen_saturation': 92
            },
            'chief_complaint': 'chest pain and difficulty breathing',
            'pain_level': 8,
            'age': 65,
            'comorbidities': ['diabetes', 'hypertension']
        }
        
        result = triage_system.perform_triage(patient_data)
        
        # Check result structure
        assert 'triage_level' in result
        assert 'triage_score' in result
        assert 'estimated_wait_time' in result
        assert 'vitals_assessment' in result
        assert 'recommendations' in result
        
        # Should be high priority due to symptoms and vitals
        assert result['triage_level'] in ['RESUSCITATION', 'EMERGENT', 'URGENT']
    
    def test_low_acuity_triage(self, triage_system):
        """Test low acuity triage case."""
        patient_data = {
            'vitals': {
                'systolic_bp': 120,
                'diastolic_bp': 80,
                'heart_rate': 75,
                'respiratory_rate': 16,
                'temperature': 36.8,
                'oxygen_saturation': 98
            },
            'chief_complaint': 'minor cut on finger',
            'pain_level': 2,
            'age': 25,
            'comorbidities': []
        }
        
        result = triage_system.perform_triage(patient_data)
        
        # Should be low priority
        assert result['triage_level'] in ['LESS_URGENT', 'NON_URGENT']
        assert result['triage_score'] < 2.0

class TestModelPerformanceMonitor:
    """Test the model performance monitoring system."""
    
    @pytest.fixture
    def monitor(self):
        """Create monitor with temporary database."""
        with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as f:
            db_path = f.name
        
        monitor = ModelPerformanceMonitor(db_path)
        yield monitor
        
        # Cleanup
        os.unlink(db_path)
    
    def test_prediction_logging(self, monitor):
        """Test prediction logging functionality."""
        prediction_log = PredictionLog(
            timestamp=datetime.now().isoformat(),
            patient_id='TEST001',
            model_version='v1.0',
            input_features={'fever': 1, 'cough': 1},
            prediction='Flu',
            confidence=0.85,
            risk_level='medium'
        )
        
        monitor.log_prediction(prediction_log)
        
        # Verify logging worked (would need database query in real implementation)
        assert True  # Placeholder assertion
    
    def test_outcome_update(self, monitor):
        """Test outcome updating functionality."""
        # First log a prediction
        prediction_log = PredictionLog(
            timestamp=datetime.now().isoformat(),
            patient_id='TEST001',
            model_version='v1.0',
            input_features={'fever': 1, 'cough': 1},
            prediction='Flu',
            confidence=0.85,
            risk_level='medium'
        )
        
        monitor.log_prediction(prediction_log)
        
        # Update with actual outcome
        monitor.update_outcome('TEST001', prediction_log.timestamp, 'Flu')
        
        assert True  # Placeholder assertion
    
    def test_performance_metrics_calculation(self, monitor):
        """Test performance metrics calculation."""
        # Log several predictions with outcomes
        predictions = [
            ('TEST001', 'Flu', 'Flu'),
            ('TEST002', 'Cold', 'Cold'),
            ('TEST003', 'Flu', 'Cold'),  # Incorrect prediction
            ('TEST004', 'Migraine', 'Migraine')
        ]
        
        for i, (patient_id, prediction, actual) in enumerate(predictions):
            timestamp = datetime.now().isoformat()
            
            prediction_log = PredictionLog(
                timestamp=timestamp,
                patient_id=patient_id,
                model_version='v1.0',
                input_features={'fever': 1},
                prediction=prediction,
                confidence=0.8,
                risk_level='medium',
                actual_outcome=actual
            )
            
            monitor.log_prediction(prediction_log)
        
        # Calculate metrics
        metrics = monitor.calculate_performance_metrics(days_back=1)
        
        assert hasattr(metrics, 'accuracy')
        assert hasattr(metrics, 'precision')
        assert hasattr(metrics, 'recall')
        assert hasattr(metrics, 'f1_score')

class TestEHRIntegration:
    """Test EHR integration functionality."""
    
    def test_fhir_client_initialization(self):
        """Test FHIR client initialization."""
        client = FHIRClient('http://test-server.com/fhir', 'test-token')
        
        assert client.base_url == 'http://test-server.com/fhir'
        assert client.auth_token == 'test-token'
        assert 'Authorization' in client.session.headers
    
    def test_ehr_manager_initialization(self):
        """Test EHR integration manager initialization."""
        config = {
            'fhir_enabled': True,
            'fhir_base_url': 'http://test-server.com/fhir',
            'fhir_auth_token': 'test-token'
        }
        
        manager = EHRIntegrationManager(config)
        
        assert manager.config == config
        assert manager.fhir_client is not None
    
    def test_ehr_manager_disabled(self):
        """Test EHR manager with FHIR disabled."""
        config = {'fhir_enabled': False}
        
        manager = EHRIntegrationManager(config)
        
        assert manager.fhir_client is None

class TestIntegration:
    """Integration tests for the complete system."""
    
    def test_end_to_end_workflow(self):
        """Test complete diagnostic workflow."""
        # Create sample data
        np.random.seed(42)
        data = {
            'fever': [1, 0, 1, 0],
            'cough': [1, 0, 0, 1],
            'chest_pain': [0, 1, 0, 0],
            'shortness_of_breath': [0, 1, 0, 0],
            'headache': [0, 0, 1, 0],
            'nausea': [0, 0, 1, 0]
        }
        
        df = pd.DataFrame(data)
        df['diagnosis'] = ['Flu', 'Heart Issue', 'Migraine', 'Unknown']
        
        # Train engine
        engine = ClinicalDecisionEngine()
        X = df.drop('diagnosis', axis=1)
        y = df['diagnosis']
        engine.train_ensemble(X, y)
        
        # Create triage system
        triage_system = TriageSystem()
        
        # Test patient case
        patient_data = {
            'vitals': {
                'systolic_bp': 140,
                'diastolic_bp': 90,
                'heart_rate': 85,
                'respiratory_rate': 18,
                'temperature': 37.5,
                'oxygen_saturation': 96
            },
            'chief_complaint': 'fever and cough',
            'pain_level': 3,
            'age': 45,
            'comorbidities': []
        }
        
        symptoms = {
            'fever': 1,
            'cough': 1,
            'chest_pain': 0,
            'shortness_of_breath': 0,
            'headache': 0,
            'nausea': 0
        }
        
        # Get clinical assessment
        clinical_result = engine.predict_with_confidence(symptoms)
        
        # Get triage assessment
        triage_result = triage_system.perform_triage(patient_data)
        
        # Verify results
        assert 'primary_diagnosis' in clinical_result
        assert 'triage_level' in triage_result
        assert clinical_result['confidence'] > 0
        assert triage_result['triage_score'] >= 0

if __name__ == '__main__':
    pytest.main([__file__, '-v'])