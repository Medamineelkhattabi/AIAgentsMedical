# ai_agent/monitoring.py
"""
Advanced monitoring and analytics system for AI model performance and clinical outcomes.
"""
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
import json
import sqlite3
from dataclasses import dataclass, asdict
import logging
from collections import defaultdict

@dataclass
class PredictionLog:
    """Log entry for AI predictions."""
    timestamp: str
    patient_id: str
    model_version: str
    input_features: Dict[str, Any]
    prediction: str
    confidence: float
    risk_level: str
    actual_outcome: Optional[str] = None
    feedback_score: Optional[int] = None

@dataclass
class ModelMetrics:
    """Model performance metrics."""
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    auc_roc: float
    calibration_error: float
    prediction_count: int
    timestamp: str

class ModelPerformanceMonitor:
    """Monitor AI model performance and detect drift."""
    
    def __init__(self, db_path: str = "monitoring.db"):
        self.db_path = db_path
        self.logger = logging.getLogger('ModelMonitor')
        self._init_database()
        
    def _init_database(self):
        """Initialize monitoring database."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Predictions table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                patient_id TEXT NOT NULL,
                model_version TEXT NOT NULL,
                input_features TEXT NOT NULL,
                prediction TEXT NOT NULL,
                confidence REAL NOT NULL,
                risk_level TEXT NOT NULL,
                actual_outcome TEXT,
                feedback_score INTEGER
            )
        ''')
        
        # Model metrics table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS model_metrics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                model_version TEXT NOT NULL,
                accuracy REAL NOT NULL,
                precision REAL NOT NULL,
                recall REAL NOT NULL,
                f1_score REAL NOT NULL,
                auc_roc REAL NOT NULL,
                calibration_error REAL NOT NULL,
                prediction_count INTEGER NOT NULL
            )
        ''')
        
        # Alerts table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS alerts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                alert_type TEXT NOT NULL,
                severity TEXT NOT NULL,
                message TEXT NOT NULL,
                resolved BOOLEAN DEFAULT FALSE
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def log_prediction(self, prediction_log: PredictionLog):
        """Log a prediction for monitoring."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO predictions 
            (timestamp, patient_id, model_version, input_features, 
             prediction, confidence, risk_level, actual_outcome, feedback_score)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            prediction_log.timestamp,
            prediction_log.patient_id,
            prediction_log.model_version,
            json.dumps(prediction_log.input_features),
            prediction_log.prediction,
            prediction_log.confidence,
            prediction_log.risk_level,
            prediction_log.actual_outcome,
            prediction_log.feedback_score
        ))
        
        conn.commit()
        conn.close()
    
    def update_outcome(self, patient_id: str, timestamp: str, actual_outcome: str):
        """Update actual outcome for a prediction."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            UPDATE predictions 
            SET actual_outcome = ? 
            WHERE patient_id = ? AND timestamp = ?
        ''', (actual_outcome, patient_id, timestamp))
        
        conn.commit()
        conn.close()
    
    def calculate_performance_metrics(self, days_back: int = 30) -> ModelMetrics:
        """Calculate model performance metrics for recent predictions."""
        conn = sqlite3.connect(self.db_path)
        
        cutoff_date = (datetime.now() - timedelta(days=days_back)).isoformat()
        
        df = pd.read_sql_query('''
            SELECT prediction, actual_outcome, confidence, risk_level
            FROM predictions 
            WHERE timestamp > ? AND actual_outcome IS NOT NULL
        ''', conn, params=(cutoff_date,))
        
        conn.close()
        
        if df.empty:
            return ModelMetrics(0, 0, 0, 0, 0, 0, 0, datetime.now().isoformat())
        
        # Calculate metrics
        y_true = df['actual_outcome']
        y_pred = df['prediction']
        
        accuracy = (y_true == y_pred).mean()
        
        # For multi-class, calculate macro averages
        from sklearn.metrics import precision_score, recall_score, f1_score
        precision = precision_score(y_true, y_pred, average='macro', zero_division=0)
        recall = recall_score(y_true, y_pred, average='macro', zero_division=0)
        f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
        
        # Calibration error (simplified)
        calibration_error = self._calculate_calibration_error(df)
        
        metrics = ModelMetrics(
            accuracy=accuracy,
            precision=precision,
            recall=recall,
            f1_score=f1,
            auc_roc=0.0,  # Would need probability scores for all classes
            calibration_error=calibration_error,
            prediction_count=len(df),
            timestamp=datetime.now().isoformat()
        )
        
        # Store metrics
        self._store_metrics(metrics)
        
        return metrics
    
    def _calculate_calibration_error(self, df: pd.DataFrame) -> float:
        """Calculate calibration error for confidence scores."""
        if df.empty:
            return 0.0
        
        # Bin predictions by confidence
        bins = np.linspace(0, 1, 11)
        bin_indices = np.digitize(df['confidence'], bins) - 1
        
        calibration_error = 0.0
        total_samples = len(df)
        
        for i in range(len(bins) - 1):
            mask = bin_indices == i
            if mask.sum() == 0:
                continue
            
            bin_accuracy = (df[mask]['prediction'] == df[mask]['actual_outcome']).mean()
            bin_confidence = df[mask]['confidence'].mean()
            bin_size = mask.sum()
            
            calibration_error += (bin_size / total_samples) * abs(bin_accuracy - bin_confidence)
        
        return calibration_error
    
    def _store_metrics(self, metrics: ModelMetrics):
        """Store calculated metrics in database."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO model_metrics 
            (timestamp, model_version, accuracy, precision, recall, 
             f1_score, auc_roc, calibration_error, prediction_count)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            metrics.timestamp, "v1.0", metrics.accuracy, metrics.precision,
            metrics.recall, metrics.f1_score, metrics.auc_roc,
            metrics.calibration_error, metrics.prediction_count
        ))
        
        conn.commit()
        conn.close()
    
    def detect_model_drift(self, baseline_days: int = 90, 
                          current_days: int = 7) -> Dict[str, Any]:
        """Detect model performance drift."""
        baseline_metrics = self._get_metrics_for_period(baseline_days)
        current_metrics = self._get_metrics_for_period(current_days)
        
        if not baseline_metrics or not current_metrics:
            return {'drift_detected': False, 'reason': 'Insufficient data'}
        
        # Calculate drift thresholds
        accuracy_drift = abs(current_metrics['accuracy'] - baseline_metrics['accuracy'])
        precision_drift = abs(current_metrics['precision'] - baseline_metrics['precision'])
        
        drift_threshold = 0.05  # 5% threshold
        
        drift_detected = (accuracy_drift > drift_threshold or 
                         precision_drift > drift_threshold)
        
        if drift_detected:
            self._create_alert(
                'MODEL_DRIFT',
                'HIGH',
                f'Model drift detected: Accuracy drift: {accuracy_drift:.3f}, '
                f'Precision drift: {precision_drift:.3f}'
            )
        
        return {
            'drift_detected': drift_detected,
            'accuracy_drift': accuracy_drift,
            'precision_drift': precision_drift,
            'baseline_period': f'{baseline_days} days',
            'current_period': f'{current_days} days'
        }
    
    def _get_metrics_for_period(self, days_back: int) -> Optional[Dict[str, float]]:
        """Get average metrics for a time period."""
        conn = sqlite3.connect(self.db_path)
        
        cutoff_date = (datetime.now() - timedelta(days=days_back)).isoformat()
        
        df = pd.read_sql_query('''
            SELECT accuracy, precision, recall, f1_score
            FROM model_metrics 
            WHERE timestamp > ?
        ''', conn, params=(cutoff_date,))
        
        conn.close()
        
        if df.empty:
            return None
        
        return {
            'accuracy': df['accuracy'].mean(),
            'precision': df['precision'].mean(),
            'recall': df['recall'].mean(),
            'f1_score': df['f1_score'].mean()
        }
    
    def _create_alert(self, alert_type: str, severity: str, message: str):
        """Create monitoring alert."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO alerts (timestamp, alert_type, severity, message)
            VALUES (?, ?, ?, ?)
        ''', (datetime.now().isoformat(), alert_type, severity, message))
        
        conn.commit()
        conn.close()
        
        self.logger.warning(f"{severity} {alert_type}: {message}")

class ClinicalOutcomeTracker:
    """Track clinical outcomes and measure AI impact."""
    
    def __init__(self, db_path: str = "outcomes.db"):
        self.db_path = db_path
        self._init_database()
    
    def _init_database(self):
        """Initialize outcomes database."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS clinical_outcomes (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                patient_id TEXT NOT NULL,
                prediction_timestamp TEXT NOT NULL,
                ai_diagnosis TEXT NOT NULL,
                ai_confidence REAL NOT NULL,
                ai_risk_level TEXT NOT NULL,
                actual_diagnosis TEXT,
                time_to_diagnosis_hours REAL,
                treatment_effectiveness TEXT,
                patient_satisfaction INTEGER,
                readmission_30_days BOOLEAN,
                outcome_timestamp TEXT NOT NULL
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def record_outcome(self, patient_id: str, prediction_timestamp: str,
                      ai_diagnosis: str, ai_confidence: float, ai_risk_level: str,
                      actual_diagnosis: str = None, time_to_diagnosis_hours: float = None,
                      treatment_effectiveness: str = None, patient_satisfaction: int = None,
                      readmission_30_days: bool = None):
        """Record clinical outcome."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO clinical_outcomes 
            (patient_id, prediction_timestamp, ai_diagnosis, ai_confidence, 
             ai_risk_level, actual_diagnosis, time_to_diagnosis_hours,
             treatment_effectiveness, patient_satisfaction, readmission_30_days,
             outcome_timestamp)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            patient_id, prediction_timestamp, ai_diagnosis, ai_confidence,
            ai_risk_level, actual_diagnosis, time_to_diagnosis_hours,
            treatment_effectiveness, patient_satisfaction, readmission_30_days,
            datetime.now().isoformat()
        ))
        
        conn.commit()
        conn.close()
    
    def generate_impact_report(self, days_back: int = 90) -> Dict[str, Any]:
        """Generate AI impact report."""
        conn = sqlite3.connect(self.db_path)
        
        cutoff_date = (datetime.now() - timedelta(days=days_back)).isoformat()
        
        df = pd.read_sql_query('''
            SELECT * FROM clinical_outcomes 
            WHERE outcome_timestamp > ?
        ''', conn, params=(cutoff_date,))
        
        conn.close()
        
        if df.empty:
            return {'error': 'No data available for the specified period'}
        
        # Calculate impact metrics
        diagnostic_accuracy = (df['ai_diagnosis'] == df['actual_diagnosis']).mean()
        avg_time_to_diagnosis = df['time_to_diagnosis_hours'].mean()
        avg_patient_satisfaction = df['patient_satisfaction'].mean()
        readmission_rate = df['readmission_30_days'].mean()
        
        # Risk stratification effectiveness
        high_risk_accuracy = df[df['ai_risk_level'] == 'high']['ai_diagnosis'].eq(
            df[df['ai_risk_level'] == 'high']['actual_diagnosis']
        ).mean()
        
        return {
            'period_days': days_back,
            'total_cases': len(df),
            'diagnostic_accuracy': diagnostic_accuracy,
            'avg_time_to_diagnosis_hours': avg_time_to_diagnosis,
            'avg_patient_satisfaction': avg_patient_satisfaction,
            'readmission_rate_30_days': readmission_rate,
            'high_risk_accuracy': high_risk_accuracy,
            'confidence_correlation': df['ai_confidence'].corr(
                (df['ai_diagnosis'] == df['actual_diagnosis']).astype(int)
            )
        }