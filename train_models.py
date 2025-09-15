#!/usr/bin/env python3
"""
Quick model training script for the clinical decision support system.
"""
import os
import sys
import pandas as pd

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from ai_agent.engine import ClinicalDecisionEngine

def main():
    print("🏥 Training Clinical Decision Support Models...")
    
    # Load enhanced dataset
    data_path = 'data/enhanced_medical_dataset.csv'
    if not os.path.exists(data_path):
        print(f"❌ Dataset not found: {data_path}")
        return
    
    print("📊 Loading dataset...")
    df = pd.read_csv(data_path, comment='#')
    
    # Prepare features and labels
    exclude_cols = ['patient_id', 'diagnosis', 'severity', 'comorbidities']
    feature_cols = [col for col in df.columns if col not in exclude_cols]
    
    # Encode categorical variables
    df_encoded = df.copy()
    if 'gender' in df_encoded.columns:
        df_encoded['gender'] = df_encoded['gender'].map({'M': 1, 'F': 0})
    
    X = df_encoded[feature_cols]
    y = df_encoded['diagnosis']
    
    print(f"📈 Training on {len(df)} samples with {len(feature_cols)} features")
    print(f"🎯 Predicting {len(y.unique())} unique diagnoses")
    
    # Train ensemble
    engine = ClinicalDecisionEngine()
    engine.train_ensemble(X, y)
    
    # Save models
    models_dir = 'models'
    os.makedirs(models_dir, exist_ok=True)
    
    model_path = os.path.join(models_dir, 'clinical_model')
    engine.save_models(model_path)
    
    print("✅ Models trained and saved successfully!")
    print(f"📁 Models saved to: {model_path}")
    
    # Display performance metrics
    print("\n📊 Model Performance:")
    for model_name, metrics in engine.performance_metrics.items():
        print(f"  {model_name.upper()}: {metrics['cv_mean']:.3f} ± {metrics['cv_std']:.3f}")
    
    print("\n🚀 Ready to start the web application!")
    print("Run: python interface/advanced_webapp.py")

if __name__ == "__main__":
    main()