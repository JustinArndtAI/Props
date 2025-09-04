import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
import xgboost as xgb
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, roc_auc_score, precision_recall_curve
from sklearn.calibration import CalibratedClassifierCV
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import optuna
import joblib
import json
from pathlib import Path
import logging
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class XGBoostPropPredictor:
    """XGBoost model for binary prop predictions."""
    
    def __init__(self, prop_type: str):
        self.prop_type = prop_type
        self.model = None
        self.calibrated_model = None
        self.feature_names = None
        self.best_params = None
        self.metrics = {}
    
    def train(self, X_train: pd.DataFrame, y_train: pd.Series,
             X_val: pd.DataFrame = None, y_val: pd.Series = None,
             optimize_hyperparams: bool = True):
        """Train XGBoost model."""
        
        logger.info(f"Training XGBoost for {self.prop_type}")
        
        self.feature_names = X_train.columns.tolist()
        
        if optimize_hyperparams:
            self.best_params = self._optimize_hyperparameters(X_train, y_train)
        else:
            self.best_params = {
                'n_estimators': 100,
                'max_depth': 6,
                'learning_rate': 0.1,
                'subsample': 0.8,
                'colsample_bytree': 0.8
            }
        
        # Train final model
        self.model = xgb.XGBClassifier(
            **self.best_params,
            use_label_encoder=False,
            eval_metric='logloss',
            random_state=42
        )
        
        if X_val is not None and y_val is not None:
            eval_set = [(X_val, y_val)]
            self.model.fit(
                X_train, y_train,
                eval_set=eval_set,
                early_stopping_rounds=10,
                verbose=False
            )
        else:
            self.model.fit(X_train, y_train)
        
        # Calibrate probabilities
        self.calibrated_model = CalibratedClassifierCV(
            self.model, cv=3, method='sigmoid'
        )
        self.calibrated_model.fit(X_train, y_train)
        
        # Calculate metrics
        self._calculate_metrics(X_train, y_train, X_val, y_val)
    
    def _optimize_hyperparameters(self, X: pd.DataFrame, y: pd.Series) -> Dict:
        """Optimize hyperparameters using Optuna."""
        
        def objective(trial):
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 50, 300),
                'max_depth': trial.suggest_int('max_depth', 3, 10),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
                'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
                'gamma': trial.suggest_float('gamma', 0, 0.5)
            }
            
            model = xgb.XGBClassifier(
                **params,
                use_label_encoder=False,
                eval_metric='logloss',
                random_state=42
            )
            
            scores = cross_val_score(model, X, y, cv=3, scoring='roc_auc')
            return scores.mean()
        
        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=20)
        
        logger.info(f"Best params: {study.best_params}")
        return study.best_params
    
    def _calculate_metrics(self, X_train, y_train, X_val, y_val):
        """Calculate and store model metrics."""
        
        # Training metrics
        train_pred = self.model.predict(X_train)
        train_prob = self.calibrated_model.predict_proba(X_train)[:, 1]
        
        self.metrics['train_accuracy'] = accuracy_score(y_train, train_pred)
        self.metrics['train_auc'] = roc_auc_score(y_train, train_prob)
        
        # Validation metrics
        if X_val is not None and y_val is not None:
            val_pred = self.model.predict(X_val)
            val_prob = self.calibrated_model.predict_proba(X_val)[:, 1]
            
            self.metrics['val_accuracy'] = accuracy_score(y_val, val_pred)
            self.metrics['val_auc'] = roc_auc_score(y_val, val_prob)
        
        logger.info(f"Metrics: {self.metrics}")
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Predict probabilities for over."""
        
        if self.calibrated_model is None:
            raise ValueError("Model not trained yet")
        
        return self.calibrated_model.predict_proba(X)[:, 1]
    
    def get_feature_importance(self) -> pd.DataFrame:
        """Get feature importance."""
        
        if self.model is None:
            return pd.DataFrame()
        
        importance_df = pd.DataFrame({
            'feature': self.feature_names,
            'importance': self.model.feature_importances_
        })
        
        return importance_df.sort_values('importance', ascending=False)
    
    def save(self, path: str):
        """Save model to disk."""
        
        model_data = {
            'prop_type': self.prop_type,
            'best_params': self.best_params,
            'feature_names': self.feature_names,
            'metrics': self.metrics
        }
        
        # Save model
        joblib.dump(self.model, f"{path}_xgb.pkl")
        joblib.dump(self.calibrated_model, f"{path}_calibrated.pkl")
        
        # Save metadata
        with open(f"{path}_meta.json", 'w') as f:
            json.dump(model_data, f, indent=2)
        
        logger.info(f"Model saved to {path}")
    
    def load(self, path: str):
        """Load model from disk."""
        
        self.model = joblib.load(f"{path}_xgb.pkl")
        self.calibrated_model = joblib.load(f"{path}_calibrated.pkl")
        
        with open(f"{path}_meta.json", 'r') as f:
            model_data = json.load(f)
        
        self.prop_type = model_data['prop_type']
        self.best_params = model_data['best_params']
        self.feature_names = model_data['feature_names']
        self.metrics = model_data['metrics']
        
        logger.info(f"Model loaded from {path}")


class LSTMPropPredictor(nn.Module):
    """LSTM model for sequence-based prop predictions."""
    
    def __init__(self, input_size: int, hidden_size: int = 128, 
                 num_layers: int = 2, dropout: float = 0.2):
        super().__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(
            input_size, hidden_size, num_layers,
            batch_first=True, dropout=dropout
        )
        
        self.fc1 = nn.Linear(hidden_size, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 1)
        
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        # LSTM forward pass
        lstm_out, _ = self.lstm(x)
        
        # Take last timestep
        last_hidden = lstm_out[:, -1, :]
        
        # Fully connected layers
        x = self.relu(self.fc1(last_hidden))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        
        return self.sigmoid(x)


class EnsemblePropPredictor:
    """Ensemble of multiple models for prop predictions."""
    
    def __init__(self, prop_type: str):
        self.prop_type = prop_type
        self.models = {}
        self.weights = {}
        self.metrics = {}
    
    def add_model(self, name: str, model, weight: float = 1.0):
        """Add a model to the ensemble."""
        self.models[name] = model
        self.weights[name] = weight
    
    def train_all(self, X_train, y_train, X_val=None, y_val=None):
        """Train all models in ensemble."""
        
        for name, model in self.models.items():
            logger.info(f"Training {name} for {self.prop_type}")
            
            if hasattr(model, 'train'):
                model.train(X_train, y_train, X_val, y_val)
            else:
                # For sklearn-style models
                model.fit(X_train, y_train)
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Get ensemble predictions."""
        
        predictions = []
        total_weight = sum(self.weights.values())
        
        for name, model in self.models.items():
            weight = self.weights[name] / total_weight
            
            if hasattr(model, 'predict_proba'):
                if len(model.predict_proba(X).shape) > 1:
                    pred = model.predict_proba(X)[:, 1]
                else:
                    pred = model.predict_proba(X)
            else:
                pred = model.predict(X)
            
            predictions.append(pred * weight)
        
        return np.sum(predictions, axis=0)
    
    def evaluate(self, X_test, y_test) -> Dict:
        """Evaluate ensemble performance."""
        
        y_pred_proba = self.predict_proba(X_test)
        y_pred = (y_pred_proba > 0.5).astype(int)
        
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'auc': roc_auc_score(y_test, y_pred_proba)
        }
        
        # Individual model metrics
        for name, model in self.models.items():
            if hasattr(model, 'predict_proba'):
                model_pred = model.predict_proba(X_test)
                if len(model_pred.shape) > 1:
                    model_pred = model_pred[:, 1]
                
                metrics[f'{name}_auc'] = roc_auc_score(y_test, model_pred)
        
        self.metrics = metrics
        return metrics


class ContinualLearning:
    """Implements continual learning for model improvement."""
    
    def __init__(self, base_model, buffer_size: int = 1000):
        self.base_model = base_model
        self.buffer_size = buffer_size
        self.replay_buffer = []
        self.update_count = 0
    
    def update_with_new_data(self, X_new, y_new, epochs: int = 5):
        """Update model with new data using replay buffer."""
        
        logger.info(f"Updating model with {len(X_new)} new samples")
        
        # Add to replay buffer
        for i in range(len(X_new)):
            self.replay_buffer.append((X_new.iloc[i], y_new.iloc[i]))
        
        # Maintain buffer size
        if len(self.replay_buffer) > self.buffer_size:
            self.replay_buffer = self.replay_buffer[-self.buffer_size:]
        
        # Create mixed batch
        buffer_X = pd.DataFrame([x for x, _ in self.replay_buffer])
        buffer_y = pd.Series([y for _, y in self.replay_buffer])
        
        # Fine-tune model
        if hasattr(self.base_model, 'partial_fit'):
            for _ in range(epochs):
                self.base_model.partial_fit(buffer_X, buffer_y)
        else:
            # Re-train with mixed data
            combined_X = pd.concat([buffer_X, X_new])
            combined_y = pd.concat([buffer_y, y_new])
            self.base_model.fit(combined_X, combined_y)
        
        self.update_count += 1
        logger.info(f"Model updated ({self.update_count} total updates)")


def main():
    """Example usage of prop predictor models."""
    print("=" * 60)
    print("PROP PREDICTOR MODELS")
    print("=" * 60)
    
    # Generate sample data
    np.random.seed(42)
    n_samples = 1000
    n_features = 20
    
    X = pd.DataFrame(
        np.random.randn(n_samples, n_features),
        columns=[f'feature_{i}' for i in range(n_features)]
    )
    y = pd.Series((X['feature_0'] + X['feature_1'] > 0).astype(int))
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    print("\n1. Training XGBoost Model...")
    xgb_model = XGBoostPropPredictor(prop_type="pass_yards")
    xgb_model.train(X_train, y_train, X_test, y_test, optimize_hyperparams=False)
    
    print(f"   Train Accuracy: {xgb_model.metrics.get('train_accuracy', 0):.3f}")
    print(f"   Val Accuracy: {xgb_model.metrics.get('val_accuracy', 0):.3f}")
    print(f"   Val AUC: {xgb_model.metrics.get('val_auc', 0):.3f}")
    
    # Feature importance
    print("\n2. Top 5 Important Features:")
    importance_df = xgb_model.get_feature_importance()
    for _, row in importance_df.head(5).iterrows():
        print(f"   {row['feature']}: {row['importance']:.3f}")
    
    # Ensemble
    print("\n3. Creating Ensemble Model...")
    ensemble = EnsemblePropPredictor(prop_type="pass_yards")
    ensemble.add_model("xgboost", xgb_model, weight=1.0)
    
    # You could add more models here
    # ensemble.add_model("lstm", lstm_model, weight=0.5)
    
    ensemble_metrics = ensemble.evaluate(X_test, y_test)
    print(f"   Ensemble Accuracy: {ensemble_metrics['accuracy']:.3f}")
    print(f"   Ensemble AUC: {ensemble_metrics['auc']:.3f}")
    
    # Continual learning
    print("\n4. Continual Learning Update...")
    cl = ContinualLearning(xgb_model.model)
    
    # Simulate new weekly data
    X_new = pd.DataFrame(
        np.random.randn(50, n_features),
        columns=[f'feature_{i}' for i in range(n_features)]
    )
    y_new = pd.Series((X_new['feature_0'] + X_new['feature_1'] > 0).astype(int))
    
    cl.update_with_new_data(X_new, y_new)
    print("   Model updated with new weekly data")
    
    # Save model
    print("\n5. Saving Model...")
    Path("c:/Props_Project/models/saved").mkdir(parents=True, exist_ok=True)
    xgb_model.save("c:/Props_Project/models/saved/pass_yards_model")
    print("   Model saved successfully")
    
    print("\n" + "=" * 60)
    print("Models ready for prop prediction!")


if __name__ == "__main__":
    main()