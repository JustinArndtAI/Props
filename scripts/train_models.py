"""
Train prop prediction models on REAL NFL data.
"""

import pandas as pd
import numpy as np
import sqlite3
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report
import xgboost as xgb
import joblib
from pathlib import Path
import logging
import json

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)


class RealPropModelTrainer:
    """Train models on actual NFL data."""
    
    def __init__(self, db_path="c:/Props_Project/data/nfl_props.db"):
        self.conn = sqlite3.connect(db_path)
        self.models = {}
        self.features = {}
        self.results = {}
    
    def create_features(self, prop_type):
        """Create features for a specific prop type from real data."""
        
        logger.info(f"Creating features for {prop_type}...")
        
        # Get prop lines with outcomes
        query = """
            SELECT 
                pl.*,
                ps.position,
                ps.team,
                ps.pass_attempts,
                ps.rush_attempts,
                ps.targets
            FROM prop_lines pl
            JOIN player_prop_stats ps 
                ON pl.player_name = ps.player_name 
                AND pl.week = ps.week 
                AND pl.season = ps.season
            WHERE pl.prop_type = ?
            ORDER BY pl.season, pl.week
        """
        
        df = pd.read_sql_query(query, self.conn, params=(prop_type,))
        
        if df.empty:
            logger.warning(f"No data for {prop_type}")
            return None
        
        # Get player historical stats for features
        features_list = []
        
        for idx, row in df.iterrows():
            features = {}
            
            # Basic info
            features['line'] = row['line']
            features['week'] = row['week']
            features['season'] = row['season']
            
            # Get player's recent performance (last 5 games)
            hist_query = """
                SELECT * FROM player_prop_stats
                WHERE player_name = ?
                AND (season < ? OR (season = ? AND week < ?))
                ORDER BY season DESC, week DESC
                LIMIT 5
            """
            
            hist_df = pd.read_sql_query(
                hist_query, self.conn, 
                params=(row['player_name'], row['season'], row['season'], row['week'])
            )
            
            if not hist_df.empty:
                # Calculate rolling averages based on prop type
                if prop_type == 'pass_yards':
                    features['avg_3games'] = hist_df.head(3)['pass_yards'].mean()
                    features['avg_5games'] = hist_df['pass_yards'].mean()
                    features['max_5games'] = hist_df['pass_yards'].max()
                    features['min_5games'] = hist_df['pass_yards'].min()
                    features['std_5games'] = hist_df['pass_yards'].std()
                    features['attempts_avg'] = hist_df['pass_attempts'].mean()
                    
                elif prop_type == 'pass_tds':
                    features['avg_3games'] = hist_df.head(3)['pass_tds'].mean()
                    features['avg_5games'] = hist_df['pass_tds'].mean()
                    features['max_5games'] = hist_df['pass_tds'].max()
                    features['td_rate'] = (hist_df['pass_tds'].sum() / 
                                          max(1, hist_df['pass_attempts'].sum()))
                    
                elif prop_type == 'rush_yards':
                    features['avg_3games'] = hist_df.head(3)['rush_yards'].mean()
                    features['avg_5games'] = hist_df['rush_yards'].mean()
                    features['max_5games'] = hist_df['rush_yards'].max()
                    features['min_5games'] = hist_df['rush_yards'].min()
                    features['attempts_avg'] = hist_df['rush_attempts'].mean()
                    features['yards_per_carry'] = (hist_df['rush_yards'].sum() / 
                                                  max(1, hist_df['rush_attempts'].sum()))
                    
                elif prop_type == 'receptions':
                    features['avg_3games'] = hist_df.head(3)['receptions'].mean()
                    features['avg_5games'] = hist_df['receptions'].mean()
                    features['targets_avg'] = hist_df['targets'].mean()
                    features['catch_rate'] = (hist_df['receptions'].sum() / 
                                             max(1, hist_df['targets'].sum()))
                    
                elif prop_type == 'rec_yards':
                    features['avg_3games'] = hist_df.head(3)['rec_yards'].mean()
                    features['avg_5games'] = hist_df['rec_yards'].mean()
                    features['max_5games'] = hist_df['rec_yards'].max()
                    features['targets_avg'] = hist_df['targets'].mean()
                    features['yards_per_target'] = (hist_df['rec_yards'].sum() / 
                                                   max(1, hist_df['targets'].sum()))
                    features['yards_per_catch'] = (hist_df['rec_yards'].sum() / 
                                                  max(1, hist_df['receptions'].sum()))
                
                # Trend (simple linear regression slope)
                if len(hist_df) >= 3:
                    x = np.arange(len(hist_df))
                    y = hist_df[self._get_stat_column(prop_type)].values
                    if len(x) > 1 and not np.all(y == y[0]):
                        features['trend'] = np.polyfit(x, y, 1)[0]
                    else:
                        features['trend'] = 0
                else:
                    features['trend'] = 0
                
                # Distance from line to average
                features['line_vs_avg'] = features.get('avg_5games', 0) - row['line']
                
            else:
                # No history - use defaults
                features['avg_3games'] = row['line']
                features['avg_5games'] = row['line']
                features['max_5games'] = row['line']
                features['min_5games'] = row['line']
                features['std_5games'] = 0
                features['trend'] = 0
                features['line_vs_avg'] = 0
            
            # Target variable
            features['hit_over'] = row['hit_over']
            features['player_name'] = row['player_name']
            
            features_list.append(features)
        
        features_df = pd.DataFrame(features_list)
        
        # Handle missing values
        features_df = features_df.fillna(0)
        features_df.replace([np.inf, -np.inf], 0, inplace=True)
        
        logger.info(f"Created {len(features_df)} samples with {len(features_df.columns)-2} features")
        
        return features_df
    
    def _get_stat_column(self, prop_type):
        """Get the stat column name for a prop type."""
        mapping = {
            'pass_yards': 'pass_yards',
            'pass_tds': 'pass_tds',
            'rush_yards': 'rush_yards',
            'receptions': 'receptions',
            'rec_yards': 'rec_yards'
        }
        return mapping.get(prop_type, prop_type)
    
    def train_model(self, prop_type):
        """Train XGBoost model for a specific prop type."""
        
        # Create features
        features_df = self.create_features(prop_type)
        
        if features_df is None or features_df.empty:
            logger.error(f"No features for {prop_type}")
            return None
        
        # Prepare data
        feature_cols = [col for col in features_df.columns 
                       if col not in ['hit_over', 'player_name']]
        
        X = features_df[feature_cols]
        y = features_df['hit_over']
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        logger.info(f"Training {prop_type} model on {len(X_train)} samples...")
        
        # Train XGBoost
        model = xgb.XGBClassifier(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            use_label_encoder=False,
            eval_metric='logloss'
        )
        
        model.fit(
            X_train, y_train,
            eval_set=[(X_test, y_test)],
            verbose=False
        )
        
        # Evaluate
        train_pred = model.predict(X_train)
        test_pred = model.predict(X_test)
        train_prob = model.predict_proba(X_train)[:, 1]
        test_prob = model.predict_proba(X_test)[:, 1]
        
        train_acc = accuracy_score(y_train, train_pred)
        test_acc = accuracy_score(y_test, test_pred)
        train_auc = roc_auc_score(y_train, train_prob)
        test_auc = roc_auc_score(y_test, test_prob)
        
        logger.info(f"{prop_type} Results:")
        logger.info(f"  Train Accuracy: {train_acc:.3f}")
        logger.info(f"  Test Accuracy: {test_acc:.3f}")
        logger.info(f"  Train AUC: {train_auc:.3f}")
        logger.info(f"  Test AUC: {test_auc:.3f}")
        
        # Feature importance
        importance = pd.DataFrame({
            'feature': feature_cols,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        print(f"\nTop Features for {prop_type}:")
        print(importance.head())
        
        # Store results
        self.models[prop_type] = model
        self.features[prop_type] = feature_cols
        self.results[prop_type] = {
            'train_acc': train_acc,
            'test_acc': test_acc,
            'train_auc': train_auc,
            'test_auc': test_auc,
            'n_samples': len(X),
            'n_features': len(feature_cols)
        }
        
        return model
    
    def save_models(self):
        """Save trained models to disk."""
        
        model_dir = Path("c:/Props_Project/models/trained")
        model_dir.mkdir(parents=True, exist_ok=True)
        
        for prop_type, model in self.models.items():
            # Save model
            model_path = model_dir / f"{prop_type}_model.pkl"
            joblib.dump(model, model_path)
            
            # Save features and metadata
            meta_path = model_dir / f"{prop_type}_meta.json"
            with open(meta_path, 'w') as f:
                json.dump({
                    'features': self.features[prop_type],
                    'results': self.results[prop_type]
                }, f, indent=2)
            
            logger.info(f"Saved {prop_type} model to {model_path}")
    
    def train_all_prop_types(self):
        """Train models for all prop types."""
        
        prop_types = ['pass_yards', 'pass_tds', 'rush_yards', 'receptions', 'rec_yards']
        
        print("\n" + "="*60)
        print("TRAINING MODELS ON REAL NFL DATA")
        print("="*60)
        
        for prop_type in prop_types:
            print(f"\n--- Training {prop_type} ---")
            self.train_model(prop_type)
        
        # Save all models
        self.save_models()
        
        # Summary
        print("\n" + "="*60)
        print("TRAINING COMPLETE - RESULTS SUMMARY")
        print("="*60)
        
        for prop_type, results in self.results.items():
            print(f"\n{prop_type.upper()}:")
            print(f"  Samples: {results['n_samples']}")
            print(f"  Test Accuracy: {results['test_acc']:.1%}")
            print(f"  Test AUC: {results['test_auc']:.3f}")
        
        avg_acc = np.mean([r['test_acc'] for r in self.results.values()])
        print(f"\nAverage Test Accuracy: {avg_acc:.1%}")
        
        return self.results


def main():
    trainer = RealPropModelTrainer()
    results = trainer.train_all_prop_types()
    
    print("\nModels are now trained on real NFL data!")
    print("Next: Generate predictions for actual player props")


if __name__ == "__main__":
    main()