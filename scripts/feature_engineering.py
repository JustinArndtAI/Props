import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
import sqlite3
from datetime import datetime, timedelta
from sklearn.preprocessing import StandardScaler, LabelEncoder
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FeatureEngineer:
    """Creates features for NFL player prop predictions."""
    
    def __init__(self, db_path: str = "data/nfl_props.db"):
        self.db_path = db_path
        self.conn = sqlite3.connect(db_path)
        self.scaler = StandardScaler()
        self.label_encoders = {}
    
    def create_player_features(self, player_id: str, prop_type: str, 
                              week: int, season: int) -> Dict:
        """Create features for a specific player and prop type."""
        
        features = {}
        
        # Get player recent performance
        recent_stats = self._get_recent_stats(player_id, week, season)
        
        if recent_stats.empty:
            logger.warning(f"No recent stats for player {player_id}")
            return features
        
        # Calculate rolling averages for the specific stat
        stat_mapping = {
            'pass_yards': 'pass_yards',
            'pass_tds': 'pass_tds',
            'rush_yards': 'rush_yards',
            'receptions': 'receptions',
            'rec_yards': 'rec_yards'
        }
        
        if prop_type in stat_mapping:
            stat_col = stat_mapping[prop_type]
            
            # Last 3 games average
            features[f'{prop_type}_l3_avg'] = recent_stats[stat_col].head(3).mean()
            features[f'{prop_type}_l3_std'] = recent_stats[stat_col].head(3).std()
            
            # Last 5 games average
            features[f'{prop_type}_l5_avg'] = recent_stats[stat_col].head(5).mean()
            features[f'{prop_type}_l5_max'] = recent_stats[stat_col].head(5).max()
            features[f'{prop_type}_l5_min'] = recent_stats[stat_col].head(5).min()
            
            # Season average
            features[f'{prop_type}_season_avg'] = recent_stats[stat_col].mean()
            
            # Trend (linear regression slope)
            if len(recent_stats) >= 3:
                x = np.arange(len(recent_stats.head(5)))
                y = recent_stats[stat_col].head(5).values
                if len(x) > 1:
                    slope = np.polyfit(x, y, 1)[0]
                    features[f'{prop_type}_trend'] = slope
        
        # Position-specific features
        position = self._get_player_position(player_id)
        features['position_encoded'] = self._encode_position(position)
        
        # Team and opponent features
        team_features = self._get_team_features(player_id, week, season)
        features.update(team_features)
        
        # Game context features
        game_features = self._get_game_context(player_id, week, season)
        features.update(game_features)
        
        # Historical prop performance
        prop_history = self._get_prop_history(player_id, prop_type)
        features.update(prop_history)
        
        return features
    
    def _get_recent_stats(self, player_id: str, week: int, season: int) -> pd.DataFrame:
        """Get recent stats for a player."""
        query = """
            SELECT * FROM player_stats
            WHERE player_id = ?
            AND (season < ? OR (season = ? AND week < ?))
            ORDER BY season DESC, week DESC
            LIMIT 10
        """
        
        return pd.read_sql_query(query, self.conn, 
                                params=(player_id, season, season, week))
    
    def _get_player_position(self, player_id: str) -> str:
        """Get player's position."""
        query = "SELECT position FROM players WHERE player_id = ?"
        cursor = self.conn.cursor()
        result = cursor.execute(query, (player_id,)).fetchone()
        return result[0] if result else "Unknown"
    
    def _encode_position(self, position: str) -> int:
        """Encode position as numeric value."""
        position_map = {
            'QB': 0, 'RB': 1, 'WR': 2, 'TE': 3, 
            'K': 4, 'DEF': 5, 'Unknown': 6
        }
        return position_map.get(position, 6)
    
    def _get_team_features(self, player_id: str, week: int, season: int) -> Dict:
        """Get team-related features."""
        features = {}
        
        # Get player's team
        query = "SELECT team FROM players WHERE player_id = ?"
        cursor = self.conn.cursor()
        result = cursor.execute(query, (player_id,)).fetchone()
        
        if result:
            team = result[0]
            
            # Team offensive ranking (placeholder - would calculate from data)
            features['team_off_rank'] = np.random.randint(1, 33)
            
            # Team pace (plays per game - placeholder)
            features['team_pace'] = np.random.uniform(60, 70)
            
            # Home/away (would determine from schedule)
            features['is_home'] = np.random.choice([0, 1])
        
        # Opponent defensive ranking (placeholder)
        features['opp_def_rank'] = np.random.randint(1, 33)
        features['opp_def_vs_position'] = np.random.uniform(0.8, 1.2)
        
        return features
    
    def _get_game_context(self, player_id: str, week: int, season: int) -> Dict:
        """Get game context features."""
        features = {}
        
        # Game spread (placeholder - would get from odds data)
        features['spread'] = np.random.uniform(-14, 14)
        features['game_total'] = np.random.uniform(38, 55)
        
        # Weather conditions (placeholder)
        features['temperature'] = np.random.uniform(30, 85)
        features['wind_speed'] = np.random.uniform(0, 20)
        features['is_dome'] = np.random.choice([0, 1])
        
        # Time slot
        features['is_primetime'] = np.random.choice([0, 1])
        features['is_division_game'] = np.random.choice([0, 1])
        
        return features
    
    def _get_prop_history(self, player_id: str, prop_type: str) -> Dict:
        """Get historical prop performance."""
        features = {}
        
        query = """
            SELECT line, actual_value, result
            FROM props
            WHERE player_id = ? AND prop_type = ?
            ORDER BY season DESC, week DESC
            LIMIT 10
        """
        
        cursor = self.conn.cursor()
        results = cursor.execute(query, (player_id, prop_type)).fetchall()
        
        if results:
            lines = [r[0] for r in results if r[0]]
            actuals = [r[1] for r in results if r[1]]
            
            if lines and actuals:
                # Historical hit rate
                overs = sum(1 for l, a in zip(lines, actuals) if a > l)
                features['prop_hit_rate'] = overs / len(lines)
                
                # Average distance from line
                features['avg_distance_from_line'] = np.mean([a - l for l, a in zip(lines, actuals)])
        else:
            features['prop_hit_rate'] = 0.5
            features['avg_distance_from_line'] = 0
        
        return features
    
    def create_training_dataset(self, prop_type: str, 
                               min_season: int = 2020) -> pd.DataFrame:
        """Create full training dataset for a prop type."""
        
        logger.info(f"Creating training dataset for {prop_type}")
        
        # Get all historical props of this type
        query = """
            SELECT p.*, ps.pass_yards, ps.pass_tds, ps.rush_yards, 
                   ps.receptions, ps.rec_yards
            FROM props p
            JOIN player_stats ps ON p.player_id = ps.player_id 
                AND p.week = ps.week AND p.season = ps.season
            WHERE p.prop_type = ? AND p.season >= ?
            AND p.actual_value IS NOT NULL
        """
        
        props_df = pd.read_sql_query(query, self.conn, 
                                    params=(prop_type, min_season))
        
        if props_df.empty:
            logger.warning(f"No data found for {prop_type}")
            return pd.DataFrame()
        
        # Create features for each prop
        all_features = []
        
        for _, row in props_df.iterrows():
            features = self.create_player_features(
                player_id=row['player_id'],
                prop_type=prop_type,
                week=row['week'],
                season=row['season']
            )
            
            if features:
                features['prop_id'] = row['prop_id']
                features['line'] = row['line']
                features['actual'] = row['actual_value']
                features['target'] = 1 if row['actual_value'] > row['line'] else 0
                
                all_features.append(features)
        
        if all_features:
            features_df = pd.DataFrame(all_features)
            
            # Handle missing values
            features_df = self._handle_missing_values(features_df)
            
            logger.info(f"Created {len(features_df)} samples for {prop_type}")
            return features_df
        
        return pd.DataFrame()
    
    def _handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values in features."""
        
        # Numeric columns: fill with median
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if df[col].isnull().any():
                df[col].fillna(df[col].median(), inplace=True)
        
        # Replace infinities
        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        df.fillna(0, inplace=True)
        
        return df
    
    def scale_features(self, df: pd.DataFrame, 
                      exclude_cols: List[str] = None) -> pd.DataFrame:
        """Scale features for model training."""
        
        if exclude_cols is None:
            exclude_cols = ['prop_id', 'target', 'line', 'actual']
        
        feature_cols = [col for col in df.columns if col not in exclude_cols]
        
        df_scaled = df.copy()
        df_scaled[feature_cols] = self.scaler.fit_transform(df[feature_cols])
        
        return df_scaled
    
    def create_live_features(self, player_id: str, prop_type: str,
                            line: float) -> np.ndarray:
        """Create features for live prediction."""
        
        # Get current week/season (placeholder)
        current_season = datetime.now().year
        current_week = (datetime.now() - datetime(current_season, 9, 1)).days // 7 + 1
        
        features = self.create_player_features(
            player_id=player_id,
            prop_type=prop_type,
            week=current_week,
            season=current_season
        )
        
        # Add line as feature
        features['line'] = line
        features['line_std'] = self._get_line_std(prop_type, line)
        
        # Convert to array and scale
        feature_names = sorted(features.keys())
        feature_array = np.array([features[f] for f in feature_names])
        
        # Scale using fitted scaler
        if hasattr(self.scaler, 'mean_'):
            feature_array = self.scaler.transform(feature_array.reshape(1, -1))
        
        return feature_array
    
    def _get_line_std(self, prop_type: str, line: float) -> float:
        """Get standardized line value for prop type."""
        
        # Get historical lines for this prop type
        query = """
            SELECT line FROM props 
            WHERE prop_type = ? AND line IS NOT NULL
        """
        
        cursor = self.conn.cursor()
        results = cursor.execute(query, (prop_type,)).fetchall()
        
        if results:
            lines = [r[0] for r in results]
            mean_line = np.mean(lines)
            std_line = np.std(lines)
            
            if std_line > 0:
                return (line - mean_line) / std_line
        
        return 0
    
    def get_feature_importance(self, model, feature_names: List[str]) -> pd.DataFrame:
        """Get feature importance from trained model."""
        
        if hasattr(model, 'feature_importances_'):
            importance_df = pd.DataFrame({
                'feature': feature_names,
                'importance': model.feature_importances_
            })
            
            return importance_df.sort_values('importance', ascending=False)
        
        return pd.DataFrame()
    
    def close(self):
        """Close database connection."""
        self.conn.close()


def main():
    """Example usage of feature engineering."""
    print("=" * 60)
    print("FEATURE ENGINEERING MODULE")
    print("=" * 60)
    
    # Initialize
    fe = FeatureEngineer(db_path="c:/Props_Project/data/nfl_props.db")
    
    # Example: Create features for a player
    print("\n1. Creating sample features for a player...")
    
    # Sample player features (would use real player_id)
    sample_features = fe.create_player_features(
        player_id="sample_player_001",
        prop_type="pass_yards",
        week=10,
        season=2024
    )
    
    if sample_features:
        print(f"   Created {len(sample_features)} features:")
        for key, value in list(sample_features.items())[:5]:
            print(f"     - {key}: {value:.2f}" if isinstance(value, float) else f"     - {key}: {value}")
    
    # Create training dataset (would need actual data in DB)
    print("\n2. Training dataset structure:")
    print("   Columns would include:")
    print("     - Player rolling averages (L3, L5, season)")
    print("     - Team offensive/defensive rankings")
    print("     - Game context (spread, total, weather)")
    print("     - Historical prop performance")
    print("     - Target variable (over/under)")
    
    print("\n3. Feature categories:")
    categories = {
        "Player Performance": ["L3 avg", "L5 avg", "Season avg", "Trend"],
        "Team Context": ["Team rank", "Opponent rank", "Home/Away"],
        "Game Context": ["Spread", "Total", "Weather", "Time slot"],
        "Prop History": ["Hit rate", "Avg distance from line"]
    }
    
    for category, features in categories.items():
        print(f"\n   {category}:")
        for feature in features:
            print(f"     - {feature}")
    
    # Clean up
    fe.close()
    
    print("\n" + "=" * 60)
    print("Feature engineering module ready!")


if __name__ == "__main__":
    main()