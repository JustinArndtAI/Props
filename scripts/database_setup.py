"""
Setup and verify SQLite database with all necessary tables
Including model versioning and wandb integration
"""

import sqlite3
from pathlib import Path
import datetime

def setup_database():
    """Create or verify all database tables."""
    
    db_path = Path("C:/Props_Project/data/nfl_props.db")
    db_path.parent.mkdir(exist_ok=True)
    
    conn = sqlite3.connect(str(db_path))
    cursor = conn.cursor()
    
    print("="*60)
    print("Setting up SQLite Database")
    print("="*60)
    
    # 1. Player Stats table (already exists from previous work)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS player_stats (
            stat_id INTEGER PRIMARY KEY AUTOINCREMENT,
            player_id TEXT,
            player_name TEXT,
            position TEXT,
            team TEXT,
            week INTEGER,
            season INTEGER,
            game_date TEXT,
            pass_yards REAL,
            pass_tds INTEGER,
            pass_completions INTEGER,
            pass_attempts INTEGER,
            rush_yards REAL,
            rush_attempts INTEGER,
            rush_tds INTEGER,
            receptions INTEGER,
            targets INTEGER,
            rec_yards REAL,
            rec_tds INTEGER,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    print("[CREATED/VERIFIED] player_stats table")
    
    # 2. Prop Outcomes table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS prop_outcomes (
            outcome_id INTEGER PRIMARY KEY AUTOINCREMENT,
            player_name TEXT,
            prop_type TEXT,
            week INTEGER,
            season INTEGER,
            line REAL,
            actual_value REAL,
            hit_over INTEGER,
            odds_over INTEGER,
            odds_under INTEGER,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    print("[CREATED/VERIFIED] prop_outcomes table")
    
    # 3. Features table for ML
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS features (
            feature_id INTEGER PRIMARY KEY AUTOINCREMENT,
            player_name TEXT,
            prop_type TEXT,
            week INTEGER,
            season INTEGER,
            feature_name TEXT,
            feature_value REAL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    print("[CREATED/VERIFIED] features table")
    
    # 4. Model Versions table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS model_versions (
            version_id INTEGER PRIMARY KEY AUTOINCREMENT,
            model_name TEXT,
            prop_type TEXT,
            version TEXT,
            accuracy REAL,
            auc_score REAL,
            training_date TIMESTAMP,
            model_path TEXT,
            wandb_run_id TEXT,
            hyperparameters TEXT,
            notes TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    print("[CREATED/VERIFIED] model_versions table")
    
    # 5. Predictions table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS predictions (
            prediction_id INTEGER PRIMARY KEY AUTOINCREMENT,
            model_version_id INTEGER,
            player_name TEXT,
            prop_type TEXT,
            week INTEGER,
            season INTEGER,
            line REAL,
            predicted_prob REAL,
            predicted_outcome INTEGER,
            actual_outcome INTEGER,
            ev_score REAL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (model_version_id) REFERENCES model_versions(version_id)
        )
    """)
    print("[CREATED/VERIFIED] predictions table")
    
    # 6. Round Robin Results table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS round_robin_results (
            rr_id INTEGER PRIMARY KEY AUTOINCREMENT,
            config_name TEXT,
            num_legs INTEGER,
            parlay_size INTEGER,
            num_parlays INTEGER,
            total_cost REAL,
            expected_profit REAL,
            expected_roi REAL,
            actual_profit REAL,
            actual_roi REAL,
            props_json TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    print("[CREATED/VERIFIED] round_robin_results table")
    
    conn.commit()
    
    # List all tables
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = cursor.fetchall()
    
    print("\n" + "-"*40)
    print("Database tables:")
    for table in tables:
        cursor.execute(f"SELECT COUNT(*) FROM {table[0]}")
        count = cursor.fetchone()[0]
        print(f"  - {table[0]}: {count} records")
    
    conn.close()
    
    print("\n" + "="*60)
    print("Database setup complete!")
    print("Location: C:/Props_Project/data/nfl_props.db")
    print("="*60)
    
    return str(db_path)

if __name__ == "__main__":
    db_path = setup_database()