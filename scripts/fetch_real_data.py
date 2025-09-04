"""
Fetch and process REAL NFL data for training prop prediction models.
This script will actually download and store NFL data.
"""

import nfl_data_py as nfl
import pandas as pd
import numpy as np
import sqlite3
from datetime import datetime
import logging
from pathlib import Path
import time

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

def fetch_and_store_nfl_data():
    """Fetch real NFL data and store in database."""
    
    # Create database connection
    db_path = Path("c:/Props_Project/data/nfl_props.db")
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path))
    
    logger.info("Starting NFL data fetch...")
    
    # 1. Fetch weekly player data for recent seasons
    logger.info("Fetching weekly player stats 2022-2024...")
    try:
        weekly_data = nfl.import_weekly_data([2022, 2023, 2024])
        logger.info(f"Fetched {len(weekly_data)} weekly stat records")
        
        # Save to database
        weekly_data.to_sql('weekly_stats_raw', conn, if_exists='replace', index=False)
        
        # Show sample
        print("\nSample Weekly Stats:")
        print(weekly_data[['player_display_name', 'position', 'week', 'season', 
                           'passing_yards', 'rushing_yards', 'receiving_yards']].head(10))
        
    except Exception as e:
        logger.error(f"Error fetching weekly data: {e}")
        return
    
    # 2. Process into prop-relevant format
    logger.info("Processing player stats for props...")
    
    # Create processed stats table
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS player_prop_stats (
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
            PRIMARY KEY (player_name, week, season)
        )
    """)
    
    # Filter for relevant positions and stats
    qb_stats = weekly_data[weekly_data['position'] == 'QB'].copy()
    rb_stats = weekly_data[weekly_data['position'] == 'RB'].copy()
    wr_stats = weekly_data[weekly_data['position'] == 'WR'].copy()
    te_stats = weekly_data[weekly_data['position'] == 'TE'].copy()
    
    all_stats = pd.concat([qb_stats, rb_stats, wr_stats, te_stats])
    
    # Standardize column names
    prop_stats = pd.DataFrame({
        'player_name': all_stats['player_display_name'],
        'position': all_stats['position'],
        'team': all_stats['recent_team'],
        'week': all_stats['week'],
        'season': all_stats['season'],
        'game_date': pd.to_datetime('today').strftime('%Y-%m-%d'),  # Placeholder
        'pass_yards': all_stats.get('passing_yards', 0),
        'pass_tds': all_stats.get('passing_tds', 0),
        'pass_completions': all_stats.get('completions', 0),
        'pass_attempts': all_stats.get('attempts', 0),
        'rush_yards': all_stats.get('rushing_yards', 0),
        'rush_attempts': all_stats.get('carries', 0),
        'rush_tds': all_stats.get('rushing_tds', 0),
        'receptions': all_stats.get('receptions', 0),
        'targets': all_stats.get('targets', 0),
        'rec_yards': all_stats.get('receiving_yards', 0),
        'rec_tds': all_stats.get('receiving_tds', 0)
    })
    
    # Clean data
    prop_stats = prop_stats.fillna(0)
    
    # Store in database
    prop_stats.to_sql('player_prop_stats', conn, if_exists='replace', index=False)
    logger.info(f"Stored {len(prop_stats)} player stat records")
    
    # 3. Calculate aggregated stats for top players
    logger.info("Calculating season averages for top players...")
    
    # Get top QBs by pass yards
    top_qbs = prop_stats[prop_stats['position'] == 'QB'].groupby('player_name').agg({
        'pass_yards': 'mean',
        'pass_tds': 'mean',
        'week': 'count'
    }).sort_values('pass_yards', ascending=False).head(20)
    
    print("\nTop 20 QBs by Average Pass Yards:")
    print(top_qbs)
    
    # Get top RBs by rush yards
    top_rbs = prop_stats[prop_stats['position'] == 'RB'].groupby('player_name').agg({
        'rush_yards': 'mean',
        'receptions': 'mean',
        'week': 'count'
    }).sort_values('rush_yards', ascending=False).head(20)
    
    print("\nTop 20 RBs by Average Rush Yards:")
    print(top_rbs)
    
    # Get top WRs by receiving yards
    top_wrs = prop_stats[prop_stats['position'] == 'WR'].groupby('player_name').agg({
        'rec_yards': 'mean',
        'receptions': 'mean',
        'targets': 'mean',
        'week': 'count'
    }).sort_values('rec_yards', ascending=False).head(20)
    
    print("\nTop 20 WRs by Average Receiving Yards:")
    print(top_wrs)
    
    # 4. Create sample prop lines based on averages
    logger.info("Creating synthetic prop lines for training...")
    
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS prop_lines (
            player_name TEXT,
            prop_type TEXT,
            week INTEGER,
            season INTEGER,
            line REAL,
            actual_result REAL,
            hit_over INTEGER,
            PRIMARY KEY (player_name, prop_type, week, season)
        )
    """)
    
    # Generate prop lines based on historical averages with variance
    prop_lines = []
    
    for _, row in prop_stats.iterrows():
        player = row['player_name']
        week = row['week']
        season = row['season']
        position = row['position']
        
        # QB props
        if position == 'QB' and row['pass_attempts'] > 10:
            # Pass yards line (avg - 5 to avg + 5)
            if row['pass_yards'] > 0:
                line = row['pass_yards'] + np.random.uniform(-15, 15)
                prop_lines.append({
                    'player_name': player,
                    'prop_type': 'pass_yards',
                    'week': week,
                    'season': season,
                    'line': round(line, 1),
                    'actual_result': row['pass_yards'],
                    'hit_over': 1 if row['pass_yards'] > line else 0
                })
            
            # Pass TDs line
            if row['pass_tds'] >= 0:
                line = max(0.5, row['pass_tds'] + np.random.uniform(-0.5, 0.5))
                prop_lines.append({
                    'player_name': player,
                    'prop_type': 'pass_tds',
                    'week': week,
                    'season': season,
                    'line': round(line, 1),
                    'actual_result': row['pass_tds'],
                    'hit_over': 1 if row['pass_tds'] > line else 0
                })
        
        # RB props
        if position == 'RB' and row['rush_attempts'] > 5:
            # Rush yards
            if row['rush_yards'] > 0:
                line = row['rush_yards'] + np.random.uniform(-10, 10)
                prop_lines.append({
                    'player_name': player,
                    'prop_type': 'rush_yards',
                    'week': week,
                    'season': season,
                    'line': round(line, 1),
                    'actual_result': row['rush_yards'],
                    'hit_over': 1 if row['rush_yards'] > line else 0
                })
        
        # WR/TE props
        if position in ['WR', 'TE'] and row['targets'] > 2:
            # Receptions
            if row['receptions'] > 0:
                line = row['receptions'] + np.random.uniform(-1, 1)
                prop_lines.append({
                    'player_name': player,
                    'prop_type': 'receptions',
                    'week': week,
                    'season': season,
                    'line': round(line, 1),
                    'actual_result': row['receptions'],
                    'hit_over': 1 if row['receptions'] > line else 0
                })
            
            # Receiving yards
            if row['rec_yards'] > 0:
                line = row['rec_yards'] + np.random.uniform(-10, 10)
                prop_lines.append({
                    'player_name': player,
                    'prop_type': 'rec_yards',
                    'week': week,
                    'season': season,
                    'line': round(line, 1),
                    'actual_result': row['rec_yards'],
                    'hit_over': 1 if row['rec_yards'] > line else 0
                })
    
    if prop_lines:
        prop_lines_df = pd.DataFrame(prop_lines)
        prop_lines_df.to_sql('prop_lines', conn, if_exists='replace', index=False)
        logger.info(f"Created {len(prop_lines_df)} synthetic prop lines")
        
        # Show hit rates
        hit_rates = prop_lines_df.groupby('prop_type')['hit_over'].agg(['mean', 'count'])
        print("\nProp Type Hit Rates:")
        print(hit_rates)
    
    # 5. Get schedule data for context
    logger.info("Fetching schedule data...")
    try:
        schedules = nfl.import_schedules([2022, 2023, 2024])
        schedules.to_sql('schedules', conn, if_exists='replace', index=False)
        logger.info(f"Stored {len(schedules)} schedule records")
    except Exception as e:
        logger.error(f"Error fetching schedules: {e}")
    
    conn.commit()
    conn.close()
    
    logger.info("Data fetch complete!")
    print("\n" + "="*60)
    print("DATABASE POPULATED WITH REAL NFL DATA!")
    print("Ready for model training")
    print("="*60)
    
    return True

if __name__ == "__main__":
    success = fetch_and_store_nfl_data()
    if success:
        print("\nNext steps:")
        print("1. Run feature engineering on this data")
        print("2. Train models on historical props")
        print("3. Generate predictions for current week")