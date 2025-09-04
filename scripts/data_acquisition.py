import nfl_data_py as nfl
import pandas as pd
import numpy as np
import requests
from bs4 import BeautifulSoup
import sqlite3
import json
from datetime import datetime, timedelta
import time
from typing import Dict, List, Optional, Tuple
import logging
from pathlib import Path

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class NFLDataAcquisition:
    """Handles all data acquisition for NFL player props prediction."""
    
    def __init__(self, db_path: str = "data/nfl_props.db"):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(self.db_path))
        self.setup_database()
    
    def setup_database(self):
        """Create database tables if they don't exist."""
        cursor = self.conn.cursor()
        
        # Players table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS players (
                player_id TEXT PRIMARY KEY,
                name TEXT,
                position TEXT,
                team TEXT,
                updated_at TIMESTAMP
            )
        """)
        
        # Player stats table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS player_stats (
                stat_id INTEGER PRIMARY KEY AUTOINCREMENT,
                player_id TEXT,
                game_id TEXT,
                week INTEGER,
                season INTEGER,
                pass_yards REAL,
                pass_tds INTEGER,
                rush_yards REAL,
                receptions INTEGER,
                rec_yards REAL,
                targets INTEGER,
                snap_count INTEGER,
                FOREIGN KEY (player_id) REFERENCES players(player_id)
            )
        """)
        
        # Props table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS props (
                prop_id INTEGER PRIMARY KEY AUTOINCREMENT,
                player_id TEXT,
                game_id TEXT,
                prop_type TEXT,
                line REAL,
                over_odds INTEGER,
                under_odds INTEGER,
                actual_value REAL,
                result TEXT,
                week INTEGER,
                season INTEGER,
                created_at TIMESTAMP,
                FOREIGN KEY (player_id) REFERENCES players(player_id)
            )
        """)
        
        # Games table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS games (
                game_id TEXT PRIMARY KEY,
                season INTEGER,
                week INTEGER,
                home_team TEXT,
                away_team TEXT,
                home_score INTEGER,
                away_score INTEGER,
                weather TEXT,
                temperature REAL,
                wind_speed REAL,
                game_date TIMESTAMP
            )
        """)
        
        self.conn.commit()
        logger.info("Database tables created/verified")
    
    def fetch_historical_stats(self, years: List[int] = None) -> pd.DataFrame:
        """Fetch historical NFL play-by-play and player stats."""
        if years is None:
            years = list(range(2020, 2025))
        
        logger.info(f"Fetching NFL data for years: {years}")
        
        all_data = []
        for year in years:
            try:
                # Fetch play-by-play data
                pbp_data = nfl.import_pbp_data([year])
                
                # Fetch weekly player stats
                weekly_data = nfl.import_weekly_data([year])
                
                # Fetch player IDs
                ids = nfl.import_ids()
                
                all_data.append({
                    'year': year,
                    'pbp': pbp_data,
                    'weekly': weekly_data,
                    'ids': ids
                })
                
                logger.info(f"Successfully fetched data for {year}")
                time.sleep(1)  # Rate limiting
                
            except Exception as e:
                logger.error(f"Error fetching data for {year}: {e}")
        
        return all_data
    
    def process_player_stats(self, data: List[Dict]) -> pd.DataFrame:
        """Process raw data into player stats format."""
        player_stats = []
        
        for year_data in data:
            year = year_data['year']
            weekly = year_data['weekly']
            
            if weekly is not None and not weekly.empty:
                # Select relevant columns
                stat_cols = [
                    'player_id', 'player_name', 'position', 'recent_team',
                    'week', 'completions', 'attempts', 'passing_yards', 
                    'passing_tds', 'interceptions', 'rushing_yards',
                    'rushing_tds', 'receptions', 'targets', 'receiving_yards',
                    'receiving_tds'
                ]
                
                available_cols = [col for col in stat_cols if col in weekly.columns]
                stats_df = weekly[available_cols].copy()
                stats_df['season'] = year
                
                player_stats.append(stats_df)
        
        if player_stats:
            combined_stats = pd.concat(player_stats, ignore_index=True)
            
            # Store in database
            self.store_player_stats(combined_stats)
            
            return combined_stats
        
        return pd.DataFrame()
    
    def store_player_stats(self, df: pd.DataFrame):
        """Store player stats in database."""
        cursor = self.conn.cursor()
        
        for _, row in df.iterrows():
            # Insert/update player
            cursor.execute("""
                INSERT OR REPLACE INTO players (player_id, name, position, team, updated_at)
                VALUES (?, ?, ?, ?, ?)
            """, (
                row.get('player_id'),
                row.get('player_name'),
                row.get('position'),
                row.get('recent_team'),
                datetime.now()
            ))
            
            # Insert stats
            cursor.execute("""
                INSERT INTO player_stats (
                    player_id, week, season, pass_yards, pass_tds,
                    rush_yards, receptions, rec_yards, targets
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                row.get('player_id'),
                row.get('week'),
                row.get('season'),
                row.get('passing_yards'),
                row.get('passing_tds'),
                row.get('rushing_yards'),
                row.get('receptions'),
                row.get('receiving_yards'),
                row.get('targets')
            ))
        
        self.conn.commit()
        logger.info(f"Stored {len(df)} player stat records")
    
    def fetch_current_week_data(self) -> Dict:
        """Fetch current week NFL data including injuries and weather."""
        current_data = {}
        
        try:
            # Get current season schedule
            schedule = nfl.import_schedules([2024])
            
            # Get current week
            current_week = schedule[schedule['gameday'] >= datetime.now().date()].iloc[0]['week']
            
            # Fetch injuries
            injuries = self.fetch_injury_data()
            
            # Fetch weather (would need API key)
            weather = self.fetch_weather_data()
            
            current_data = {
                'week': current_week,
                'schedule': schedule[schedule['week'] == current_week],
                'injuries': injuries,
                'weather': weather
            }
            
        except Exception as e:
            logger.error(f"Error fetching current week data: {e}")
        
        return current_data
    
    def fetch_injury_data(self) -> pd.DataFrame:
        """Fetch current injury reports."""
        try:
            # This would typically scrape from a source or use an API
            # For now, returning empty DataFrame as placeholder
            logger.info("Fetching injury data (placeholder)")
            return pd.DataFrame()
        except Exception as e:
            logger.error(f"Error fetching injuries: {e}")
            return pd.DataFrame()
    
    def fetch_weather_data(self) -> pd.DataFrame:
        """Fetch weather data for upcoming games."""
        try:
            # This would typically use a weather API
            # For now, returning empty DataFrame as placeholder
            logger.info("Fetching weather data (placeholder)")
            return pd.DataFrame()
        except Exception as e:
            logger.error(f"Error fetching weather: {e}")
            return pd.DataFrame()
    
    def scrape_prop_odds(self, week: int = None) -> List[Dict]:
        """
        Scrape current prop odds from sportsbooks.
        NOTE: This is a placeholder - actual implementation would need
        to comply with website terms of service.
        """
        props = []
        
        logger.warning("Prop odds scraping is placeholder - implement with caution")
        
        # Example structure of what would be returned
        example_props = [
            {
                'player': 'Patrick Mahomes',
                'prop_type': 'pass_yards',
                'line': 275.5,
                'over_odds': -110,
                'under_odds': -110,
                'sportsbook': 'example'
            },
            {
                'player': 'Travis Kelce',
                'prop_type': 'receptions',
                'line': 5.5,
                'over_odds': -120,
                'under_odds': +100,
                'sportsbook': 'example'
            }
        ]
        
        return example_props
    
    def fetch_historical_props(self, source: str = None) -> pd.DataFrame:
        """
        Fetch historical prop lines and outcomes.
        This would typically come from a paid data source or careful scraping.
        """
        logger.info("Fetching historical props (placeholder)")
        
        # Placeholder for historical props
        # In reality, this would fetch from a reliable source
        return pd.DataFrame()
    
    def update_weekly_data(self):
        """Weekly update routine for all data sources."""
        logger.info("Starting weekly data update")
        
        try:
            # Fetch latest stats
            current_year = datetime.now().year
            recent_data = self.fetch_historical_stats([current_year])
            
            # Process and store
            if recent_data:
                self.process_player_stats(recent_data)
            
            # Fetch current week specifics
            current_week = self.fetch_current_week_data()
            
            # Fetch prop odds
            props = self.scrape_prop_odds()
            
            logger.info("Weekly data update completed")
            
        except Exception as e:
            logger.error(f"Error in weekly update: {e}")
    
    def get_player_recent_stats(self, player_id: str, num_weeks: int = 5) -> pd.DataFrame:
        """Get recent stats for a specific player."""
        query = """
            SELECT * FROM player_stats
            WHERE player_id = ?
            ORDER BY season DESC, week DESC
            LIMIT ?
        """
        
        return pd.read_sql_query(query, self.conn, params=(player_id, num_weeks))
    
    def close(self):
        """Close database connection."""
        self.conn.close()


def main():
    """Example usage of data acquisition."""
    print("=" * 60)
    print("NFL DATA ACQUISITION MODULE")
    print("=" * 60)
    
    # Initialize
    data_acq = NFLDataAcquisition(db_path="c:/Props_Project/data/nfl_props.db")
    
    print("\n1. Fetching historical data (2023-2024)...")
    historical_data = data_acq.fetch_historical_stats([2023, 2024])
    
    if historical_data:
        print(f"   Fetched data for {len(historical_data)} seasons")
        
        print("\n2. Processing player stats...")
        stats_df = data_acq.process_player_stats(historical_data)
        
        if not stats_df.empty:
            print(f"   Processed {len(stats_df)} player stat records")
            print(f"   Unique players: {stats_df['player_id'].nunique()}")
            print(f"   Date range: {stats_df['season'].min()}-{stats_df['season'].max()}")
    
    print("\n3. Fetching current week data...")
    current_data = data_acq.fetch_current_week_data()
    if current_data:
        print(f"   Current week: {current_data.get('week', 'Unknown')}")
    
    print("\n4. Example prop odds (placeholder)...")
    props = data_acq.scrape_prop_odds()
    for prop in props[:3]:
        print(f"   {prop['player']}: {prop['prop_type']} O/U {prop['line']}")
    
    # Clean up
    data_acq.close()
    
    print("\n" + "=" * 60)
    print("Data acquisition setup complete!")
    print("Database created at: c:/Props_Project/data/nfl_props.db")


if __name__ == "__main__":
    main()