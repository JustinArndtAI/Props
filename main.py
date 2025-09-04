"""
NFL Player Props Predictor - Main Application
==============================================
A self-improving ML system for predicting NFL player props.
For research purposes only - not for actual betting.
"""

import sys
from pathlib import Path
import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
import logging

# Add project directories to path
sys.path.append(str(Path(__file__).parent))

from utils.round_robin_calculator import RoundRobinCalculator
from scripts.data_acquisition import NFLDataAcquisition
from scripts.feature_engineering import FeatureEngineer
from scripts.parlay_generator import ParlayGenerator, EVCalculator
from models.prop_predictor import XGBoostPropPredictor, EnsemblePropPredictor

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PropsPredictor:
    """Main application class for NFL props prediction."""
    
    def __init__(self):
        self.data_acq = None
        self.feature_eng = None
        self.models = {}
        self.parlay_gen = ParlayGenerator()
        self.rr_calc = RoundRobinCalculator()
        self.ev_calc = EVCalculator()
        
    def initialize(self):
        """Initialize all components."""
        logger.info("Initializing Props Predictor System...")
        
        # Initialize data acquisition
        self.data_acq = NFLDataAcquisition(
            db_path="c:/Props_Project/data/nfl_props.db"
        )
        
        # Initialize feature engineering
        self.feature_eng = FeatureEngineer(
            db_path="c:/Props_Project/data/nfl_props.db"
        )
        
        # Load or train models
        self.load_models()
        
        # Set up parlay generator
        self.parlay_gen.set_models(self.models)
        self.parlay_gen.set_feature_engineer(self.feature_eng)
        
        logger.info("System initialized successfully")
    
    def load_models(self):
        """Load pre-trained models or create new ones."""
        prop_types = ['pass_yards', 'pass_tds', 'rush_yards', 'receptions', 'rec_yards']
        
        for prop_type in prop_types:
            model_path = Path(f"c:/Props_Project/models/saved/{prop_type}_model")
            
            if model_path.with_suffix('_xgb.pkl').exists():
                # Load existing model
                model = XGBoostPropPredictor(prop_type)
                model.load(str(model_path))
                self.models[prop_type] = model
                logger.info(f"Loaded model for {prop_type}")
            else:
                # Create new model (will train when data available)
                self.models[prop_type] = XGBoostPropPredictor(prop_type)
                logger.info(f"Created new model for {prop_type}")
    
    def train_models(self):
        """Train all models on historical data."""
        logger.info("Training models on historical data...")
        
        for prop_type, model in self.models.items():
            # Create training dataset
            train_data = self.feature_eng.create_training_dataset(prop_type)
            
            if not train_data.empty:
                # Prepare features and target
                feature_cols = [col for col in train_data.columns 
                              if col not in ['prop_id', 'target', 'line', 'actual']]
                
                X = train_data[feature_cols]
                y = train_data['target']
                
                # Train model
                from sklearn.model_selection import train_test_split
                X_train, X_val, y_train, y_val = train_test_split(
                    X, y, test_size=0.2, random_state=42
                )
                
                model.train(X_train, y_train, X_val, y_val, optimize_hyperparams=False)
                
                # Save model
                model_path = f"c:/Props_Project/models/saved/{prop_type}_model"
                Path("c:/Props_Project/models/saved").mkdir(parents=True, exist_ok=True)
                model.save(model_path)
                
                logger.info(f"Trained and saved model for {prop_type}")
    
    def weekly_update(self):
        """Perform weekly data update and model retraining."""
        logger.info("Starting weekly update...")
        
        # Update data
        self.data_acq.update_weekly_data()
        
        # Retrain models with new data
        # (Implement continual learning here)
        
        logger.info("Weekly update completed")
    
    def get_week_predictions(self, week: int = None) -> pd.DataFrame:
        """Get predictions for current/specified week."""
        
        # Get available props (placeholder)
        available_props = [
            {
                'player': 'Patrick Mahomes',
                'player_id': 'mahomes_001',
                'prop_type': 'pass_yards',
                'line': 275.5,
                'odds': -110
            },
            {
                'player': 'Josh Allen',
                'player_id': 'allen_001',
                'prop_type': 'pass_yards',
                'line': 265.5,
                'odds': -115
            },
            # Add more props...
        ]
        
        # Get predictions
        props_with_pred = self.parlay_gen.get_prop_predictions(available_props)
        
        # Rank by EV
        props_ranked = self.ev_calc.rank_props_by_ev(props_with_pred)
        
        return pd.DataFrame(props_ranked)
    
    def close(self):
        """Clean up resources."""
        if self.data_acq:
            self.data_acq.close()
        if self.feature_eng:
            self.feature_eng.close()


def create_streamlit_app():
    """Create Streamlit web interface."""
    
    st.set_page_config(
        page_title="NFL Props Predictor",
        page_icon="🏈",
        layout="wide"
    )
    
    st.title("🏈 NFL Player Props Predictor")
    st.markdown("*For research purposes only - not for actual betting*")
    
    # Initialize predictor
    if 'predictor' not in st.session_state:
        with st.spinner("Initializing system..."):
            predictor = PropsPredictor()
            predictor.initialize()
            st.session_state.predictor = predictor
    
    predictor = st.session_state.predictor
    
    # Sidebar
    st.sidebar.header("Settings")
    
    risk_tolerance = st.sidebar.selectbox(
        "Risk Tolerance",
        ["conservative", "moderate", "aggressive"],
        index=1
    )
    
    bankroll = st.sidebar.number_input(
        "Bankroll ($)",
        min_value=10.0,
        max_value=10000.0,
        value=100.0,
        step=10.0
    )
    
    # Main content tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 Predictions", 
        "🎰 Round Robins", 
        "💰 EV Calculator",
        "📈 Performance",
        "🔧 Tools"
    ])
    
    with tab1:
        st.header("This Week's Prop Predictions")
        
        col1, col2 = st.columns(2)
        
        with col1:
            prop_type = st.selectbox(
                "Prop Type",
                ["pass_yards", "pass_tds", "rush_yards", "receptions", "rec_yards"]
            )
        
        with col2:
            min_ev = st.slider(
                "Minimum EV Threshold",
                min_value=0.0,
                max_value=0.20,
                value=0.03,
                step=0.01,
                format="%.2f"
            )
        
        if st.button("Get Predictions"):
            with st.spinner("Generating predictions..."):
                predictions_df = predictor.get_week_predictions()
                
                if not predictions_df.empty:
                    # Filter by EV
                    filtered = predictions_df[predictions_df['ev'] >= min_ev]
                    
                    st.subheader(f"Top {len(filtered)} Props (EV >= {min_ev:.2f})")
                    
                    # Display table
                    display_cols = ['player', 'prop_type', 'line', 'odds', 'probability', 'ev']
                    st.dataframe(
                        filtered[display_cols].head(10),
                        use_container_width=True
                    )
                    
                    # Visualizations
                    import plotly.express as px
                    
                    fig = px.scatter(
                        filtered.head(20),
                        x='probability',
                        y='ev',
                        size='kelly_bet',
                        color='prop_type',
                        hover_data=['player', 'line'],
                        title="Probability vs Expected Value"
                    )
                    st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        st.header("Round Robin Generator")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            num_legs = st.slider("Number of Legs", 4, 12, 8)
        
        with col2:
            parlay_size = st.slider("Parlay Size", 2, 6, 4)
        
        with col3:
            bet_per_parlay = st.number_input(
                "Bet per Parlay ($)",
                min_value=0.10,
                max_value=10.00,
                value=0.10,
                step=0.10
            )
        
        if st.button("Calculate Round Robin"):
            # Calculate round robin
            rr_result = predictor.rr_calc.calculate_round_robin(
                num_legs=num_legs,
                parlay_size=parlay_size,
                bet_per_parlay=bet_per_parlay
            )
            
            # Display results
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Total Parlays", rr_result['num_parlays'])
            
            with col2:
                st.metric("Total Cost", f"${rr_result['total_cost']:.2f}")
            
            with col3:
                st.metric("Max Payout", f"${rr_result['max_payout']:.2f}")
            
            with col4:
                st.metric("Max ROI", f"{rr_result['roi_if_all_hit']:.1f}%")
            
            # Profitability simulation
            st.subheader("Profitability Simulation")
            
            win_rate = st.slider(
                "Expected Win Rate per Prop",
                min_value=0.50,
                max_value=0.80,
                value=0.65,
                step=0.01,
                format="%.2f"
            )
            
            if st.button("Run Simulation"):
                sim_result = predictor.rr_calc.simulate_profitability(
                    num_legs=num_legs,
                    parlay_size=parlay_size,
                    prop_probabilities=[win_rate] * num_legs,
                    odds_per_leg=[-110] * num_legs,
                    bet_per_parlay=bet_per_parlay,
                    num_simulations=10000
                )
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Expected Profit", f"${sim_result['expected_profit']:.2f}")
                
                with col2:
                    st.metric("Win Rate", f"{sim_result['win_rate']:.1f}%")
                
                with col3:
                    st.metric("Expected ROI", f"{sim_result['expected_roi']:.1f}%")
                
                # Distribution chart
                st.info(f"25th Percentile: ${sim_result['percentile_25']:.2f} | "
                       f"Median: ${sim_result['median_profit']:.2f} | "
                       f"75th Percentile: ${sim_result['percentile_75']:.2f}")
    
    with tab3:
        st.header("Expected Value Calculator")
        
        col1, col2 = st.columns(2)
        
        with col1:
            odds = st.number_input(
                "American Odds",
                min_value=-500,
                max_value=500,
                value=-110,
                step=5
            )
        
        with col2:
            probability = st.slider(
                "Win Probability",
                min_value=0.0,
                max_value=1.0,
                value=0.55,
                step=0.01,
                format="%.2f"
            )
        
        # Calculate EV
        ev = predictor.ev_calc.calculate_prop_ev(probability, odds)
        kelly = predictor.ev_calc.calculate_kelly_criterion(probability, odds)
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Expected Value", f"{ev:.3f}")
        
        with col2:
            st.metric("Kelly Bet Size", f"{kelly*100:.1f}%")
        
        with col3:
            decimal_odds = predictor.rr_calc.american_to_decimal(odds)
            st.metric("Decimal Odds", f"{decimal_odds:.2f}")
        
        # EV explanation
        if ev > 0:
            st.success(f"✅ Positive EV! Expected return of {ev*100:.1f}% per dollar bet")
        else:
            st.error(f"❌ Negative EV. Expected loss of {abs(ev)*100:.1f}% per dollar bet")
    
    with tab4:
        st.header("Performance Tracking")
        st.info("Performance tracking will be available after placing predictions")
        
        # Placeholder for performance metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Season Accuracy", "Coming Soon")
        
        with col2:
            st.metric("Last Week ROI", "Coming Soon")
        
        with col3:
            st.metric("Total Profit", "Coming Soon")
        
        with col4:
            st.metric("Best Prop Type", "Coming Soon")
    
    with tab5:
        st.header("System Tools")
        
        if st.button("📥 Update Data"):
            with st.spinner("Updating data..."):
                predictor.weekly_update()
                st.success("Data updated successfully!")
        
        if st.button("🎯 Train Models"):
            with st.spinner("Training models..."):
                predictor.train_models()
                st.success("Models trained successfully!")
        
        if st.button("📊 Generate Weekly Report"):
            st.info("Weekly report generation coming soon!")
    
    # Footer
    st.markdown("---")
    st.markdown(
        """
        **Disclaimer:** This tool is for educational and research purposes only. 
        Sports betting involves risk and may not be legal in your jurisdiction. 
        Never bet more than you can afford to lose.
        """
    )


def main():
    """Main entry point."""
    
    import argparse
    
    parser = argparse.ArgumentParser(description='NFL Props Predictor')
    parser.add_argument('--mode', choices=['cli', 'web'], default='web',
                       help='Run mode: cli or web')
    parser.add_argument('--train', action='store_true',
                       help='Train models on historical data')
    parser.add_argument('--update', action='store_true',
                       help='Perform weekly update')
    
    args = parser.parse_args()
    
    if args.mode == 'web':
        # Run Streamlit app
        create_streamlit_app()
    else:
        # CLI mode
        print("=" * 60)
        print("NFL PLAYER PROPS PREDICTOR")
        print("=" * 60)
        
        predictor = PropsPredictor()
        predictor.initialize()
        
        if args.train:
            print("\nTraining models...")
            predictor.train_models()
        
        if args.update:
            print("\nPerforming weekly update...")
            predictor.weekly_update()
        
        # Get predictions
        print("\nGenerating predictions...")
        predictions = predictor.get_week_predictions()
        
        if not predictions.empty:
            print("\nTop 5 Props by EV:")
            print(predictions[['player', 'prop_type', 'line', 'odds', 'probability', 'ev']].head())
        
        predictor.close()
        print("\n" + "=" * 60)
        print("Complete!")


if __name__ == "__main__":
    main()