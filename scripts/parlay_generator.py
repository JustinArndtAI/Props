import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from itertools import combinations
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from utils.round_robin_calculator import RoundRobinCalculator
from models.prop_predictor import XGBoostPropPredictor, EnsemblePropPredictor
from scripts.feature_engineering import FeatureEngineer
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EVCalculator:
    """Calculate Expected Value for props and parlays."""
    
    def __init__(self):
        self.rr_calc = RoundRobinCalculator()
    
    def calculate_prop_ev(self, probability: float, odds: int) -> float:
        """
        Calculate EV for a single prop.
        
        Args:
            probability: Win probability (0-1)
            odds: American odds
        
        Returns:
            Expected value (positive = profitable)
        """
        decimal_odds = self.rr_calc.american_to_decimal(odds)
        ev = (probability * decimal_odds) - 1
        return ev
    
    def calculate_kelly_criterion(self, probability: float, odds: int, 
                                 kelly_fraction: float = 0.25) -> float:
        """
        Calculate optimal bet size using Kelly Criterion.
        
        Args:
            probability: Win probability
            odds: American odds
            kelly_fraction: Fraction of Kelly to use (default 0.25 for safety)
        
        Returns:
            Optimal bet size as fraction of bankroll
        """
        decimal_odds = self.rr_calc.american_to_decimal(odds)
        b = decimal_odds - 1
        p = probability
        q = 1 - p
        
        # Kelly formula: f = (bp - q) / b
        kelly_full = (b * p - q) / b
        
        # Use fractional Kelly for safety
        if kelly_full > 0:
            return kelly_full * kelly_fraction
        return 0
    
    def rank_props_by_ev(self, props: List[Dict]) -> List[Dict]:
        """
        Rank props by expected value.
        
        Args:
            props: List of prop dictionaries with 'probability' and 'odds'
        
        Returns:
            Sorted list by EV (descending)
        """
        for prop in props:
            prop['ev'] = self.calculate_prop_ev(
                prop['probability'], 
                prop['odds']
            )
            prop['kelly_bet'] = self.calculate_kelly_criterion(
                prop['probability'],
                prop['odds']
            )
        
        return sorted(props, key=lambda x: x['ev'], reverse=True)


class ParlayGenerator:
    """Generate optimal parlays from available props."""
    
    def __init__(self, models: Dict = None):
        self.models = models if models else {}
        self.ev_calc = EVCalculator()
        self.rr_calc = RoundRobinCalculator()
        self.feature_engineer = None
    
    def set_models(self, models: Dict):
        """Set prediction models."""
        self.models = models
    
    def set_feature_engineer(self, fe: FeatureEngineer):
        """Set feature engineer."""
        self.feature_engineer = fe
    
    def get_prop_predictions(self, available_props: List[Dict]) -> List[Dict]:
        """
        Get model predictions for available props.
        
        Args:
            available_props: List of props with player, type, line, odds
        
        Returns:
            Props with predicted probabilities
        """
        predictions = []
        
        for prop in available_props:
            prop_type = prop['prop_type']
            
            # Check if we have a model for this prop type
            if prop_type not in self.models:
                # Use default probability
                prop['probability'] = 0.5
            else:
                # Get model prediction
                if self.feature_engineer:
                    features = self.feature_engineer.create_live_features(
                        player_id=prop.get('player_id'),
                        prop_type=prop_type,
                        line=prop['line']
                    )
                    
                    model = self.models[prop_type]
                    prop['probability'] = model.predict_proba(features)[0]
                else:
                    # Placeholder probability
                    prop['probability'] = np.random.uniform(0.45, 0.65)
            
            predictions.append(prop)
        
        return predictions
    
    def generate_round_robins(self, props: List[Dict], 
                             min_ev_threshold: float = 0.03,
                             max_legs: int = 12,
                             parlay_sizes: List[int] = None) -> List[Dict]:
        """
        Generate round robin combinations.
        
        Args:
            props: List of props with probabilities and odds
            min_ev_threshold: Minimum EV to include prop
            max_legs: Maximum number of legs
            parlay_sizes: List of parlay sizes to consider
        
        Returns:
            List of round robin configurations
        """
        if parlay_sizes is None:
            parlay_sizes = [2, 3, 4, 6]
        
        # Filter by EV threshold
        high_ev_props = [p for p in props if p.get('ev', 0) >= min_ev_threshold]
        
        if len(high_ev_props) < 4:
            logger.warning(f"Only {len(high_ev_props)} props meet EV threshold")
            return []
        
        # Limit to max legs
        top_props = high_ev_props[:max_legs]
        
        round_robins = []
        
        for num_legs in range(4, min(len(top_props) + 1, max_legs + 1)):
            for parlay_size in parlay_sizes:
                if parlay_size > num_legs:
                    continue
                
                # Calculate round robin details
                rr_result = self.rr_calc.calculate_round_robin(
                    num_legs=num_legs,
                    parlay_size=parlay_size,
                    odds_per_leg=[p['odds'] for p in top_props[:num_legs]],
                    bet_per_parlay=0.10
                )
                
                # Simulate profitability
                sim_result = self.rr_calc.simulate_profitability(
                    num_legs=num_legs,
                    parlay_size=parlay_size,
                    prop_probabilities=[p['probability'] for p in top_props[:num_legs]],
                    odds_per_leg=[p['odds'] for p in top_props[:num_legs]],
                    bet_per_parlay=0.10,
                    num_simulations=1000
                )
                
                config = {
                    'num_legs': num_legs,
                    'parlay_size': parlay_size,
                    'props': top_props[:num_legs],
                    'num_parlays': rr_result['num_parlays'],
                    'total_cost': rr_result['total_cost'],
                    'max_payout': rr_result['max_payout'],
                    'expected_profit': sim_result['expected_profit'],
                    'expected_roi': sim_result['expected_roi'],
                    'win_rate': sim_result['win_rate'],
                    'profit_25_percentile': sim_result['percentile_25'],
                    'profit_75_percentile': sim_result['percentile_75']
                }
                
                round_robins.append(config)
        
        # Sort by expected ROI
        return sorted(round_robins, key=lambda x: x['expected_roi'], reverse=True)
    
    def generate_straight_bets(self, props: List[Dict],
                              min_ev_threshold: float = 0.05,
                              max_bets: int = 10) -> List[Dict]:
        """
        Generate straight bet recommendations.
        
        Args:
            props: List of props with probabilities and odds
            min_ev_threshold: Minimum EV to bet
            max_bets: Maximum number of bets
        
        Returns:
            List of recommended straight bets
        """
        # Filter and sort by EV
        high_ev = [p for p in props if p.get('ev', 0) >= min_ev_threshold]
        high_ev = sorted(high_ev, key=lambda x: x['ev'], reverse=True)
        
        recommendations = []
        
        for prop in high_ev[:max_bets]:
            bet = {
                'player': prop['player'],
                'prop_type': prop['prop_type'],
                'line': prop['line'],
                'side': prop.get('side', 'over'),
                'odds': prop['odds'],
                'probability': prop['probability'],
                'ev': prop['ev'],
                'kelly_bet_pct': prop.get('kelly_bet', 0) * 100,
                'confidence': self._get_confidence_rating(prop['probability'])
            }
            recommendations.append(bet)
        
        return recommendations
    
    def _get_confidence_rating(self, probability: float) -> str:
        """Get confidence rating based on probability."""
        if probability >= 0.70:
            return "HIGH"
        elif probability >= 0.60:
            return "MEDIUM"
        elif probability >= 0.55:
            return "LOW"
        else:
            return "MARGINAL"
    
    def optimize_parlay_portfolio(self, props: List[Dict],
                                 bankroll: float = 100,
                                 risk_tolerance: str = "moderate") -> Dict:
        """
        Optimize portfolio of parlays and straight bets.
        
        Args:
            props: Available props
            bankroll: Total bankroll
            risk_tolerance: 'conservative', 'moderate', or 'aggressive'
        
        Returns:
            Optimized betting portfolio
        """
        risk_params = {
            'conservative': {
                'max_exposure': 0.05,
                'min_ev': 0.05,
                'kelly_fraction': 0.15,
                'parlay_pct': 0.2
            },
            'moderate': {
                'max_exposure': 0.10,
                'min_ev': 0.03,
                'kelly_fraction': 0.25,
                'parlay_pct': 0.35
            },
            'aggressive': {
                'max_exposure': 0.20,
                'min_ev': 0.01,
                'kelly_fraction': 0.40,
                'parlay_pct': 0.50
            }
        }
        
        params = risk_params.get(risk_tolerance, risk_params['moderate'])
        
        # Calculate allocations
        total_exposure = bankroll * params['max_exposure']
        parlay_allocation = total_exposure * params['parlay_pct']
        straight_allocation = total_exposure * (1 - params['parlay_pct'])
        
        # Get predictions and EV
        props_with_pred = self.get_prop_predictions(props)
        props_ranked = self.ev_calc.rank_props_by_ev(props_with_pred)
        
        # Generate bets
        straight_bets = self.generate_straight_bets(
            props_ranked,
            min_ev_threshold=params['min_ev']
        )
        
        round_robins = self.generate_round_robins(
            props_ranked,
            min_ev_threshold=params['min_ev']
        )
        
        portfolio = {
            'bankroll': bankroll,
            'risk_tolerance': risk_tolerance,
            'total_exposure': total_exposure,
            'allocations': {
                'straight_bets': straight_allocation,
                'parlays': parlay_allocation
            },
            'straight_bets': straight_bets[:5],  # Top 5
            'round_robins': round_robins[:3],    # Top 3 configurations
            'expected_return': self._calculate_portfolio_return(
                straight_bets[:5], round_robins[:3],
                straight_allocation, parlay_allocation
            )
        }
        
        return portfolio
    
    def _calculate_portfolio_return(self, straight_bets, round_robins,
                                   straight_alloc, parlay_alloc) -> float:
        """Calculate expected return for portfolio."""
        
        expected_return = 0
        
        # Straight bets return
        if straight_bets:
            num_bets = len(straight_bets)
            bet_size = straight_alloc / num_bets if num_bets > 0 else 0
            
            for bet in straight_bets:
                expected_return += bet_size * bet['ev']
        
        # Parlay return
        if round_robins and len(round_robins) > 0:
            # Use best round robin
            best_rr = round_robins[0]
            if best_rr['total_cost'] > 0:
                rr_scale = parlay_alloc / best_rr['total_cost']
                expected_return += best_rr['expected_profit'] * rr_scale
        
        return expected_return


def main():
    """Example usage of parlay generator."""
    print("=" * 60)
    print("PARLAY GENERATION SYSTEM")
    print("=" * 60)
    
    # Initialize
    parlay_gen = ParlayGenerator()
    
    # Example props for Week 1
    available_props = [
        {
            'player': 'Patrick Mahomes',
            'player_id': 'mahomes_001',
            'prop_type': 'pass_yards',
            'line': 275.5,
            'odds': -110,
            'side': 'over'
        },
        {
            'player': 'Josh Allen',
            'player_id': 'allen_001',
            'prop_type': 'pass_yards',
            'line': 265.5,
            'odds': -115,
            'side': 'over'
        },
        {
            'player': 'Travis Kelce',
            'player_id': 'kelce_001',
            'prop_type': 'receptions',
            'line': 5.5,
            'odds': -120,
            'side': 'over'
        },
        {
            'player': 'Stefon Diggs',
            'player_id': 'diggs_001',
            'prop_type': 'rec_yards',
            'line': 75.5,
            'odds': +105,
            'side': 'over'
        },
        {
            'player': 'Christian McCaffrey',
            'player_id': 'cmc_001',
            'prop_type': 'rush_yards',
            'line': 85.5,
            'odds': -105,
            'side': 'over'
        },
        {
            'player': 'Tyreek Hill',
            'player_id': 'hill_001',
            'prop_type': 'rec_yards',
            'line': 90.5,
            'odds': -110,
            'side': 'over'
        },
        {
            'player': 'Dak Prescott',
            'player_id': 'dak_001',
            'prop_type': 'pass_tds',
            'line': 1.5,
            'odds': -130,
            'side': 'over'
        },
        {
            'player': 'Derrick Henry',
            'player_id': 'henry_001',
            'prop_type': 'rush_yards',
            'line': 70.5,
            'odds': +115,
            'side': 'over'
        }
    ]
    
    # Get predictions (using placeholder probabilities)
    print("\n1. Getting Prop Predictions...")
    props_with_pred = parlay_gen.get_prop_predictions(available_props)
    
    # Rank by EV
    ev_calc = EVCalculator()
    props_ranked = ev_calc.rank_props_by_ev(props_with_pred)
    
    print("\n2. Top Props by Expected Value:")
    for i, prop in enumerate(props_ranked[:5], 1):
        print(f"   {i}. {prop['player']} - {prop['prop_type']} O{prop['line']}")
        print(f"      Probability: {prop['probability']:.2%}")
        print(f"      EV: {prop['ev']:.3f}")
        print(f"      Kelly Bet: {prop['kelly_bet']*100:.1f}% of bankroll")
    
    # Generate round robins
    print("\n3. Best Round Robin Configurations:")
    round_robins = parlay_gen.generate_round_robins(props_ranked)
    
    for i, rr in enumerate(round_robins[:3], 1):
        print(f"\n   Config {i}: {rr['num_legs']}-leg by {rr['parlay_size']}s")
        print(f"      Parlays: {rr['num_parlays']}")
        print(f"      Cost: ${rr['total_cost']:.2f}")
        print(f"      Max Payout: ${rr['max_payout']:.2f}")
        print(f"      Expected ROI: {rr['expected_roi']:.1f}%")
        print(f"      Win Rate: {rr['win_rate']:.1f}%")
    
    # Generate portfolio
    print("\n4. Optimized Betting Portfolio:")
    portfolio = parlay_gen.optimize_parlay_portfolio(
        available_props,
        bankroll=100,
        risk_tolerance="moderate"
    )
    
    print(f"   Bankroll: ${portfolio['bankroll']}")
    print(f"   Risk Level: {portfolio['risk_tolerance']}")
    print(f"   Total Exposure: ${portfolio['total_exposure']:.2f}")
    print(f"   Expected Return: ${portfolio['expected_return']:.2f}")
    
    print("\n   Straight Bets:")
    for bet in portfolio['straight_bets'][:3]:
        print(f"      - {bet['player']} {bet['prop_type']} O{bet['line']} @ {bet['odds']}")
        print(f"        Confidence: {bet['confidence']}, EV: {bet['ev']:.3f}")
    
    if portfolio['round_robins']:
        best_rr = portfolio['round_robins'][0]
        print(f"\n   Best Round Robin: {best_rr['num_legs']}-leg by {best_rr['parlay_size']}s")
        print(f"      Expected ROI: {best_rr['expected_roi']:.1f}%")
    
    print("\n" + "=" * 60)
    print("Parlay generation system ready!")
    print("Remember: For research purposes only!")


if __name__ == "__main__":
    main()