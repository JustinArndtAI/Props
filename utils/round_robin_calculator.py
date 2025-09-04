import math
from itertools import combinations
from typing import List, Dict, Tuple, Optional
import numpy as np

class RoundRobinCalculator:
    """
    Calculator for round robin parlays with NFL player props.
    Supports variable odds per leg and profitability simulations.
    """
    
    def __init__(self):
        self.american_to_decimal_cache = {}
    
    def american_to_decimal(self, odds: int) -> float:
        """Convert American odds to decimal odds."""
        if odds in self.american_to_decimal_cache:
            return self.american_to_decimal_cache[odds]
        
        if odds > 0:
            decimal = (odds / 100) + 1
        else:
            decimal = (100 / abs(odds)) + 1
        
        self.american_to_decimal_cache[odds] = decimal
        return decimal
    
    def decimal_to_american(self, decimal: float) -> int:
        """Convert decimal odds to American odds."""
        if decimal >= 2.0:
            return int((decimal - 1) * 100)
        else:
            return int(-100 / (decimal - 1))
    
    def calculate_parlay_odds(self, legs: List[int]) -> float:
        """Calculate combined parlay odds from individual leg odds (American format)."""
        decimal_odds = 1.0
        for odds in legs:
            decimal_odds *= self.american_to_decimal(odds)
        return decimal_odds
    
    def calculate_round_robin(
        self,
        num_legs: int,
        parlay_size: int,
        odds_per_leg: Optional[List[int]] = None,
        bet_per_parlay: float = 0.10
    ) -> Dict:
        """
        Calculate round robin parlay details.
        
        Args:
            num_legs: Total number of prop bets
            parlay_size: Size of each parlay (e.g., 4 for "by 4s")
            odds_per_leg: List of American odds for each leg (default: -110 for all)
            bet_per_parlay: Bet amount per parlay
        
        Returns:
            Dictionary with calculation results
        """
        if odds_per_leg is None:
            odds_per_leg = [-110] * num_legs
        
        if len(odds_per_leg) != num_legs:
            raise ValueError(f"odds_per_leg length ({len(odds_per_leg)}) must match num_legs ({num_legs})")
        
        # Calculate number of parlays
        num_parlays = math.comb(num_legs, parlay_size)
        total_cost = num_parlays * bet_per_parlay
        
        # Generate all combinations
        all_combinations = list(combinations(range(num_legs), parlay_size))
        
        # Calculate payouts for each combination
        payouts = []
        for combo in all_combinations:
            combo_odds = [odds_per_leg[i] for i in combo]
            parlay_decimal = self.calculate_parlay_odds(combo_odds)
            payout = bet_per_parlay * parlay_decimal
            payouts.append(payout)
        
        max_payout = sum(payouts)
        avg_payout_per_parlay = np.mean(payouts)
        
        # Calculate breakeven scenarios
        breakeven_wins = 0
        cumulative_payout = 0
        sorted_payouts = sorted(payouts, reverse=True)
        
        for payout in sorted_payouts:
            cumulative_payout += payout
            breakeven_wins += 1
            if cumulative_payout >= total_cost:
                break
        
        breakeven_percentage = (breakeven_wins / num_parlays) * 100
        
        return {
            "num_legs": num_legs,
            "parlay_size": parlay_size,
            "num_parlays": num_parlays,
            "bet_per_parlay": bet_per_parlay,
            "total_cost": total_cost,
            "max_payout": max_payout,
            "max_profit": max_payout - total_cost,
            "avg_payout_per_parlay": avg_payout_per_parlay,
            "breakeven_wins": breakeven_wins,
            "breakeven_percentage": breakeven_percentage,
            "roi_if_all_hit": ((max_payout - total_cost) / total_cost) * 100
        }
    
    def simulate_profitability(
        self,
        num_legs: int,
        parlay_size: int,
        prop_probabilities: List[float],
        odds_per_leg: List[int],
        bet_per_parlay: float = 0.10,
        num_simulations: int = 10000
    ) -> Dict:
        """
        Monte Carlo simulation of round robin profitability.
        
        Args:
            num_legs: Total number of prop bets
            parlay_size: Size of each parlay
            prop_probabilities: Win probability for each prop
            odds_per_leg: American odds for each leg
            bet_per_parlay: Bet amount per parlay
            num_simulations: Number of Monte Carlo runs
        
        Returns:
            Dictionary with simulation results
        """
        all_combinations = list(combinations(range(num_legs), parlay_size))
        num_parlays = len(all_combinations)
        total_cost = num_parlays * bet_per_parlay
        
        profits = []
        win_counts = []
        
        for _ in range(num_simulations):
            # Simulate prop outcomes
            prop_outcomes = [np.random.random() < prob for prob in prop_probabilities]
            
            # Calculate winnings
            total_payout = 0
            parlays_won = 0
            
            for combo in all_combinations:
                if all(prop_outcomes[i] for i in combo):
                    combo_odds = [odds_per_leg[i] for i in combo]
                    parlay_decimal = self.calculate_parlay_odds(combo_odds)
                    total_payout += bet_per_parlay * parlay_decimal
                    parlays_won += 1
            
            profit = total_payout - total_cost
            profits.append(profit)
            win_counts.append(parlays_won)
        
        profits = np.array(profits)
        win_counts = np.array(win_counts)
        
        return {
            "expected_profit": np.mean(profits),
            "profit_std": np.std(profits),
            "win_rate": (profits > 0).mean() * 100,
            "median_profit": np.median(profits),
            "percentile_25": np.percentile(profits, 25),
            "percentile_75": np.percentile(profits, 75),
            "max_profit_sim": np.max(profits),
            "min_profit_sim": np.min(profits),
            "avg_parlays_won": np.mean(win_counts),
            "expected_roi": (np.mean(profits) / total_cost) * 100
        }
    
    def optimize_round_robin(
        self,
        available_props: List[Dict],
        min_legs: int = 4,
        max_legs: int = 12,
        min_parlay_size: int = 2,
        max_parlay_size: int = 8,
        bet_per_parlay: float = 0.10,
        target_roi: float = 20.0
    ) -> List[Dict]:
        """
        Find optimal round robin configurations based on available props.
        
        Args:
            available_props: List of dicts with 'odds' and 'probability' keys
            min_legs: Minimum number of legs to consider
            max_legs: Maximum number of legs to consider
            min_parlay_size: Minimum parlay size
            max_parlay_size: Maximum parlay size
            bet_per_parlay: Bet amount per parlay
            target_roi: Target ROI percentage
        
        Returns:
            List of optimal configurations sorted by expected ROI
        """
        results = []
        
        # Sort props by expected value
        for prop in available_props:
            decimal_odds = self.american_to_decimal(prop['odds'])
            prop['ev'] = (prop['probability'] * decimal_odds) - 1
        
        sorted_props = sorted(available_props, key=lambda x: x['ev'], reverse=True)
        
        # Test different configurations
        for num_legs in range(min_legs, min(max_legs + 1, len(sorted_props) + 1)):
            top_props = sorted_props[:num_legs]
            
            for parlay_size in range(min_parlay_size, min(max_parlay_size + 1, num_legs + 1)):
                if math.comb(num_legs, parlay_size) > 500:  # Skip if too many parlays
                    continue
                
                sim_result = self.simulate_profitability(
                    num_legs=num_legs,
                    parlay_size=parlay_size,
                    prop_probabilities=[p['probability'] for p in top_props],
                    odds_per_leg=[p['odds'] for p in top_props],
                    bet_per_parlay=bet_per_parlay,
                    num_simulations=1000
                )
                
                if sim_result['expected_roi'] >= target_roi:
                    config = {
                        'num_legs': num_legs,
                        'parlay_size': parlay_size,
                        'expected_roi': sim_result['expected_roi'],
                        'win_rate': sim_result['win_rate'],
                        'expected_profit': sim_result['expected_profit'],
                        'props_used': top_props[:num_legs]
                    }
                    results.append(config)
        
        return sorted(results, key=lambda x: x['expected_roi'], reverse=True)


def main():
    """Example usage of the Round Robin Calculator."""
    calc = RoundRobinCalculator()
    
    print("=" * 60)
    print("NFL PLAYER PROPS ROUND ROBIN CALCULATOR")
    print("=" * 60)
    
    # Example 1: Basic 8-leg by 4s round robin
    print("\n1. Basic 8-leg by 4s Round Robin (all -110):")
    result = calc.calculate_round_robin(
        num_legs=8,
        parlay_size=4,
        bet_per_parlay=0.10
    )
    for key, value in result.items():
        if isinstance(value, float):
            print(f"   {key}: ${value:.2f}" if 'payout' in key or 'cost' in key or 'profit' in key 
                  else f"   {key}: {value:.2f}%")
        else:
            print(f"   {key}: {value}")
    
    # Example 2: Mixed odds round robin
    print("\n2. 6-leg by 3s with mixed odds:")
    mixed_odds = [-110, +150, -120, +180, -105, +120]
    result = calc.calculate_round_robin(
        num_legs=6,
        parlay_size=3,
        odds_per_leg=mixed_odds,
        bet_per_parlay=0.25
    )
    for key, value in result.items():
        if isinstance(value, float):
            print(f"   {key}: ${value:.2f}" if 'payout' in key or 'cost' in key or 'profit' in key 
                  else f"   {key}: {value:.2f}%")
        else:
            print(f"   {key}: {value}")
    
    # Example 3: Profitability simulation
    print("\n3. Profitability Simulation (8-leg by 4s, 65% win rate per prop):")
    sim_result = calc.simulate_profitability(
        num_legs=8,
        parlay_size=4,
        prop_probabilities=[0.65] * 8,
        odds_per_leg=[-110] * 8,
        bet_per_parlay=0.10,
        num_simulations=10000
    )
    print(f"   Expected Profit: ${sim_result['expected_profit']:.2f}")
    print(f"   Expected ROI: {sim_result['expected_roi']:.2f}%")
    print(f"   Win Rate: {sim_result['win_rate']:.2f}%")
    print(f"   25th Percentile: ${sim_result['percentile_25']:.2f}")
    print(f"   Median Profit: ${sim_result['median_profit']:.2f}")
    print(f"   75th Percentile: ${sim_result['percentile_75']:.2f}")
    
    # Example 4: Find optimal configuration
    print("\n4. Finding Optimal Configuration:")
    available_props = [
        {'odds': -110, 'probability': 0.70, 'player': 'Mahomes', 'prop': 'Over 275.5 pass yds'},
        {'odds': +150, 'probability': 0.45, 'player': 'Hill', 'prop': 'Over 5.5 receptions'},
        {'odds': -120, 'probability': 0.68, 'player': 'Kelce', 'prop': 'Over 65.5 rec yds'},
        {'odds': +180, 'probability': 0.40, 'player': 'Mahomes', 'prop': 'Over 2.5 pass TDs'},
        {'odds': -105, 'probability': 0.62, 'player': 'Pacheco', 'prop': 'Over 55.5 rush yds'},
        {'odds': +120, 'probability': 0.50, 'player': 'Rice', 'prop': 'Over 4.5 receptions'},
    ]
    
    optimal = calc.optimize_round_robin(
        available_props=available_props,
        min_legs=4,
        max_legs=6,
        min_parlay_size=2,
        max_parlay_size=4,
        target_roi=15.0
    )
    
    if optimal:
        best = optimal[0]
        print(f"   Best Configuration: {best['num_legs']}-leg by {best['parlay_size']}s")
        print(f"   Expected ROI: {best['expected_roi']:.2f}%")
        print(f"   Win Rate: {best['win_rate']:.2f}%")
        print(f"   Props to use:")
        for prop in best['props_used']:
            print(f"      - {prop['player']}: {prop['prop']} @ {prop['odds']}")
    else:
        print("   No configurations meet the target ROI")
    
    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()