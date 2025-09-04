# NFL Player Props Predictor 🏈

A comprehensive machine learning system for predicting NFL player props with focus on Expected Value (EV) optimization and round robin parlay generation.

**⚠️ IMPORTANT: For research and educational purposes only - NOT for actual betting**

## Features

- **Data Acquisition**: Automated fetching of NFL player statistics and prop lines
- **Feature Engineering**: Advanced feature creation including rolling averages, trends, and matchup analysis
- **Machine Learning Models**: 
  - XGBoost baseline models
  - LSTM for sequence prediction
  - Ensemble methods
  - Continual learning for weekly improvements
- **EV Optimization**: Calculate expected value and Kelly Criterion for optimal betting
- **Round Robin Calculator**: Generate and simulate profitability of round robin parlays
- **Portfolio Management**: Risk-adjusted betting portfolio recommendations

## Quick Start

### 1. Setup Environment

```bash
# Navigate to project directory
cd c:\Props_Project

# Run setup script
setup_env.bat
```

### 2. Run Round Robin Calculator (Standalone)

```bash
# Activate environment
conda activate props_predictor

# Run calculator
python utils/round_robin_calculator.py
```

### 3. Launch Web Interface

```bash
# Activate environment
conda activate props_predictor

# Run Streamlit app
streamlit run main.py
```

### 4. CLI Usage

```bash
# Train models on historical data
python main.py --mode cli --train

# Perform weekly update
python main.py --mode cli --update

# Get predictions
python main.py --mode cli
```

## Project Structure

```
c:\Props_Project\
├── data/                   # Data storage
│   └── nfl_props.db       # SQLite database
├── models/                 # ML models
│   ├── prop_predictor.py  # Model implementations
│   └── saved/             # Saved models
├── scripts/               # Core scripts
│   ├── data_acquisition.py
│   ├── feature_engineering.py
│   └── parlay_generator.py
├── utils/                 # Utilities
│   └── round_robin_calculator.py
├── notebooks/             # Jupyter notebooks
├── tests/                 # Unit tests
├── main.py               # Main application
├── requirements.txt      # Dependencies
└── setup_env.bat        # Environment setup
```

## Key Components

### Round Robin Calculator
Calculate optimal round robin configurations:
- Number of parlays and total cost
- Maximum payout and ROI
- Monte Carlo profitability simulations
- Breakeven analysis

### Data Pipeline
1. **Historical Data**: 2020-2024 play-by-play and player stats
2. **Weekly Updates**: Current week injuries, weather, and odds
3. **Feature Engineering**: 20+ features per prop including:
   - Player rolling averages (L3, L5, season)
   - Team offensive/defensive rankings
   - Game context (spread, total, weather)
   - Historical prop performance

### Model Training
- **Baseline**: XGBoost with hyperparameter optimization
- **Advanced**: LSTM for sequence modeling
- **Ensemble**: Weighted combination of models
- **Continual Learning**: Weekly model updates with replay buffer

### EV Optimization
- Calculate expected value for each prop
- Kelly Criterion for optimal bet sizing
- Portfolio allocation between straight bets and parlays
- Risk-adjusted recommendations

## Performance Targets

- **Initial Accuracy**: 60-70% per prop
- **Mid-Season Goal**: 70-80% accuracy
- **Focus**: High-EV selections for round robin parlays
- **Risk Management**: Conservative, moderate, and aggressive portfolios

## Example Workflow

1. **Monday**: Fetch previous week's results, update models
2. **Tuesday-Wednesday**: Analyze upcoming matchups
3. **Thursday**: Generate prop predictions and EV rankings
4. **Friday**: Create round robin configurations
5. **Sunday**: Track performance (research only)
6. **Monday**: Repeat with learnings

## Testing

```bash
# Run all tests
pytest tests/

# Test round robin calculator
python utils/round_robin_calculator.py

# Test data acquisition
python scripts/data_acquisition.py

# Test feature engineering
python scripts/feature_engineering.py

# Test model training
python models/prop_predictor.py

# Test parlay generation
python scripts/parlay_generator.py
```

## GPU Support

The system automatically detects and uses CUDA-enabled GPUs for:
- XGBoost training
- PyTorch LSTM models
- Large-scale simulations

Verify GPU support:
```python
import torch
print(torch.cuda.is_available())
```

## Notes

- Models improve weekly through continual learning
- Focus on props with highest edge (pass yards, receptions, rush yards)
- Round robins reduce variance compared to straight parlays
- Always consider bankroll management

## Disclaimer

This software is provided for educational and research purposes only. Sports betting involves financial risk and may not be legal in your jurisdiction. The creators of this software do not encourage or endorse gambling. Users are responsible for complying with all applicable laws and regulations.

## License

For research use only. Not for commercial distribution.

---

*Built with Python, XGBoost, PyTorch, and Streamlit*