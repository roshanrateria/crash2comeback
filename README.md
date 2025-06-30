# Indian Stock Market Recovery Analyzer

## Overview

The Indian Stock Market Recovery Analyzer is a powerful Streamlit application designed for traders and investors to make data-driven decisions based on historical market crash recovery patterns. This tool identifies market crash points across multiple Indian indices and analyzes how individual stocks recover following these market downturns.

## Key Features

- **Cross-Market Analysis**: Identify crash points in one market (e.g., NIFTY 50) and analyze how stocks from another market (e.g., BANKNIFTY) recover from those crash events
- **Multiple Market Support**: Analysis of NIFTY50, NIFTY500, BANKNIFTY, NIFTYMIDCAP, and NIFTYSMALLCAP indices
- **Customizable Crash Detection**: Adjust crash thresholds and detection windows to suit different trading strategies
- **Recovery Pattern Analysis**: Categorize stock recovery patterns into 7-day, 15-day, 30-day, and unrecoverable categories
- **Industry-Level Insights**: Analyze recovery patterns across different industry sectors
- **Trading Recommendations**: Get data-driven suggestions for stocks to BUY, HOLD, or SELL based on their historical recovery patterns
- **Advanced Analytics**: View momentum stocks, value opportunities, and sector rotation strategies
- **Interactive Visualizations**: Explore data through charts, heatmaps, and interactive tables
- **Performance Optimization**: Sample option for faster analysis of large indices like NIFTY 500

## How It Works

1. **Crash Detection**: The application identifies market crash points using the selected index data based on user-defined crash threshold and window parameters
2. **Stock Analysis**: For each crash point, the tool analyzes how individual stocks from the selected analysis market responded
3. **Recovery Categorization**: Stocks are categorized based on their recovery time: 7-day, 15-day, 30-day, or unrecoverable
4. **Statistical Analysis**: Enhanced statistics including recovery rates, stability scores, and risk scores are calculated
5. **Trading Insights**: Based on recovery patterns, the application generates trading insights and recommendations
6. **Visualization**: Results are presented through interactive visualizations and tables

## Use Cases

- **Crash Resilience Assessment**: Identify which stocks historically recover quickly from market crashes
- **Risk Management**: Determine which stocks are most vulnerable during market downturns
- **Sector Rotation Strategy**: Develop sector rotation strategies based on different market conditions
- **Buy-on-Dip Strategies**: Find stocks with consistent quick recovery patterns for "buy on dip" strategies
- **Cross-Market Analysis**: Study how stocks in one market (e.g., BANKNIFTY) respond to crashes in another market (e.g., NIFTY 50)

## Getting Started

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/StockRecoveryAnalyzer.git

# Navigate to the project directory
cd StockRecoveryAnalyzer

# Install required packages
pip install -r requirements.txt

# Run the application
streamlit run app.py
```

### Usage

1. **Select Market Indices**:
   - Choose a market index for crash detection (e.g., NIFTY50)
   - Choose the same or a different market index for stock analysis (e.g., BANKNIFTY)

2. **Configure Parameters**:
   - Adjust years of historical data (1-20 years)
   - Set crash threshold percentage (3-10%)
   - Define crash detection window (1-5 days)
   - Select recovery time windows (7, 15, 30 days)

3. **Run Analysis**:
   - Click "Start Analysis" to process the data
   - Review results across various tabs and visualizations
   - Export data in CSV format for further analysis

### Example Cross-Market Analysis Scenarios

1. **NIFTY50 crashes affecting Banking stocks**:
   - Crash Detection: NIFTY50
   - Stock Analysis: BANKNIFTY
   - This analysis shows how banking stocks respond to overall market crashes

2. **BANKNIFTY crashes affecting Midcap stocks**:
   - Crash Detection: BANKNIFTY
   - Stock Analysis: NIFTYMIDCAP
   - This reveals how mid-cap stocks behave when the banking sector crashes

3. **Broader market crashes affecting specific sectors**:
   - Crash Detection: NIFTY500
   - Stock Analysis: NIFTY50
   - This helps identify blue-chip stocks that are resistant to broader market downturns

## Data Source

The application uses Yahoo Finance (via yfinance) to retrieve historical price data for Indian indices and stocks.

## System Requirements

- Python 3.7+
- Required packages: streamlit, pandas, numpy, plotly, yfinance

## License

[MIT License](LICENSE)

## Disclaimer

This application is for informational purposes only and does not constitute investment advice. Past performance is no guarantee of future results.
