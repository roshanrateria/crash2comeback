import pandas as pd
import numpy as np
import os
from datetime import datetime

def load_market_stocks(market_type="NIFTY50"):
    """
    Load stock list from the attached CSV files based on market type
    
    Args:
        market_type (str): Type of market index ('NIFTY50', 'NIFTY500', etc.)
    
    Returns:
        pd.DataFrame: DataFrame containing stock information
    """
    try:
        if market_type == "NIFTY50":
            csv_path = "attached_assets/ind_nifty50list_1750423694608.csv"
        elif market_type == "NIFTY500":
            csv_path = "attached_assets/ind_nifty500list_1750424863302.csv"
        else:
            csv_path = "attached_assets/ind_nifty50list_1750423694608.csv"  # Default fallback
        
        if os.path.exists(csv_path):
            nifty_df = pd.read_csv(csv_path)
        else:
            # No fallback data - use actual files only
            raise Exception(f"Market data file not found: {csv_path}")
        
        # Clean up any empty rows
        nifty_df = nifty_df.dropna(subset=['Symbol'])
        nifty_df = nifty_df[nifty_df['Symbol'].str.strip() != '']
        
        return nifty_df
        
    except Exception as e:
        raise Exception(f"Error loading {market_type} stock data: {str(e)}")

def get_market_indices():
    """
    Get available market indices and their index symbols
    
    Returns:
        dict: Market indices with their Yahoo Finance symbols
    """
    return {
        "NIFTY50": "^NSEI",
        "NIFTY500": "^CRSLDX",  # Closest proxy for NIFTY 500
        "BANKNIFTY": "^NSEBANK",
        "NIFTYMIDCAP": "^NSEMDCP50",
        "NIFTYSMALLCAP": "NIFTY_SMLCAP_100.NS"
    }

def get_trading_insights(recovery_stats, crash_data):
    """
    Generate comprehensive trading insights based on recovery analysis
    
    Args:
        recovery_stats (pd.DataFrame): Enhanced recovery statistics
        crash_data (pd.DataFrame): Market crash data
        
    Returns:
        dict: Trading insights and recommendations
    """
    insights = {}
    
    # Quick recovery stocks (Buy on crashes)
    quick_recovery = recovery_stats[
        (recovery_stats['7_day_Recovery_Rate'] > 60) & 
        (recovery_stats['Overall_Recovery_Rate'] > 80)
    ].sort_values('7_day_Recovery_Rate', ascending=False)
    
    # Stable performers (Hold during volatility)
    stable_stocks = recovery_stats[
        (recovery_stats['Stability_Score'] > 70) & 
        (recovery_stats['Risk_Score'] < 40)
    ].sort_values('Stability_Score', ascending=False)
    
    # High risk stocks (Consider selling before crashes)
    high_risk = recovery_stats[
        (recovery_stats['Unrecoverable_Rate'] > 30) | 
        (recovery_stats['Risk_Score'] > 70)
    ].sort_values('Risk_Score', ascending=False)
    
    # Sector resilience
    sector_performance = recovery_stats.groupby('Industry').agg({
        'Overall_Recovery_Rate': 'mean',
        'Stability_Score': 'mean',
        'Risk_Score': 'mean',
        '7_day_Recovery_Rate': 'mean',
        'Average_Drop_Percentage': 'mean',
        'Symbol': 'count'
    }).round(2)
    
    # Advanced analytics
    # Crash severity analysis
    crash_severity = analyze_crash_severity(crash_data)
    
    # Momentum indicators
    momentum_stocks = recovery_stats[
        (recovery_stats['7_day_Recovery_Rate'] > recovery_stats['7_day_Recovery_Rate'].quantile(0.8)) &
        (recovery_stats['15_day_Recovery_Rate'] > recovery_stats['15_day_Recovery_Rate'].quantile(0.7))
    ].sort_values(['7_day_Recovery_Rate', '15_day_Recovery_Rate'], ascending=False)
    
    # Value opportunities (high drop but good recovery)
    value_opportunities = recovery_stats[
        (recovery_stats['Average_Drop_Percentage'] > recovery_stats['Average_Drop_Percentage'].quantile(0.7)) &
        (recovery_stats['Overall_Recovery_Rate'] > 70)
    ].sort_values('Average_Drop_Percentage', ascending=False)
    
    insights['buy_on_crash'] = quick_recovery.head(15)
    insights['hold_stocks'] = stable_stocks.head(15)
    insights['sell_candidates'] = high_risk.head(15)
    insights['momentum_stocks'] = momentum_stocks.head(10)
    insights['value_opportunities'] = value_opportunities.head(10)
    insights['sector_performance'] = sector_performance.sort_values('Overall_Recovery_Rate', ascending=False)
    insights['crash_severity'] = crash_severity
    
    return insights

def analyze_crash_severity(crash_data):
    """
    Analyze crash severity patterns
    
    Args:
        crash_data (pd.DataFrame): Market crash data
        
    Returns:
        dict: Crash severity analysis
    """
    if crash_data.empty:
        return {}
    
    severity_analysis = {}
    
    # Classify crashes by severity
    severe_crashes = crash_data[crash_data['Drop_Percentage'] >= 10]
    moderate_crashes = crash_data[(crash_data['Drop_Percentage'] >= 7) & (crash_data['Drop_Percentage'] < 10)]
    mild_crashes = crash_data[crash_data['Drop_Percentage'] < 7]
    
    severity_analysis['total_crashes'] = len(crash_data)
    severity_analysis['severe_crashes'] = len(severe_crashes)
    severity_analysis['moderate_crashes'] = len(moderate_crashes)
    severity_analysis['mild_crashes'] = len(mild_crashes)
    severity_analysis['avg_drop'] = crash_data['Drop_Percentage'].mean()
    severity_analysis['max_drop'] = crash_data['Drop_Percentage'].max()
    severity_analysis['crash_frequency_per_year'] = len(crash_data) / ((crash_data['Date'].max() - crash_data['Date'].min()).days / 365.25) if len(crash_data) > 0 else 0
    
    return severity_analysis

def get_sector_rotation_insights(recovery_stats):
    """
    Generate sector rotation insights for different market conditions
    
    Args:
        recovery_stats (pd.DataFrame): Enhanced recovery statistics
        
    Returns:
        dict: Sector rotation recommendations
    """
    rotation_insights = {}
    
    # Defensive sectors (low volatility, stable recovery)
    defensive = recovery_stats.groupby('Industry').agg({
        'Stability_Score': 'mean',
        'Risk_Score': 'mean',
        'Overall_Recovery_Rate': 'mean',
        'Symbol': 'count'
    }).query('Stability_Score > 60 and Risk_Score < 50').sort_values('Stability_Score', ascending=False)
    
    # Growth sectors (high recovery rates, momentum)
    growth = recovery_stats.groupby('Industry').agg({
        '7_day_Recovery_Rate': 'mean',
        'Overall_Recovery_Rate': 'mean',
        'Symbol': 'count'
    }).query('`7_day_Recovery_Rate` > 40').sort_values('7_day_Recovery_Rate', ascending=False)
    
    # Cyclical sectors (high volatility but good recovery)
    cyclical = recovery_stats.groupby('Industry').agg({
        'Average_Drop_Percentage': 'mean',
        'Overall_Recovery_Rate': 'mean',
        'Risk_Score': 'mean',
        'Symbol': 'count'
    }).query('Average_Drop_Percentage > 8 and Overall_Recovery_Rate > 60').sort_values('Overall_Recovery_Rate', ascending=False)
    
    rotation_insights['defensive_sectors'] = defensive
    rotation_insights['growth_sectors'] = growth
    rotation_insights['cyclical_sectors'] = cyclical
    
    return rotation_insights

def format_currency(amount, currency_symbol="₹"):
    """
    Format currency amount with proper formatting
    
    Args:
        amount (float): Amount to format
        currency_symbol (str): Currency symbol to use
        
    Returns:
        str: Formatted currency string
    """
    if pd.isna(amount) or amount is None:
        return "N/A"
    
    if amount >= 10000000:  # 1 crore
        return f"{currency_symbol}{amount/10000000:.2f}Cr"
    elif amount >= 100000:  # 1 lakh
        return f"{currency_symbol}{amount/100000:.2f}L"
    elif amount >= 1000:  # 1 thousand
        return f"{currency_symbol}{amount/1000:.2f}K"
    else:
        return f"{currency_symbol}{amount:.2f}"

def format_percentage(percentage):
    """
    Format percentage with proper decimal places
    
    Args:
        percentage (float): Percentage to format
        
    Returns:
        str: Formatted percentage string
    """
    if pd.isna(percentage) or percentage is None:
        return "N/A"
    
    return f"{percentage:.2f}%"

def validate_date_range(start_date, end_date):
    """
    Validate date range for stock analysis
    
    Args:
        start_date (datetime): Start date
        end_date (datetime): End date
        
    Returns:
        bool: True if valid, False otherwise
    """
    if start_date >= end_date:
        return False
    
    if (end_date - start_date).days < 30:  # Minimum 30 days
        return False
    
    if start_date < datetime(2010, 1, 1):  # Don't go too far back
        return False
    
    if end_date > datetime.now():
        return False
    
    return True

def calculate_trading_days(start_date, end_date):
    """
    Calculate approximate number of trading days between two dates
    
    Args:
        start_date (datetime): Start date
        end_date (datetime): End date
        
    Returns:
        int: Approximate trading days
    """
    total_days = (end_date - start_date).days
    # Approximate: 5/7 of days are trading days (excluding weekends)
    # Further reduce by ~10% for holidays
    trading_days = int(total_days * 5/7 * 0.9)
    return max(1, trading_days)

def get_sector_color_map():
    """
    Get color mapping for different sectors/industries
    
    Returns:
        dict: Mapping of sectors to colors
    """
    return {
        'Information Technology': '#1f77b4',
        'Financial Services': '#ff7f0e', 
        'Healthcare': '#2ca02c',
        'Fast Moving Consumer Goods': '#d62728',
        'Automobile and Auto Components': '#9467bd',
        'Metals & Mining': '#8c564b',
        'Oil Gas & Consumable Fuels': '#e377c2',
        'Consumer Durables': '#7f7f7f',
        'Construction Materials': '#bcbd22',
        'Capital Goods': '#17becf',
        'Telecommunication': '#ff9896',
        'Power': '#98df8a',
        'Consumer Services': '#ffbb78',
        'Construction': '#c5b0d5',
        'Services': '#c49c94'
    }

def clean_stock_symbol(symbol):
    """
    Clean stock symbol for Yahoo Finance API
    
    Args:
        symbol (str): Raw stock symbol
        
    Returns:
        str: Cleaned symbol for yfinance
    """
    # Remove special characters and add .NS suffix for NSE stocks
    cleaned = symbol.replace('&', '').replace('-', '').strip()
    return f"{cleaned}.NS"

def safe_divide(numerator, denominator, default=0):
    """
    Safely divide two numbers, returning default if denominator is zero
    
    Args:
        numerator (float): Numerator
        denominator (float): Denominator
        default (float): Default value if division by zero
        
    Returns:
        float: Result of division or default
    """
    if denominator == 0 or pd.isna(denominator) or pd.isna(numerator):
        return default
    return numerator / denominator

def get_recovery_category_color(category):
    """
    Get color for recovery category
    
    Args:
        category (str): Recovery category
        
    Returns:
        str: Color code
    """
    color_map = {
        '7-day': '#2E8B57',      # Sea Green
        '15-day': '#1f77b4',     # Blue
        '30-day': '#ff7f0e',     # Orange
        'Unrecoverable': '#dc3545'  # Red
    }
    return color_map.get(category, '#6c757d')  # Default gray

def format_number(number, decimals=2):
    """
    Format number with proper decimal places and thousand separators
    
    Args:
        number (float): Number to format
        decimals (int): Number of decimal places
        
    Returns:
        str: Formatted number string
    """
    if pd.isna(number) or number is None:
        return "N/A"
    
    return f"{number:,.{decimals}f}"

def get_performance_emoji(score):
    """
    Get emoji based on performance score
    
    Args:
        score (float): Performance score (0-100)
        
    Returns:
        str: Appropriate emoji
    """
    if score >= 80:
        return "🏆"  # Trophy
    elif score >= 60:
        return "👍"  # Thumbs up
    elif score >= 40:
        return "👌"  # OK hand
    elif score >= 20:
        return "⚠️"   # Warning
    else:
        return "🔻"  # Red triangle down
