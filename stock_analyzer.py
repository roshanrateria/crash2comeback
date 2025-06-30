import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import yfinance as yf

class StockAnalyzer:
    """
    Class to analyze stock crash and recovery patterns
    """
    
    def __init__(self, crash_threshold=5.0, crash_window=2, recovery_windows=[7, 15, 30]):
        """
        Initialize the stock analyzer
        
        Args:
            crash_threshold (float): Minimum percentage drop to identify as crash
            crash_window (int): Number of days over which to detect crash
            recovery_windows (list): Time windows to analyze for recovery
        """
        self.crash_threshold = crash_threshold
        self.crash_window = crash_window
        self.recovery_windows = sorted(recovery_windows)
    
    def identify_market_crashes(self, market_data):
        """
        Identify market crash points based on NIFTY 50 index data
        
        Args:
            market_data (pd.DataFrame): Market index data with OHLCV columns
            
        Returns:
            list: List of market crash events with details
        """
        crashes = []
        
        if len(market_data) < self.crash_window + 1:
            return crashes
        
        # Calculate rolling minimum over crash window
        market_data = market_data.copy()
        market_data['Rolling_Min'] = market_data['Low'].rolling(window=self.crash_window).min()
        market_data['Rolling_Max_Pre'] = market_data['High'].shift(1).rolling(window=self.crash_window).max()
        
        for i in range(self.crash_window, len(market_data)):
            current_date = market_data.index[i]
            
            # Get pre-crash high and current low
            pre_crash_high = market_data['Rolling_Max_Pre'].iloc[i]
            crash_low = market_data['Rolling_Min'].iloc[i]
            
            if pd.isna(pre_crash_high) or pd.isna(crash_low):
                continue
            
            # Calculate drop percentage
            drop_percentage = ((pre_crash_high - crash_low) / pre_crash_high) * 100
            
            if drop_percentage >= self.crash_threshold:
                # Ensure we haven't identified a crash too recently
                if crashes:
                    last_crash_date = crashes[-1]['Date']
                    if (current_date - last_crash_date).days < self.crash_window * 2:
                        continue
                
                crash_info = {
                    'Date': current_date,
                    'Pre_Crash_Index': pre_crash_high,
                    'Crash_Index': crash_low,
                    'Drop_Percentage': drop_percentage,
                    'Close_Index': market_data['Close'].iloc[i]
                }
                crashes.append(crash_info)
        
        return crashes
    
    def get_stock_prices_at_crash(self, stock_data_dict, crash_dates):
        """
        Get stock prices for all stocks at market crash dates with edge case handling
        
        Args:
            stock_data_dict (dict): Dictionary of stock data
            crash_dates (list): List of crash dates
            
        Returns:
            pd.DataFrame: Stock prices at crash points
        """
        crash_stock_data = []
        
        for crash_date in crash_dates:
            for symbol, stock_data in stock_data_dict.items():
                # Check if stock data covers the crash date
                if stock_data.index.min() > crash_date:
                    # Stock was listed after the crash date, skip this crash
                    continue
                
                # Find the closest date to crash date
                crash_idx = None
                for idx, date in enumerate(stock_data.index):
                    if date >= crash_date:
                        crash_idx = idx
                        break
                
                if crash_idx is None or crash_idx == 0:
                    # No data available for this crash date or no pre-crash data
                    continue
                
                # Check if we have at least 2 days of data before crash
                if crash_idx < 1:
                    continue
                
                # Get pre-crash price (day before crash)
                pre_crash_price = stock_data['Close'].iloc[crash_idx - 1]
                crash_price = stock_data['Close'].iloc[crash_idx]
                
                # Handle edge case where prices might be NaN
                if pd.isna(pre_crash_price) or pd.isna(crash_price) or pre_crash_price <= 0:
                    continue
                
                drop_percentage = ((pre_crash_price - crash_price) / pre_crash_price) * 100
                
                crash_stock_data.append({
                    'Symbol': symbol,
                    'Crash_Date': crash_date,
                    'Pre_Crash_Price': pre_crash_price,
                    'Crash_Price': crash_price,
                    'Drop_Percentage': drop_percentage,
                    'Stock_Listing_Date': stock_data.index.min(),  # Track when stock was first available
                    'Data_Points_Available': len(stock_data)
                })
        
        return pd.DataFrame(crash_stock_data)
    
    def analyze_recovery(self, stock_data, crash_date, pre_crash_price):
        """
        Analyze recovery pattern after a crash with enhanced edge case handling
        
        Args:
            stock_data (pd.DataFrame): Stock price data
            crash_date (datetime): Date when crash occurred
            pre_crash_price (float): Price before the crash
            
        Returns:
            dict: Recovery analysis results
        """
        recovery_info = {
            'Recovery_Category': 'Unrecoverable',
            'Days_to_Recovery': None,
            'Recovery_Price': None,
            'Max_Recovery_Price': None,
            'Insufficient_Data': False
        }
        
        # Handle edge case: invalid pre_crash_price
        if pd.isna(pre_crash_price) or pre_crash_price <= 0:
            recovery_info['Insufficient_Data'] = True
            return recovery_info
        
        # Find crash date index
        crash_idx = None
        for idx, date in enumerate(stock_data.index):
            if date >= crash_date:
                crash_idx = idx
                break
        
        if crash_idx is None or crash_idx >= len(stock_data) - 1:
            recovery_info['Insufficient_Data'] = True
            return recovery_info
        
        # Check if we have sufficient data after crash for analysis
        post_crash_data = stock_data.iloc[crash_idx:]
        if len(post_crash_data) < 7:  # Less than a week of data
            recovery_info['Insufficient_Data'] = True
            return recovery_info
        
        # Analyze recovery for each time window
        max_recovery_price = 0
        
        for window_days in self.recovery_windows:
            end_date = crash_date + timedelta(days=window_days)
            
            # Find the end date index
            end_idx = None
            for idx in range(crash_idx, len(stock_data)):
                if stock_data.index[idx] >= end_date:
                    end_idx = idx
                    break
            
            if end_idx is None:
                end_idx = len(stock_data) - 1
            
            # Skip if we don't have enough data for this window
            if end_idx <= crash_idx:
                continue
            
            # Check if recovery occurred within this window
            recovery_data = stock_data.iloc[crash_idx:end_idx + 1]
            
            # Handle edge case: empty recovery data
            if recovery_data.empty:
                continue
            
            max_price_in_window = recovery_data['High'].max()
            
            # Handle NaN values
            if pd.isna(max_price_in_window):
                continue
            
            max_recovery_price = max(max_recovery_price, max_price_in_window)
            
            if max_price_in_window >= pre_crash_price:
                # Find the exact day when recovery occurred
                for i, (date, row) in enumerate(recovery_data.iterrows()):
                    if not pd.isna(row['High']) and row['High'] >= pre_crash_price:
                        recovery_info['Recovery_Category'] = f'{window_days}-day'
                        recovery_info['Days_to_Recovery'] = (date - crash_date).days
                        recovery_info['Recovery_Price'] = row['High']
                        recovery_info['Max_Recovery_Price'] = max_recovery_price
                        return recovery_info
        
        # No recovery within any window
        recovery_info['Max_Recovery_Price'] = max_recovery_price if not pd.isna(max_recovery_price) else 0
        
        # Calculate days to end of available data
        if len(stock_data) > crash_idx:
            recovery_info['Days_to_Recovery'] = (stock_data.index[-1] - crash_date).days
        
        return recovery_info
    
    def calculate_stock_statistics(self, recovery_data, symbol):
        """
        Calculate comprehensive statistics for a specific stock
        
        Args:
            recovery_data (pd.DataFrame): Recovery analysis results
            symbol (str): Stock symbol
            
        Returns:
            dict: Stock statistics
        """
        stock_recoveries = recovery_data[recovery_data['Symbol'] == symbol]
        
        if len(stock_recoveries) == 0:
            return None
        
        total_crashes = len(stock_recoveries)
        
        # Calculate recovery rates for each category
        recovery_counts = stock_recoveries['Recovery_Category'].value_counts()
        
        stats = {
            'Symbol': symbol,
            'Total_Crashes': total_crashes,
            '7_day_Recovery': recovery_counts.get('7-day', 0),
            '15_day_Recovery': recovery_counts.get('15-day', 0),
            '30_day_Recovery': recovery_counts.get('30-day', 0),
            'Unrecoverable': recovery_counts.get('Unrecoverable', 0),
            '7_day_Recovery_Rate': (recovery_counts.get('7-day', 0) / total_crashes) * 100,
            '15_day_Recovery_Rate': (recovery_counts.get('15-day', 0) / total_crashes) * 100,
            '30_day_Recovery_Rate': (recovery_counts.get('30-day', 0) / total_crashes) * 100,
            'Unrecoverable_Rate': (recovery_counts.get('Unrecoverable', 0) / total_crashes) * 100,
            'Overall_Recovery_Rate': ((total_crashes - recovery_counts.get('Unrecoverable', 0)) / total_crashes) * 100,
            'Average_Drop_Percentage': stock_recoveries['Drop_Percentage'].mean(),
            'Max_Drop_Percentage': stock_recoveries['Drop_Percentage'].max(),
            'Average_Recovery_Days': stock_recoveries[stock_recoveries['Days_to_Recovery'].notna()]['Days_to_Recovery'].mean()
        }
        
        return stats
    
    def get_market_sentiment_score(self, stock_data, date, window=30):
        """
        Calculate a simple market sentiment score based on recent price movements
        
        Args:
            stock_data (pd.DataFrame): Stock price data
            date (datetime): Reference date
            window (int): Number of days to look back for sentiment
            
        Returns:
            float: Sentiment score between -1 (very bearish) and 1 (very bullish)
        """
        try:
            # Find the date index
            date_idx = None
            for idx, stock_date in enumerate(stock_data.index):
                if stock_date >= date:
                    date_idx = idx
                    break
            
            if date_idx is None or date_idx < window:
                return 0
            
            # Get data for sentiment analysis
            sentiment_data = stock_data.iloc[date_idx - window:date_idx]
            
            if len(sentiment_data) < window:
                return 0
            
            # Calculate various sentiment indicators
            returns = sentiment_data['Close'].pct_change().dropna()
            
            # Simple sentiment metrics
            positive_days = (returns > 0).sum()
            negative_days = (returns < 0).sum()
            
            if positive_days + negative_days == 0:
                return 0
            
            # Sentiment score based on positive vs negative days
            sentiment_score = (positive_days - negative_days) / (positive_days + negative_days)
            
            # Adjust by magnitude of moves
            avg_positive_return = returns[returns > 0].mean() if positive_days > 0 else 0
            avg_negative_return = abs(returns[returns < 0].mean()) if negative_days > 0 else 0
            
            if avg_positive_return + avg_negative_return > 0:
                magnitude_factor = (avg_positive_return - avg_negative_return) / (avg_positive_return + avg_negative_return)
                sentiment_score = (sentiment_score + magnitude_factor) / 2
            
            return max(-1, min(1, sentiment_score))
            
        except Exception:
            return 0
