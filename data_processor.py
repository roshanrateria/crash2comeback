import pandas as pd
import numpy as np
from datetime import datetime, timedelta

class DataProcessor:
    """
    Class to process and enhance stock recovery data
    """
    
    def __init__(self):
        pass
    
    def generate_enhanced_statistics(self, recovery_data, market_stocks):
        """
        Generate enhanced statistics for all stocks with edge case handling
        
        Args:
            recovery_data (pd.DataFrame): Raw recovery analysis data
            market_stocks (pd.DataFrame): Market stock information
            
        Returns:
            pd.DataFrame: Enhanced statistics for each stock
        """
        enhanced_stats = []
        
        # Get unique symbols from recovery data
        symbols = recovery_data['Symbol'].unique()
        
        for symbol in symbols:
            stock_recoveries = recovery_data[recovery_data['Symbol'] == symbol]
            stock_info = market_stocks[market_stocks['Symbol'] == symbol].iloc[0] if len(market_stocks[market_stocks['Symbol'] == symbol]) > 0 else None
            
            if len(stock_recoveries) == 0:
                continue
            
            # Filter out entries with insufficient data
            valid_recoveries = stock_recoveries[~stock_recoveries.get('Insufficient_Data', False)]
            
            total_crashes = len(valid_recoveries)
            
            # Skip stocks with no valid crash data
            if total_crashes == 0:
                # Add entry with zero values for newly listed stocks
                stats = {
                    'Symbol': symbol,
                    'Company_Name': stock_info['Company Name'] if stock_info is not None else 'Unknown',
                    'Industry': stock_info['Industry'] if stock_info is not None else 'Unknown',
                    'ISIN_Code': stock_info['ISIN Code'] if stock_info is not None else 'Unknown',
                    'Total_Crashes': 0,
                    '7_day_Recovery': 0,
                    '15_day_Recovery': 0,
                    '30_day_Recovery': 0,
                    'Unrecoverable': 0,
                    '7_day_Recovery_Rate': 0.0,
                    '15_day_Recovery_Rate': 0.0,
                    '30_day_Recovery_Rate': 0.0,
                    'Unrecoverable_Rate': 0.0,
                    'Overall_Recovery_Rate': 0.0,
                    'Average_Drop_Percentage': 0.0,
                    'Max_Drop_Percentage': 0.0,
                    'Min_Drop_Percentage': 0.0,
                    'Std_Drop_Percentage': 0.0,
                    'Average_Recovery_Days': 0.0,
                    'Median_Recovery_Days': 0.0,
                    'Fastest_Recovery_Days': 0.0,
                    'Slowest_Recovery_Days': 0.0,
                    'Stability_Score': 50.0,  # Neutral score for no data
                    'Resilience_Score': 50.0,  # Neutral score for no data
                    'Risk_Score': 50.0,  # Neutral score for no data
                    'Data_Quality': 'Insufficient Data'
                }
                enhanced_stats.append(stats)
                continue
            
            # Calculate recovery rates for each category
            recovery_counts = valid_recoveries['Recovery_Category'].value_counts()
            
            # Handle edge cases in drop percentage calculations
            drop_percentages = valid_recoveries['Drop_Percentage'].dropna()
            if len(drop_percentages) == 0:
                avg_drop = max_drop = min_drop = std_drop = 0.0
            else:
                avg_drop = drop_percentages.mean()
                max_drop = drop_percentages.max()
                min_drop = drop_percentages.min()
                std_drop = drop_percentages.std() if len(drop_percentages) > 1 else 0.0
            
            # Calculate comprehensive statistics
            stats = {
                'Symbol': symbol,
                'Company_Name': stock_info['Company Name'] if stock_info is not None else 'Unknown',
                'Industry': stock_info['Industry'] if stock_info is not None else 'Unknown',
                'ISIN_Code': stock_info['ISIN Code'] if stock_info is not None else 'Unknown',
                'Total_Crashes': total_crashes,
                '7_day_Recovery': recovery_counts.get('7-day', 0),
                '15_day_Recovery': recovery_counts.get('15-day', 0),
                '30_day_Recovery': recovery_counts.get('30-day', 0),
                'Unrecoverable': recovery_counts.get('Unrecoverable', 0),
                '7_day_Recovery_Rate': (recovery_counts.get('7-day', 0) / total_crashes) * 100 if total_crashes > 0 else 0.0,
                '15_day_Recovery_Rate': (recovery_counts.get('15-day', 0) / total_crashes) * 100 if total_crashes > 0 else 0.0,
                '30_day_Recovery_Rate': (recovery_counts.get('30-day', 0) / total_crashes) * 100 if total_crashes > 0 else 0.0,
                'Unrecoverable_Rate': (recovery_counts.get('Unrecoverable', 0) / total_crashes) * 100 if total_crashes > 0 else 0.0,
                'Overall_Recovery_Rate': ((total_crashes - recovery_counts.get('Unrecoverable', 0)) / total_crashes) * 100 if total_crashes > 0 else 0.0,
                'Average_Drop_Percentage': avg_drop,
                'Max_Drop_Percentage': max_drop,
                'Min_Drop_Percentage': min_drop,
                'Std_Drop_Percentage': std_drop,
                'Average_Recovery_Days': self._calculate_average_recovery_days(valid_recoveries),
                'Median_Recovery_Days': self._calculate_median_recovery_days(valid_recoveries),
                'Fastest_Recovery_Days': self._calculate_fastest_recovery(valid_recoveries),
                'Slowest_Recovery_Days': self._calculate_slowest_recovery(valid_recoveries),
                'Stability_Score': self._calculate_stability_score(valid_recoveries),
                'Resilience_Score': self._calculate_resilience_score(valid_recoveries),
                'Risk_Score': self._calculate_risk_score(valid_recoveries),
                'Data_Quality': 'Sufficient' if total_crashes >= 3 else 'Limited'
            }
            
            enhanced_stats.append(stats)
        
        enhanced_df = pd.DataFrame(enhanced_stats)
        
        if enhanced_df.empty:
            return enhanced_df
        
        # Add additional insights
        enhanced_df = self._add_comparative_metrics(enhanced_df)
        
        return enhanced_df.sort_values('Overall_Recovery_Rate', ascending=False).reset_index(drop=True)
    
    def _calculate_average_recovery_days(self, stock_recoveries):
        """Calculate average days to recovery (excluding unrecoverable)"""
        recovered_stocks = stock_recoveries[stock_recoveries['Days_to_Recovery'].notna() & 
                                          (stock_recoveries['Recovery_Category'] != 'Unrecoverable')]
        return recovered_stocks['Days_to_Recovery'].mean() if len(recovered_stocks) > 0 else None
    
    def _calculate_median_recovery_days(self, stock_recoveries):
        """Calculate median days to recovery (excluding unrecoverable)"""
        recovered_stocks = stock_recoveries[stock_recoveries['Days_to_Recovery'].notna() & 
                                          (stock_recoveries['Recovery_Category'] != 'Unrecoverable')]
        return recovered_stocks['Days_to_Recovery'].median() if len(recovered_stocks) > 0 else None
    
    def _calculate_fastest_recovery(self, stock_recoveries):
        """Calculate fastest recovery time"""
        recovered_stocks = stock_recoveries[stock_recoveries['Days_to_Recovery'].notna() & 
                                          (stock_recoveries['Recovery_Category'] != 'Unrecoverable')]
        return recovered_stocks['Days_to_Recovery'].min() if len(recovered_stocks) > 0 else None
    
    def _calculate_slowest_recovery(self, stock_recoveries):
        """Calculate slowest recovery time (within 30 days)"""
        recovered_stocks = stock_recoveries[stock_recoveries['Days_to_Recovery'].notna() & 
                                          (stock_recoveries['Recovery_Category'] != 'Unrecoverable')]
        return recovered_stocks['Days_to_Recovery'].max() if len(recovered_stocks) > 0 else None
    
    def _calculate_stability_score(self, stock_recoveries):
        """
        Calculate stability score based on recovery consistency and crash frequency
        Score from 0-100 where higher is more stable
        """
        total_crashes = len(stock_recoveries)
        recovery_rate = ((total_crashes - (stock_recoveries['Recovery_Category'] == 'Unrecoverable').sum()) / total_crashes) * 100
        
        # Factor in drop consistency (lower std deviation = more stable)
        drop_std = stock_recoveries['Drop_Percentage'].std()
        drop_consistency = max(0, 100 - (drop_std * 2))  # Normalize std deviation
        
        # Factor in recovery time consistency
        recovered_stocks = stock_recoveries[stock_recoveries['Days_to_Recovery'].notna() & 
                                          (stock_recoveries['Recovery_Category'] != 'Unrecoverable')]
        if len(recovered_stocks) > 1:
            recovery_time_std = recovered_stocks['Days_to_Recovery'].std()
            recovery_consistency = max(0, 100 - (recovery_time_std * 3))
        else:
            recovery_consistency = 50
        
        # Combined stability score
        stability_score = (recovery_rate * 0.5 + drop_consistency * 0.3 + recovery_consistency * 0.2)
        return min(100, max(0, stability_score))
    
    def _calculate_resilience_score(self, stock_recoveries):
        """
        Calculate resilience score based on ability to recover from severe drops
        Score from 0-100 where higher is more resilient
        """
        total_crashes = len(stock_recoveries)
        
        # Base resilience on recovery rate
        recovery_rate = ((total_crashes - (stock_recoveries['Recovery_Category'] == 'Unrecoverable').sum()) / total_crashes) * 100
        
        # Bonus for quick recoveries
        quick_recoveries = (stock_recoveries['Recovery_Category'] == '7-day').sum()
        quick_recovery_bonus = (quick_recoveries / total_crashes) * 20
        
        # Penalty for severe drops that don't recover
        severe_drops = stock_recoveries[stock_recoveries['Drop_Percentage'] > 10]
        if len(severe_drops) > 0:
            severe_unrecoverable = (severe_drops['Recovery_Category'] == 'Unrecoverable').sum()
            severe_penalty = (severe_unrecoverable / len(severe_drops)) * 30
        else:
            severe_penalty = 0
        
        resilience_score = recovery_rate + quick_recovery_bonus - severe_penalty
        return min(100, max(0, resilience_score))
    
    def _calculate_risk_score(self, stock_recoveries):
        """
        Calculate risk score based on crash frequency and severity
        Score from 0-100 where higher is more risky
        """
        total_crashes = len(stock_recoveries)
        
        # Base risk on crash frequency (assuming this is relative to time period)
        frequency_risk = min(100, total_crashes * 10)  # Normalize based on expected frequency
        
        # Risk from severe drops
        avg_drop = stock_recoveries['Drop_Percentage'].mean()
        severity_risk = min(100, (avg_drop - 5) * 5)  # Risk increases beyond 5% threshold
        
        # Risk from unrecoverable crashes
        unrecoverable_rate = (stock_recoveries['Recovery_Category'] == 'Unrecoverable').sum() / total_crashes
        unrecoverable_risk = unrecoverable_rate * 100
        
        # Combined risk score
        risk_score = (frequency_risk * 0.3 + severity_risk * 0.4 + unrecoverable_risk * 0.3)
        return min(100, max(0, risk_score))
    
    def _add_comparative_metrics(self, enhanced_df):
        """Add comparative metrics across all stocks"""
        if len(enhanced_df) == 0:
            return enhanced_df
        
        # Add percentile rankings
        enhanced_df['Recovery_Rate_Percentile'] = enhanced_df['Overall_Recovery_Rate'].rank(pct=True) * 100
        enhanced_df['Stability_Percentile'] = enhanced_df['Stability_Score'].rank(pct=True) * 100
        enhanced_df['Resilience_Percentile'] = enhanced_df['Resilience_Score'].rank(pct=True) * 100
        enhanced_df['Risk_Percentile'] = (1 - enhanced_df['Risk_Score'].rank(pct=True)) * 100  # Lower risk = higher percentile
        
        # Add overall performance score
        enhanced_df['Overall_Performance_Score'] = (
            enhanced_df['Recovery_Rate_Percentile'] * 0.4 +
            enhanced_df['Stability_Percentile'] * 0.2 +
            enhanced_df['Resilience_Percentile'] * 0.2 +
            enhanced_df['Risk_Percentile'] * 0.2
        )
        
        # Add performance categories
        enhanced_df['Performance_Category'] = pd.cut(
            enhanced_df['Overall_Performance_Score'],
            bins=[0, 20, 40, 60, 80, 100],
            labels=['Poor', 'Below Average', 'Average', 'Good', 'Excellent']
        )
        
        return enhanced_df
    
    def generate_sector_analysis(self, enhanced_df):
        """Generate sector-wise analysis"""
        if 'Industry' not in enhanced_df.columns:
            return pd.DataFrame()
        
        sector_stats = enhanced_df.groupby('Industry').agg({
            'Total_Crashes': 'sum',
            'Overall_Recovery_Rate': 'mean',
            'Stability_Score': 'mean',
            'Resilience_Score': 'mean',
            'Risk_Score': 'mean',
            'Average_Drop_Percentage': 'mean',
            'Symbol': 'count'
        }).round(2)
        
        sector_stats.columns = [
            'Total_Sector_Crashes', 'Avg_Recovery_Rate', 'Avg_Stability_Score',
            'Avg_Resilience_Score', 'Avg_Risk_Score', 'Avg_Drop_Percentage', 'Stock_Count'
        ]
        
        return sector_stats.sort_values('Avg_Recovery_Rate', ascending=False)
    
    def identify_outliers(self, enhanced_df):
        """Identify statistical outliers in the dataset"""
        outliers = {}
        
        numeric_columns = ['Total_Crashes', 'Overall_Recovery_Rate', 'Average_Drop_Percentage', 
                          'Stability_Score', 'Resilience_Score', 'Risk_Score']
        
        for col in numeric_columns:
            if col in enhanced_df.columns:
                Q1 = enhanced_df[col].quantile(0.25)
                Q3 = enhanced_df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                outlier_stocks = enhanced_df[
                    (enhanced_df[col] < lower_bound) | (enhanced_df[col] > upper_bound)
                ]['Symbol'].tolist()
                
                if outlier_stocks:
                    outliers[col] = outlier_stocks
        
        return outliers
