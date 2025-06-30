import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import yfinance as yf
from datetime import datetime, timedelta
import time
from stock_analyzer import StockAnalyzer
from data_processor import DataProcessor
from utils import load_market_stocks, get_market_indices, get_trading_insights, get_sector_rotation_insights, format_currency, format_percentage
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Indian Stock Market Recovery Analyzer",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Title and description
st.title("📈 Indian Stock Market Recovery Analyzer")
st.markdown("""
Analyze stock recovery patterns after market crashes across multiple Indian indices.
This application identifies market crash points in one index and categorizes how stocks from the same or different index recover into 7-day, 15-day, 30-day, and Unrecoverable patterns.
""")

# Initialize session state
if 'analysis_complete' not in st.session_state:
    st.session_state.analysis_complete = False
if 'recovery_data' not in st.session_state:
    st.session_state.recovery_data = None
if 'crash_data' not in st.session_state:
    st.session_state.crash_data = None
if 'enhanced_stats' not in st.session_state:
    st.session_state.enhanced_stats = None

# Sidebar parameters
st.sidebar.header("⚙️ Analysis Parameters")

# Add info about cross-market analysis
st.sidebar.info("✨ **Cross-Market Analysis**: You can now analyze how stocks from one market react to crashes in another market!")

st.sidebar.subheader("Market Selection")

# Crash detection market selection
market_indices = get_market_indices()
crash_detection_market = st.sidebar.selectbox(
    "Market Index for Crash Detection",
    options=list(market_indices.keys()),
    index=0,
    help="Choose the market index to identify crash points"
)

# Stock analysis market selection
analysis_market = st.sidebar.selectbox(
    "Market Index for Stock Analysis",
    options=list(market_indices.keys()),
    index=0,
    help="Choose which market's stocks to analyze for recovery patterns"
)

# Load stocks for analysis market
try:
    market_stocks = load_market_stocks(analysis_market)
    st.sidebar.success(f"✅ Loaded {len(market_stocks)} {analysis_market} stocks")
except Exception as e:
    st.sidebar.error(f"❌ Error loading {analysis_market} stocks: {str(e)}")
    st.stop()

# Parameter inputs
years_back = st.sidebar.slider(
    "Years of Historical Data",
    min_value=1,
    max_value=20,
    value=5,
    help="Number of years of historical data to analyze"
)

crash_threshold = st.sidebar.slider(
    "Crash Threshold (%)",
    min_value=3.0,
    max_value=10.0,
    value=5.0,
    step=0.5,
    help="Minimum percentage drop to identify as a crash"
)

crash_window = st.sidebar.slider(
    "Crash Detection Window (days)",
    min_value=1,
    max_value=5,
    value=2,
    help="Number of days over which to detect the crash"
)

recovery_windows = st.sidebar.multiselect(
    "Recovery Analysis Windows (days)",
    options=[7, 15, 30],
    default=[7, 15, 30],
    help="Time windows to analyze for recovery"
)

# Stock selection
selected_stocks = st.sidebar.multiselect(
    f"Select Stocks (Leave empty for all {analysis_market})",
    options=market_stocks['Symbol'].tolist(),
    help=f"Select specific stocks or leave empty to analyze all {analysis_market} stocks"
)

if not selected_stocks:
    selected_stocks = market_stocks['Symbol'].tolist()
    
# Limit selection for performance (especially for NIFTY 500)
if len(selected_stocks) > 500:
    st.sidebar.warning(f"⚠️ Analyzing {len(selected_stocks)} stocks may take significant time. Consider selecting fewer stocks for faster analysis.")
    
if len(selected_stocks) > 100 and analysis_market == "NIFTY500":
    sample_size = st.sidebar.slider("Sample Size (for faster analysis)", 50, len(selected_stocks), 100)
    if len(selected_stocks) > sample_size:
        selected_stocks = selected_stocks[:sample_size]
        st.sidebar.info(f"📊 Using first {sample_size} stocks for analysis")

# Analysis button
analyze_button = st.sidebar.button(
    "🚀 Start Analysis",
    type="primary",
    use_container_width=True
)

# Main content area
if analyze_button:
    st.session_state.analysis_complete = False
    
    # Initialize analyzers
    data_processor = DataProcessor()
    stock_analyzer = StockAnalyzer(crash_threshold, crash_window, recovery_windows)
    
    # Calculate date range
    end_date = datetime.now()
    start_date = end_date - timedelta(days=years_back * 365)
    
    st.info(f"🔄 Starting analysis for {len(selected_stocks)} {analysis_market} stocks using {crash_detection_market} crash points from {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
    
    # Progress tracking
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    try:
        # Step 1: Fetch stock data
        status_text.text("📥 Fetching stock data...")
        stock_data = {}
        failed_stocks = []
        
        for i, symbol in enumerate(selected_stocks):
            try:
                progress_bar.progress((i + 1) / len(selected_stocks) * 0.3)
                status_text.text(f"📥 Fetching data for {symbol} ({i+1}/{len(selected_stocks)})")
                
                ticker = yf.Ticker(f"{symbol}.NS")
                data = ticker.history(start=start_date, end=end_date)
                
                if not data.empty:
                    # Check if stock has sufficient data for analysis
                    if len(data) > 30:  # At least 30 days of data
                        stock_data[symbol] = data
                    else:
                        # Stock might be newly listed, use available data but note it
                        stock_data[symbol] = data
                        st.sidebar.info(f"⚠️ {symbol}: Limited data ({len(data)} days) - newly listed stock")
                else:
                    failed_stocks.append(symbol)
                    
            except Exception as e:
                failed_stocks.append(symbol)
                st.warning(f"Failed to fetch data for {symbol}: {str(e)}")
        
        if failed_stocks:
            st.warning(f"⚠️ Failed to fetch data for {len(failed_stocks)} stocks: {', '.join(failed_stocks)}")
        
        # Check for stocks with limited data (newly listed)
        limited_data_stocks = []
        for symbol, data in stock_data.items():
            data_years = (data.index.max() - data.index.min()).days / 365.25
            if data_years < years_back * 0.5:  # Less than half the requested period
                limited_data_stocks.append(f"{symbol} ({data_years:.1f}y)")
        
        if limited_data_stocks:
            st.info(f"📊 Stocks with limited historical data: {', '.join(limited_data_stocks)}")
        
        if not stock_data:
            st.error("❌ No stock data could be fetched. Please check your internet connection and try again.")
            st.stop()
        
        # Step 2: Fetch market index data and identify market crashes
        status_text.text(f"🔍 Fetching {crash_detection_market} index data...")
        progress_bar.progress(0.35)
        
        try:
            market_symbol = market_indices[crash_detection_market]
            market_ticker = yf.Ticker(market_symbol)
            market_data = market_ticker.history(start=start_date, end=end_date)
            
            if market_data.empty:
                st.error(f"❌ Failed to fetch {crash_detection_market} index data. Please try again.")
                st.stop()
                
        except Exception as e:
            st.error(f"❌ Error fetching {crash_detection_market} data: {str(e)}")
            st.stop()
        
        # Step 3: Identify market crash points
        status_text.text("🔍 Identifying market crash points...")
        progress_bar.progress(0.4)
        
        market_crashes = stock_analyzer.identify_market_crashes(market_data)
        
        if not market_crashes:
            st.warning("⚠️ No market crash points identified with the current parameters. Try adjusting the crash threshold or window.")
            st.stop()
        
        # Step 4: Get stock prices at crash dates
        status_text.text("📊 Getting stock prices at crash dates...")
        progress_bar.progress(0.5)
        
        crash_dates = [crash['Date'] for crash in market_crashes]
        crash_stock_df = stock_analyzer.get_stock_prices_at_crash(stock_data, crash_dates)
        
        if crash_stock_df.empty:
            st.warning("⚠️ No stock data available for identified crash dates.")
            st.stop()
        
        # Save market crash data
        market_crash_df = pd.DataFrame(market_crashes)
        st.session_state.crash_data = market_crash_df
        st.session_state.crash_stock_data = crash_stock_df
        
        # Step 5: Analyze recovery patterns for each stock at crash dates
        status_text.text("📊 Analyzing recovery patterns...")
        progress_bar.progress(0.6)
        
        recovery_results = []
        total_analyses = len(crash_stock_df)
        
        for i, (_, row) in enumerate(crash_stock_df.iterrows()):
            progress_bar.progress(0.6 + (i + 1) / total_analyses * 0.3)
            status_text.text(f"📊 Analyzing recovery {i+1}/{total_analyses}")
            
            symbol = row['Symbol']
            crash_date = row['Crash_Date']
            pre_crash_price = row['Pre_Crash_Price']
            
            if symbol in stock_data:
                recovery = stock_analyzer.analyze_recovery(
                    stock_data[symbol], 
                    crash_date, 
                    pre_crash_price
                )
                recovery['Symbol'] = symbol
                recovery['Crash_Date'] = crash_date
                recovery['Pre_Crash_Price'] = pre_crash_price
                recovery['Crash_Price'] = row['Crash_Price']
                recovery['Drop_Percentage'] = row['Drop_Percentage']
                recovery_results.append(recovery)
        
        recovery_df = pd.DataFrame(recovery_results)
        st.session_state.recovery_data = recovery_df
        
        # Step 4: Generate enhanced statistics
        status_text.text("📈 Generating enhanced statistics...")
        progress_bar.progress(0.9)
        
        enhanced_stats = data_processor.generate_enhanced_statistics(recovery_df, market_stocks)
        st.session_state.enhanced_stats = enhanced_stats
        
        # Step 5: Generate trading insights
        status_text.text("💡 Generating trading insights...")
        progress_bar.progress(0.95)
        
        trading_insights = get_trading_insights(enhanced_stats, market_crash_df)
        st.session_state.trading_insights = trading_insights
        
        # Generate sector rotation insights
        sector_rotation = get_sector_rotation_insights(enhanced_stats)
        st.session_state.sector_rotation = sector_rotation
        
        # Complete
        progress_bar.progress(1.0)
        status_text.text("✅ Analysis complete!")
        st.session_state.analysis_complete = True
        
        time.sleep(1)
        progress_bar.empty()
        status_text.empty()
        
    except Exception as e:
        st.error(f"❌ Error during analysis: {str(e)}")
        st.stop()

# Display results if analysis is complete
if st.session_state.analysis_complete and st.session_state.recovery_data is not None:
    
    # Summary metrics
    st.header("📊 Analysis Summary")
    
    # Display the cross-market analysis information
    if crash_detection_market == analysis_market:
        st.info(f"📈 Analysis of **{analysis_market}** stocks based on **{crash_detection_market}** market crash points")
    else:
        st.success(f"🔄 **Cross-Market Analysis**: Analyzing how **{analysis_market}** stocks respond to **{crash_detection_market}** market crashes")
    
    col1, col2, col3, col4 = st.columns(4)
    
    total_market_crashes = len(st.session_state.crash_data)
    total_stock_instances = len(st.session_state.recovery_data)
    unique_stocks = st.session_state.recovery_data['Symbol'].nunique()
    
    # Handle edge case where Drop_Percentage might have NaN values
    valid_drops = st.session_state.recovery_data['Drop_Percentage'].dropna()
    avg_drop = valid_drops.mean() if len(valid_drops) > 0 else 0.0
    
    with col1:
        st.metric("Market Crashes Identified", total_market_crashes)
    
    with col2:
        st.metric("Stock Recovery Instances", total_stock_instances)
    
    with col3:
        st.metric("Unique Stocks Analyzed", unique_stocks)
    
    with col4:
        # Handle edge case for recovery rate calculation
        valid_recovery_data = st.session_state.recovery_data[st.session_state.recovery_data['Recovery_Category'].notna()]
        if len(valid_recovery_data) > 0:
            recovery_rate = (valid_recovery_data['Recovery_Category'] != 'Unrecoverable').mean() * 100
        else:
            recovery_rate = 0.0
        st.metric("Overall Recovery Rate", f"{recovery_rate:.1f}%")
    
    # Recovery distribution
    st.header("🎯 Recovery Pattern Distribution")
    
    recovery_counts = st.session_state.recovery_data['Recovery_Category'].value_counts()
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        fig_pie = px.pie(
            values=recovery_counts.values,
            names=recovery_counts.index,
            title="Recovery Category Distribution",
            color_discrete_map={
                '7-day': '#2E8B57',
                '15-day': '#1f77b4', 
                '30-day': '#ff7f0e',
                'Unrecoverable': '#dc3545'
            }
        )
        fig_pie.update_traces(textposition='inside', textinfo='percent+label')
        st.plotly_chart(fig_pie, use_container_width=True)
    
    with col2:
        st.subheader("Recovery Statistics")
        for category, count in recovery_counts.items():
            percentage = (count / len(st.session_state.recovery_data)) * 100
            st.metric(f"{category} Recovery", f"{count} ({percentage:.1f}%)")
    
    # Enhanced stock statistics
    if st.session_state.enhanced_stats is not None:
        st.header("🏆 Stock Performance Analysis")
        
        # Top performers
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("🥇 Best Recovery Stocks")
            top_recovery = st.session_state.enhanced_stats.nlargest(10, 'Overall_Recovery_Rate')[
                ['Symbol', 'Company_Name', 'Industry', 'Overall_Recovery_Rate', 'Total_Crashes']
            ]
            st.dataframe(top_recovery, use_container_width=True)
        
        with col2:
            st.subheader("⚠️ Most Crash-Prone Stocks")
            most_crashes = st.session_state.enhanced_stats.nlargest(10, 'Total_Crashes')[
                ['Symbol', 'Company_Name', 'Industry', 'Total_Crashes', 'Overall_Recovery_Rate']
            ]
            st.dataframe(most_crashes, use_container_width=True)
        
        # Industry analysis
        st.subheader("🏭 Industry-wise Recovery Analysis")
        
        industry_stats = st.session_state.enhanced_stats.groupby('Industry').agg({
            'Overall_Recovery_Rate': 'mean',
            'Total_Crashes': 'sum',
            'Symbol': 'count'
        }).round(2)
        industry_stats.columns = ['Avg_Recovery_Rate', 'Total_Crashes', 'Stock_Count']
        industry_stats = industry_stats.sort_values('Avg_Recovery_Rate', ascending=False)
        
        fig_industry = px.bar(
            x=industry_stats.index,
            y=industry_stats['Avg_Recovery_Rate'],
            title="Average Recovery Rate by Industry",
            labels={'x': 'Industry', 'y': 'Average Recovery Rate (%)'}
        )
        fig_industry.update_layout(xaxis_tickangle=45)
        st.plotly_chart(fig_industry, use_container_width=True)
        
        # Detailed stock table
        st.subheader("📋 Detailed Stock Analysis")
        
        # Filter options
        col1, col2, col3 = st.columns(3)
        
        with col1:
            min_recovery_rate = st.slider("Minimum Recovery Rate (%)", 0, 100, 0)
        
        with col2:
            selected_industries = st.multiselect(
                "Filter by Industry",
                options=st.session_state.enhanced_stats['Industry'].unique(),
                default=st.session_state.enhanced_stats['Industry'].unique()
            )
        
        with col3:
            sort_by = st.selectbox(
                "Sort by",
                options=['Overall_Recovery_Rate', 'Total_Crashes', '7_day_Recovery_Rate', 
                        '15_day_Recovery_Rate', '30_day_Recovery_Rate', 'Unrecoverable_Rate']
            )
        
        # Apply filters
        filtered_stats = st.session_state.enhanced_stats[
            (st.session_state.enhanced_stats['Overall_Recovery_Rate'] >= min_recovery_rate) &
            (st.session_state.enhanced_stats['Industry'].isin(selected_industries))
        ].sort_values(sort_by, ascending=False)
        
        # Display table
        display_columns = [
            'Symbol', 'Company_Name', 'Industry', 'Total_Crashes',
            '7_day_Recovery_Rate', '15_day_Recovery_Rate', '30_day_Recovery_Rate',
            'Unrecoverable_Rate', 'Overall_Recovery_Rate'
        ]
        
        st.dataframe(
            filtered_stats[display_columns].style.format({
                '7_day_Recovery_Rate': '{:.1f}%',
                '15_day_Recovery_Rate': '{:.1f}%', 
                '30_day_Recovery_Rate': '{:.1f}%',
                'Unrecoverable_Rate': '{:.1f}%',
                'Overall_Recovery_Rate': '{:.1f}%'
            }),
            use_container_width=True
        )
    
    # Trading Insights Section
    if 'trading_insights' in st.session_state:
        st.header("💡 Trading Insights & Recommendations")
        
        tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(["🟢 Buy on Crash", "🔵 Hold Stocks", "🔴 Sell Candidates", "🏭 Sector Analysis", "🚀 Advanced Analytics", "🔄 Sector Rotation"])
        
        with tab1:
            st.subheader("Stocks to BUY when market crashes")
            st.markdown("**Criteria:** High 7-day recovery rate (>60%) + Overall recovery rate (>80%)")
            buy_stocks = st.session_state.trading_insights['buy_on_crash']
            if not buy_stocks.empty:
                display_cols = ['Symbol', 'Company_Name', 'Industry', '7_day_Recovery_Rate', 'Overall_Recovery_Rate', 'Total_Crashes']
                st.dataframe(
                    buy_stocks[display_cols].style.format({
                        '7_day_Recovery_Rate': '{:.1f}%',
                        'Overall_Recovery_Rate': '{:.1f}%'
                    }),
                    use_container_width=True
                )
            else:
                st.info("No stocks meet the quick recovery criteria with current parameters.")
        
        with tab2:
            st.subheader("Stocks to HOLD during market volatility")
            st.markdown("**Criteria:** High stability score (>70) + Low risk score (<40)")
            hold_stocks = st.session_state.trading_insights['hold_stocks']
            if not hold_stocks.empty:
                display_cols = ['Symbol', 'Company_Name', 'Industry', 'Stability_Score', 'Risk_Score', 'Overall_Recovery_Rate']
                st.dataframe(
                    hold_stocks[display_cols].style.format({
                        'Stability_Score': '{:.1f}',
                        'Risk_Score': '{:.1f}',
                        'Overall_Recovery_Rate': '{:.1f}%'
                    }),
                    use_container_width=True
                )
            else:
                st.info("No stocks meet the stable performance criteria with current parameters.")
        
        with tab3:
            st.subheader("Stocks to consider SELLING before crashes")
            st.markdown("**Criteria:** High unrecoverable rate (>30%) OR High risk score (>70)")
            sell_stocks = st.session_state.trading_insights['sell_candidates']
            if not sell_stocks.empty:
                display_cols = ['Symbol', 'Company_Name', 'Industry', 'Unrecoverable_Rate', 'Risk_Score', 'Overall_Recovery_Rate']
                st.dataframe(
                    sell_stocks[display_cols].style.format({
                        'Unrecoverable_Rate': '{:.1f}%',
                        'Risk_Score': '{:.1f}',
                        'Overall_Recovery_Rate': '{:.1f}%'
                    }),
                    use_container_width=True
                )
            else:
                st.info("No stocks meet the high-risk criteria with current parameters.")
        
        with tab4:
            st.subheader("Sector-wise Recovery Performance")
            sector_perf = st.session_state.trading_insights['sector_performance']
            if not sector_perf.empty:
                # Sector performance chart
                fig_sector = px.bar(
                    x=sector_perf.index,
                    y=sector_perf['Overall_Recovery_Rate'],
                    title="Sector Recovery Performance",
                    labels={'x': 'Industry', 'y': 'Average Recovery Rate (%)'},
                    color=sector_perf['Overall_Recovery_Rate'],
                    color_continuous_scale='RdYlGn'
                )
                fig_sector.update_layout(xaxis_tickangle=45)
                st.plotly_chart(fig_sector, use_container_width=True)
                
                # Sector table
                st.dataframe(
                    sector_perf.style.format({
                        'Overall_Recovery_Rate': '{:.1f}%',
                        'Stability_Score': '{:.1f}',
                        'Risk_Score': '{:.1f}',
                        '7_day_Recovery_Rate': '{:.1f}%',
                        'Average_Drop_Percentage': '{:.1f}%'
                    }),
                    use_container_width=True
                )
        
        with tab5:
            st.subheader("Advanced Market Analytics")
            
            # Crash severity analysis
            if 'crash_severity' in st.session_state.trading_insights:
                crash_severity = st.session_state.trading_insights['crash_severity']
                
                if crash_severity:
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        st.metric("Total Crashes", crash_severity.get('total_crashes', 0))
                    with col2:
                        st.metric("Severe Crashes (>10%)", crash_severity.get('severe_crashes', 0))
                    with col3:
                        st.metric("Average Drop", f"{crash_severity.get('avg_drop', 0):.1f}%")
                    with col4:
                        st.metric("Crashes/Year", f"{crash_severity.get('crash_frequency_per_year', 0):.1f}")
                    
                    # Crash severity distribution
                    severity_data = {
                        'Severe (>10%)': crash_severity.get('severe_crashes', 0),
                        'Moderate (7-10%)': crash_severity.get('moderate_crashes', 0),
                        'Mild (<7%)': crash_severity.get('mild_crashes', 0)
                    }
                    
                    fig_severity = px.pie(
                        values=list(severity_data.values()),
                        names=list(severity_data.keys()),
                        title="Crash Severity Distribution",
                        color_discrete_sequence=['#ff6b6b', '#feca57', '#48cae4']
                    )
                    st.plotly_chart(fig_severity, use_container_width=True)
            
            # Momentum stocks
            st.subheader("Momentum Stocks")
            st.markdown("**High momentum stocks with strong 7-day and 15-day recovery rates**")
            momentum_stocks = st.session_state.trading_insights.get('momentum_stocks', pd.DataFrame())
            if not momentum_stocks.empty:
                display_cols = ['Symbol', 'Company_Name', 'Industry', '7_day_Recovery_Rate', '15_day_Recovery_Rate', 'Overall_Recovery_Rate']
                st.dataframe(
                    momentum_stocks[display_cols].style.format({
                        '7_day_Recovery_Rate': '{:.1f}%',
                        '15_day_Recovery_Rate': '{:.1f}%',
                        'Overall_Recovery_Rate': '{:.1f}%'
                    }),
                    use_container_width=True
                )
            
            # Value opportunities
            st.subheader("Value Opportunities")
            st.markdown("**Stocks with high drops but good recovery potential**")
            value_stocks = st.session_state.trading_insights.get('value_opportunities', pd.DataFrame())
            if not value_stocks.empty:
                display_cols = ['Symbol', 'Company_Name', 'Industry', 'Average_Drop_Percentage', 'Overall_Recovery_Rate', 'Total_Crashes']
                st.dataframe(
                    value_stocks[display_cols].style.format({
                        'Average_Drop_Percentage': '{:.1f}%',
                        'Overall_Recovery_Rate': '{:.1f}%'
                    }),
                    use_container_width=True
                )
        
        with tab6:
            st.subheader("Sector Rotation Strategy")
            
            if 'sector_rotation' in st.session_state:
                rotation = st.session_state.sector_rotation
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.markdown("#### 🛡️ Defensive Sectors")
                    st.markdown("*Low risk, stable during downturns*")
                    defensive = rotation.get('defensive_sectors', pd.DataFrame())
                    if not defensive.empty:
                        st.dataframe(
                            defensive.style.format({
                                'Stability_Score': '{:.1f}',
                                'Risk_Score': '{:.1f}',
                                'Overall_Recovery_Rate': '{:.1f}%'
                            }),
                            use_container_width=True
                        )
                    else:
                        st.info("No defensive sectors identified")
                
                with col2:
                    st.markdown("#### 📈 Growth Sectors")
                    st.markdown("*High recovery rates, good for bull markets*")
                    growth = rotation.get('growth_sectors', pd.DataFrame())
                    if not growth.empty:
                        st.dataframe(
                            growth.style.format({
                                '7_day_Recovery_Rate': '{:.1f}%',
                                'Overall_Recovery_Rate': '{:.1f}%'
                            }),
                            use_container_width=True
                        )
                    else:
                        st.info("No growth sectors identified")
                
                with col3:
                    st.markdown("#### 🔄 Cyclical Sectors")
                    st.markdown("*High volatility but strong recovery*")
                    cyclical = rotation.get('cyclical_sectors', pd.DataFrame())
                    if not cyclical.empty:
                        st.dataframe(
                            cyclical.style.format({
                                'Average_Drop_Percentage': '{:.1f}%',
                                'Overall_Recovery_Rate': '{:.1f}%',
                                'Risk_Score': '{:.1f}'
                            }),
                            use_container_width=True
                        )
                    else:
                        st.info("No cyclical sectors identified")
    
    # Recovery timeline visualization
    if crash_detection_market == analysis_market:
        st.header("📅 Market Crash Timeline Analysis")
    else:
        st.header(f"📅 {crash_detection_market} Crash Timeline Analysis with {analysis_market} Stock Recovery")
    
    # Monthly crash distribution
    crash_data_copy = st.session_state.crash_data.copy()
    crash_data_copy['Month'] = pd.to_datetime(crash_data_copy['Date']).dt.to_period('M')
    monthly_crashes = crash_data_copy.groupby('Month').size().reset_index()
    monthly_crashes['Month_str'] = monthly_crashes['Month'].astype(str)
    
    if len(monthly_crashes) > 0:
        fig_timeline = px.bar(
            monthly_crashes,
            x='Month_str',
            y=0,
            title=f"Monthly {crash_detection_market} Crash Distribution",
            labels={'Month_str': 'Month', '0': 'Number of Market Crashes'}
        )
        fig_timeline.update_layout(
            xaxis_title="Month",
            yaxis_title="Number of Market Crashes",
            xaxis_tickangle=45,
            showlegend=False
        )
        st.plotly_chart(fig_timeline, use_container_width=True)
    else:
        st.info("No crash data available for timeline visualization")
    
    # Market crash details table
    st.subheader("📋 Market Crash Details")
    crash_display = crash_data_copy[['Date', 'Pre_Crash_Index', 'Crash_Index', 'Drop_Percentage', 'Close_Index']].copy()
    crash_display['Date'] = crash_display['Date'].dt.strftime('%Y-%m-%d')
    crash_display = crash_display.round(2)
    st.dataframe(crash_display, use_container_width=True)
    
    # Additional visualizations
    st.header("📊 Additional Market Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Recovery vs Drop relationship
        st.subheader("📈 Recovery vs Drop Severity")
        
        recovery_scatter_data = st.session_state.recovery_data.copy()
        recovery_scatter_data = recovery_scatter_data[recovery_scatter_data['Days_to_Recovery'].notna()]
        
        if not recovery_scatter_data.empty:
            fig_scatter = px.scatter(
                recovery_scatter_data,
                x='Drop_Percentage',
                y='Days_to_Recovery',
                color='Recovery_Category',
                hover_data=['Symbol', 'Crash_Date'],
                title="Recovery Time vs Drop Severity",
                labels={'Drop_Percentage': 'Drop Percentage (%)', 'Days_to_Recovery': 'Days to Recovery'},
                color_discrete_map={
                    '7-day': '#2E8B57',
                    '15-day': '#1f77b4',
                    '30-day': '#ff7f0e', 
                    'Unrecoverable': '#dc3545'
                }
            )
            st.plotly_chart(fig_scatter, use_container_width=True)
        else:
            st.info("No recovery data available for scatter plot")
    
    with col2:
        # Industry recovery performance
        st.subheader("🏭 Industry Recovery Heatmap")
        
        if st.session_state.enhanced_stats is not None:
            industry_pivot = st.session_state.enhanced_stats.pivot_table(
                values=['7_day_Recovery_Rate', '15_day_Recovery_Rate', '30_day_Recovery_Rate', 'Unrecoverable_Rate'],
                index='Industry',
                aggfunc='mean'
            ).round(1)
            
            if not industry_pivot.empty:
                fig_heatmap = px.imshow(
                    industry_pivot.T,
                    title="Recovery Rate by Industry and Timeframe",
                    labels=dict(x="Industry", y="Recovery Window", color="Rate (%)"),
                    aspect="auto",
                    color_continuous_scale="RdYlGn"
                )
                fig_heatmap.update_layout(xaxis_tickangle=45)
                st.plotly_chart(fig_heatmap, use_container_width=True)
    
    # Trading insights visualization section
    st.header("💰 Trading Insights Visualization")
    
    if 'trading_insights' in st.session_state:
        insights = st.session_state.trading_insights
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Buy recommendations distribution by industry
            st.subheader("🟢 Buy Recommendations by Industry")
            buy_stocks = insights.get('buy_on_crash', pd.DataFrame())
            if not buy_stocks.empty:
                buy_industry_dist = buy_stocks['Industry'].value_counts()
                fig_buy_dist = px.pie(
                    values=buy_industry_dist.values,
                    names=buy_industry_dist.index,
                    title="Buy Recommendations Distribution"
                )
                st.plotly_chart(fig_buy_dist, use_container_width=True)
                
                # Display buy recommendations table
                st.dataframe(
                    buy_stocks[['Symbol', 'Company_Name', 'Industry', '7_day_Recovery_Rate', 'Overall_Recovery_Rate']].style.format({
                        '7_day_Recovery_Rate': '{:.1f}%',
                        'Overall_Recovery_Rate': '{:.1f}%'
                    }),
                    use_container_width=True
                )
            else:
                st.info("No buy recommendations with current criteria")
        
        with col2:
            # Risk distribution
            st.subheader("🔴 Risk Analysis")
            sell_stocks = insights.get('sell_candidates', pd.DataFrame())
            if not sell_stocks.empty:
                risk_industry_dist = sell_stocks['Industry'].value_counts()
                fig_risk_dist = px.pie(
                    values=risk_industry_dist.values,
                    names=risk_industry_dist.index,
                    title="High-Risk Stocks by Industry"
                )
                st.plotly_chart(fig_risk_dist, use_container_width=True)
                
                # Display sell candidates table
                st.dataframe(
                    sell_stocks[['Symbol', 'Company_Name', 'Industry', 'Risk_Score', 'Unrecoverable_Rate']].style.format({
                        'Risk_Score': '{:.1f}',
                        'Unrecoverable_Rate': '{:.1f}%'
                    }),
                    use_container_width=True
                )
            else:
                st.info("No high-risk stocks identified with current criteria")
    
    # Advanced analytics visualization
    st.header("🚀 Advanced Analytics Dashboard")
    
    if 'trading_insights' in st.session_state:
        insights = st.session_state.trading_insights
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Momentum stocks performance
            st.subheader("🚀 Momentum Stocks Analysis")
            momentum_stocks = insights.get('momentum_stocks', pd.DataFrame())
            if not momentum_stocks.empty:
                fig_momentum = px.scatter(
                    momentum_stocks.head(20),
                    x='7_day_Recovery_Rate',
                    y='15_day_Recovery_Rate',
                    size='Total_Crashes',
                    color='Industry',
                    hover_data=['Symbol', 'Company_Name'],
                    title="Momentum Stocks: 7-day vs 15-day Recovery",
                    labels={'7_day_Recovery_Rate': '7-day Recovery Rate (%)', '15_day_Recovery_Rate': '15-day Recovery Rate (%)'}
                )
                st.plotly_chart(fig_momentum, use_container_width=True)
                
                # Display momentum stocks table
                st.dataframe(
                    momentum_stocks[['Symbol', 'Company_Name', 'Industry', '7_day_Recovery_Rate', '15_day_Recovery_Rate', 'Overall_Recovery_Rate']].style.format({
                        '7_day_Recovery_Rate': '{:.1f}%',
                        '15_day_Recovery_Rate': '{:.1f}%',
                        'Overall_Recovery_Rate': '{:.1f}%'
                    }),
                    use_container_width=True
                )
            else:
                st.info("No momentum stocks identified")
        
        with col2:
            # Value opportunities
            st.subheader("💎 Value Opportunities")
            value_stocks = insights.get('value_opportunities', pd.DataFrame())
            if not value_stocks.empty:
                fig_value = px.scatter(
                    value_stocks.head(20),
                    x='Average_Drop_Percentage',
                    y='Overall_Recovery_Rate',
                    size='Total_Crashes',
                    color='Industry',
                    hover_data=['Symbol', 'Company_Name'],
                    title="Value Opportunities: Drop vs Recovery",
                    labels={'Average_Drop_Percentage': 'Average Drop (%)', 'Overall_Recovery_Rate': 'Overall Recovery Rate (%)'}
                )
                st.plotly_chart(fig_value, use_container_width=True)
                
                # Display value opportunities table
                st.dataframe(
                    value_stocks[['Symbol', 'Company_Name', 'Industry', 'Average_Drop_Percentage', 'Overall_Recovery_Rate', 'Total_Crashes']].style.format({
                        'Average_Drop_Percentage': '{:.1f}%',
                        'Overall_Recovery_Rate': '{:.1f}%'
                    }),
                    use_container_width=True
                )
            else:
                st.info("No value opportunities identified")
    
    # Performance comparison charts
    st.header("📈 Performance Comparison Charts")
    
    if st.session_state.enhanced_stats is not None:
        col1, col2 = st.columns(2)
        
        with col1:
            # Top vs Bottom performers
            st.subheader("🏆 Top vs Bottom Performers")
            top_performers = st.session_state.enhanced_stats.nlargest(10, 'Overall_Recovery_Rate')
            bottom_performers = st.session_state.enhanced_stats.nsmallest(10, 'Overall_Recovery_Rate')
            
            comparison_data = pd.concat([
                top_performers[['Symbol', 'Overall_Recovery_Rate']].assign(Category='Top 10'),
                bottom_performers[['Symbol', 'Overall_Recovery_Rate']].assign(Category='Bottom 10')
            ])
            
            fig_comparison = px.bar(
                comparison_data,
                x='Symbol',
                y='Overall_Recovery_Rate',
                color='Category',
                title="Top 10 vs Bottom 10 Recovery Rates",
                labels={'Overall_Recovery_Rate': 'Recovery Rate (%)'}
            )
            fig_comparison.update_layout(xaxis_tickangle=45)
            st.plotly_chart(fig_comparison, use_container_width=True)
        
        with col2:
            # Stability vs Risk scatter
            st.subheader("⚖️ Stability vs Risk Analysis")
            fig_stability_risk = px.scatter(
                st.session_state.enhanced_stats,
                x='Risk_Score',
                y='Stability_Score',
                size='Total_Crashes',
                color='Industry',
                hover_data=['Symbol', 'Company_Name'],
                title="Stability Score vs Risk Score",
                labels={'Risk_Score': 'Risk Score', 'Stability_Score': 'Stability Score'}
            )
            st.plotly_chart(fig_stability_risk, use_container_width=True)
    
    # Data export section
    st.header("📤 Export Data")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        csv_market_crashes = st.session_state.crash_data.to_csv(index=False)
        st.download_button(
            label="📊 Market Crashes CSV",
            data=csv_market_crashes,
            file_name=f"{crash_detection_market}_crashes_{datetime.now().strftime('%Y%m%d')}.csv",
            mime="text/csv"
        )
    
    with col2:
        if 'crash_stock_data' in st.session_state:
            csv_crash_stocks = st.session_state.crash_stock_data.to_csv(index=False)
            st.download_button(
                label="📈 Stock at Crashes CSV",
                data=csv_crash_stocks,
                file_name=f"{analysis_market}_stocks_at_{crash_detection_market}_crashes_{datetime.now().strftime('%Y%m%d')}.csv",
                mime="text/csv"
            )
    
    with col3:
        csv_recovery = st.session_state.recovery_data.to_csv(index=False)
        st.download_button(
            label="📈 Recovery Analysis CSV", 
            data=csv_recovery,
            file_name=f"{analysis_market}_recovery_after_{crash_detection_market}_crashes_{datetime.now().strftime('%Y%m%d')}.csv",
            mime="text/csv"
        )
    
    with col4:
        if st.session_state.enhanced_stats is not None:
            csv_stats = st.session_state.enhanced_stats.to_csv(index=False)
            st.download_button(
                label="📋 Enhanced Stats CSV",
                data=csv_stats, 
                file_name=f"enhanced_stats_{datetime.now().strftime('%Y%m%d')}.csv",
                mime="text/csv"
            )
    
    # Trading insights export
    if 'trading_insights' in st.session_state:
        st.subheader("📈 Trading Insights Export")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            buy_csv = st.session_state.trading_insights['buy_on_crash'].to_csv(index=False)
            st.download_button(
                label="🟢 Buy Recommendations CSV",
                data=buy_csv,
                file_name=f"buy_recommendations_{datetime.now().strftime('%Y%m%d')}.csv",
                mime="text/csv"
            )
        
        with col2:
            hold_csv = st.session_state.trading_insights['hold_stocks'].to_csv(index=False)
            st.download_button(
                label="🔵 Hold Recommendations CSV",
                data=hold_csv,
                file_name=f"hold_recommendations_{datetime.now().strftime('%Y%m%d')}.csv",
                mime="text/csv"
            )
        
        with col3:
            sell_csv = st.session_state.trading_insights['sell_candidates'].to_csv(index=False)
            st.download_button(
                label="🔴 Sell Candidates CSV",
                data=sell_csv,
                file_name=f"sell_candidates_{datetime.now().strftime('%Y%m%d')}.csv",
                mime="text/csv"
            )
        
        with col4:
            sector_csv = st.session_state.trading_insights['sector_performance'].to_csv()
            st.download_button(
                label="🏭 Sector Analysis CSV",
                data=sector_csv,
                file_name=f"sector_analysis_{datetime.now().strftime('%Y%m%d')}.csv",
                mime="text/csv"
            )
        
        # Additional insights export
        st.subheader("📊 Advanced Analytics Export")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            momentum_csv = st.session_state.trading_insights.get('momentum_stocks', pd.DataFrame()).to_csv(index=False)
            st.download_button(
                label="🚀 Momentum Stocks CSV",
                data=momentum_csv,
                file_name=f"momentum_stocks_{datetime.now().strftime('%Y%m%d')}.csv",
                mime="text/csv"
            )
        
        with col2:
            value_csv = st.session_state.trading_insights.get('value_opportunities', pd.DataFrame()).to_csv(index=False)
            st.download_button(
                label="💎 Value Opportunities CSV",
                data=value_csv,
                file_name=f"value_opportunities_{datetime.now().strftime('%Y%m%d')}.csv",
                mime="text/csv"
            )
        
        with col3:
            if 'sector_rotation' in st.session_state:
                # Combine all sector rotation data
                rotation_data = []
                for strategy, data in st.session_state.sector_rotation.items():
                    if not data.empty:
                        temp_data = data.copy()
                        temp_data['Strategy'] = strategy
                        rotation_data.append(temp_data)
                
                if rotation_data:
                    combined_rotation = pd.concat(rotation_data, ignore_index=True)
                    rotation_csv = combined_rotation.to_csv(index=False)
                    st.download_button(
                        label="🔄 Sector Rotation CSV",
                        data=rotation_csv,
                        file_name=f"sector_rotation_{datetime.now().strftime('%Y%m%d')}.csv",
                        mime="text/csv"
                    )
    
    # Auto-save feature
    st.subheader("💾 Auto-Save Feature")
    if st.button("💾 Save All Data as CSV Files"):
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Save market crashes
        st.session_state.crash_data.to_csv(f"{crash_detection_market}_crashes_{timestamp}.csv", index=False)
        
        # Save stock data at crashes
        if 'crash_stock_data' in st.session_state:
            st.session_state.crash_stock_data.to_csv(f"{analysis_market}_stocks_at_{crash_detection_market}_crashes_{timestamp}.csv", index=False)
        
        # Save recovery analysis
        st.session_state.recovery_data.to_csv(f"{analysis_market}_recovery_after_{crash_detection_market}_crashes_{timestamp}.csv", index=False)
        
        # Save enhanced statistics
        if st.session_state.enhanced_stats is not None:
            st.session_state.enhanced_stats.to_csv(f"enhanced_stats_{timestamp}.csv", index=False)
        
        # Save trading insights
        if 'trading_insights' in st.session_state:
            insights = st.session_state.trading_insights
            insights['buy_on_crash'].to_csv(f"buy_recommendations_{timestamp}.csv", index=False)
            insights['hold_stocks'].to_csv(f"hold_recommendations_{timestamp}.csv", index=False)
            insights['sell_candidates'].to_csv(f"sell_candidates_{timestamp}.csv", index=False)
            insights['sector_performance'].to_csv(f"sector_analysis_{timestamp}.csv")
            
            # Save advanced analytics
            if 'momentum_stocks' in insights:
                insights['momentum_stocks'].to_csv(f"momentum_stocks_{timestamp}.csv", index=False)
            if 'value_opportunities' in insights:
                insights['value_opportunities'].to_csv(f"value_opportunities_{timestamp}.csv", index=False)
        
        # Save sector rotation insights
        if 'sector_rotation' in st.session_state:
            rotation_data = []
            for strategy, data in st.session_state.sector_rotation.items():
                if not data.empty:
                    temp_data = data.copy()
                    temp_data['Strategy'] = strategy
                    rotation_data.append(temp_data)
            
            if rotation_data:
                combined_rotation = pd.concat(rotation_data, ignore_index=True)
                combined_rotation.to_csv(f"sector_rotation_{timestamp}.csv", index=False)
        
        st.success(f"✅ All data and trading insights saved with timestamp {timestamp}")
        st.info("Files saved in the current directory with timestamp for easy identification.")

# Footer
st.markdown("---")
st.markdown("""
**About this application:**
- Cross-market analysis: Identify crashes in one market and analyze stocks from the same or different market 
- Market crash detection: Based on major Indian index movements (NIFTY 50, NIFTY 500, BANKNIFTY, etc.)
- Data source: Yahoo Finance via yfinance  
- Stock recovery analysis: Individual stock recovery after market crashes
- Recovery categories: 7-day, 15-day, 30-day, and Unrecoverable
- Trading insights: Buy/Hold/Sell recommendations based on recovery patterns
- Analysis approach: Market-level crash identification → Stock-level recovery tracking → Trading recommendations
""")
