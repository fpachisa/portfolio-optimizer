# performance_analysis.py
import pandas as pd
import numpy as np
import traceback

def calculate_security_analysis_returns(daily_returns, analysis_start_date, analysis_end_date):
    """
    Calculates the annualized total logarithmic return for each security over the 
    analysis period.

    Log returns are calculated daily, summed over the period, and then annualized.

    Args:
        daily_returns (pd.DataFrame): DataFrame of *arithmetic* daily returns for all assets.
        analysis_start_date (pd.Timestamp): Start date for the analysis period.
        analysis_end_date (pd.Timestamp): End date for the analysis period.

    Returns:
        pd.DataFrame: DataFrame with Ticker and Annualized Total Log Return, or None if error.
    """
    print("\nCalculating annualized total log returns for each security during analysis period...")
    try:
        # Filter arithmetic returns for the analysis period
        returns_period = daily_returns.loc[analysis_start_date:analysis_end_date]

        if returns_period.empty:
            print(f"Warning: No daily returns data found in the analysis period "
                  f"{analysis_start_date.date()} to {analysis_end_date.date()}. Cannot calculate security log returns.")
            return None

        # Calculate daily log returns: log(1 + arithmetic_return)
        # Add a small epsilon to avoid log(0) if any arithmetic return is exactly -1
        log_returns_period = np.log(1 + returns_period.clip(lower=-0.999999)) 

        # Sum the daily log returns over the period for each security
        total_log_return = log_returns_period.sum()

        # Annualize the total log return
        # Get the number of trading days in the actual analysis period
        num_actual_days = len(returns_period.index)
        
        if num_actual_days == 0:
             print(f"Warning: Zero trading days found in the analysis period {analysis_start_date.date()} to {analysis_end_date.date()}. Cannot annualize.")
             # Return zero or NaN returns based on preference
             annualized_total_log_returns = total_log_return * 0.0 # Set to zero
        else:
            # Use 252 trading days for annualization scale factor
            trading_days_per_year = 252
            annualization_factor = trading_days_per_year / num_actual_days
            annualized_total_log_returns = total_log_return * annualization_factor

        # Format as DataFrame
        results_df = annualized_total_log_returns.reset_index() # Convert Series to DataFrame
        # Rename columns appropriately
        results_df.columns = ['Ticker', 'Annualized_Total_Log_Return'] 

        print("Security annualized total log return calculation complete.")
        return results_df

    except Exception as e:
        print(f"Error calculating security annualized total log returns: {e}")
        traceback.print_exc()
        return None

def calculate_max_drawdown(returns_series):
    """
    Calculates the maximum drawdown for a given returns series.

    Args:
        returns_series (pd.Series): Series of arithmetic returns (e.g., daily).

    Returns:
        float: Maximum drawdown as a negative decimal (e.g., -0.1 for -10%). 
               Returns 0.0 if input is empty or invalid.
    """
    if returns_series is None or returns_series.empty:
        return 0.0
        
    # Calculate cumulative returns (wealth index)
    cumulative_returns = (1 + returns_series).cumprod()
    
    # Calculate running maximum
    running_max = cumulative_returns.cummax()
    
    # Calculate drawdown
    drawdown = (cumulative_returns / running_max) - 1
    
    # Find the minimum drawdown (most negative value)
    max_drawdown = drawdown.min()
    
    # Handle cases where there are no losses (max_drawdown would be 0 or NaN)
    return max_drawdown if pd.notna(max_drawdown) else 0.0

def calculate_portfolio_performance(daily_returns, weekly_returns, weights_series,
                                    analysis_start_date, analysis_end_date, risk_free_rate,
                                    security_annualized_log_returns=None): # MODIFIED: Added new optional arg
    """
    Calculates extended annualized performance and risk metrics for a single
    portfolio over a specific period.

    Args:
        daily_returns (pd.DataFrame): DataFrame of daily arithmetic returns for all assets.
        weekly_returns (pd.DataFrame): DataFrame of weekly arithmetic returns for all assets.
        weights_series (pd.Series): Weights for a single portfolio (Index=Ticker, Values=Weight).
        analysis_start_date (pd.Timestamp): Start date for the analysis period.
        analysis_end_date (pd.Timestamp): End date for the analysis period.
        risk_free_rate (float): Annual risk-free rate.
        security_annualized_log_returns (pd.Series, optional): Series of annualized total log returns
                                                               for securities (Index=Ticker).
                                                               Defaults to None.

    Returns:
        dict: Dictionary containing performance and risk metrics.
              Returns None if calculation fails critically.
    """
    portfolio_name = weights_series.name if weights_series.name else 'Unnamed Portfolio'
    metrics = { # Initialize with NaN or 0.0
        'Annualized Return': np.nan,
        'Annualized Volatility (Daily)': np.nan,
        'Sharpe Ratio': np.nan,
        'Max Drawdown': 0.0, # Max drawdown is 0 if no losses
        'Annualized Volatility (Weekly)': np.nan,
        'Skewness (Daily)': np.nan,
        'Kurtosis (Daily)': np.nan,
        'Weighted Ann. Sec. Log Return': np.nan # ADDED: New metric
    }

    try:
        # --- Input Validation and Filtering ---
        if daily_returns is None or weekly_returns is None or weights_series is None:
             print(f"Error: Missing input data for portfolio '{portfolio_name}'.")
             return None # Critical failure

        active_tickers = weights_series[weights_series > 1e-6].index.tolist()
        if not active_tickers:
             print(f"Warning: Portfolio '{portfolio_name}' has no active tickers. Returning zero/NaN metrics.")
             metrics.update({k: 0.0 for k in metrics if k not in ['Max Drawdown', 'Weighted Ann. Sec. Log Return']})
             return metrics

        available_tickers_daily = daily_returns.columns.intersection(active_tickers)
        available_tickers_weekly = weekly_returns.columns.intersection(active_tickers)

        if not available_tickers_daily.tolist():
            print(f"Warning: No daily returns data found for active tickers in portfolio '{portfolio_name}'. Cannot calculate daily stats.")
        if not available_tickers_weekly.tolist():
             print(f"Warning: No weekly returns data found for active tickers in portfolio '{portfolio_name}'. Cannot calculate weekly volatility.")
             if not available_tickers_daily.tolist() and security_annualized_log_returns is None: # If no data for any calc
                 return metrics

        daily_returns_period = daily_returns.loc[analysis_start_date:analysis_end_date, available_tickers_daily]
        weekly_returns_period = weekly_returns.loc[analysis_start_date:analysis_end_date, available_tickers_weekly]

        daily_returns_period = daily_returns_period.dropna(axis=1, how='all')
        weekly_returns_period = weekly_returns_period.dropna(axis=1, how='all')

        # --- Alignment and Renormalization ---
        aligned_weights_daily = pd.Series(dtype=float) # Initialize
        if not daily_returns_period.empty:
            aligned_weights_daily = weights_series.loc[daily_returns_period.columns]
            if not np.isclose(aligned_weights_daily.sum(), 1.0):
                if aligned_weights_daily.sum() != 0:
                     aligned_weights_daily = aligned_weights_daily / aligned_weights_daily.sum()
                else:
                    aligned_weights_daily = pd.Series(dtype=float)
        elif security_annualized_log_returns is not None: # If daily returns are empty but we have log returns for the new metric
            # Align weights based on tickers present in security_annualized_log_returns and original weights
            tickers_for_log_metric = security_annualized_log_returns.index.intersection(weights_series.index)
            aligned_weights_daily = weights_series.loc[tickers_for_log_metric]
            if not np.isclose(aligned_weights_daily.sum(), 1.0) and aligned_weights_daily.sum() != 0:
                aligned_weights_daily = aligned_weights_daily / aligned_weights_daily.sum()


        aligned_weights_weekly = pd.Series(dtype=float)
        if not weekly_returns_period.empty:
            aligned_weights_weekly = weights_series.loc[weekly_returns_period.columns]
            if not np.isclose(aligned_weights_weekly.sum(), 1.0):
                 if aligned_weights_weekly.sum() != 0:
                     aligned_weights_weekly = aligned_weights_weekly / aligned_weights_weekly.sum()
                 else:
                     aligned_weights_weekly = pd.Series(dtype=float)

        # --- Calculate Portfolio Returns (Arithmetic for standard metrics) ---
        portfolio_daily_returns = pd.Series(dtype=float)
        if not daily_returns_period.empty and not aligned_weights_daily.empty:
            # Ensure weights and returns columns are perfectly aligned before dot product
            common_idx = daily_returns_period.columns.intersection(aligned_weights_daily.index)
            portfolio_daily_returns = daily_returns_period[common_idx].dot(aligned_weights_daily[common_idx])


        portfolio_weekly_returns = pd.Series(dtype=float)
        if not weekly_returns_period.empty and not aligned_weights_weekly.empty:
             # Ensure weights and returns columns are perfectly aligned
            common_idx_wk = weekly_returns_period.columns.intersection(aligned_weights_weekly.index)
            portfolio_weekly_returns = weekly_returns_period[common_idx_wk].dot(aligned_weights_weekly[common_idx_wk])


        # --- Calculate Metrics ---
        trading_days_per_year = 252
        trading_weeks_per_year = 52

        # Metrics based on Daily Arithmetic Returns
        if not portfolio_daily_returns.empty:
            mean_daily_return = portfolio_daily_returns.mean()
            std_dev_daily_return = portfolio_daily_returns.std()

            metrics['Annualized Return'] = mean_daily_return * trading_days_per_year # This is arithmetic
            metrics['Annualized Volatility (Daily)'] = std_dev_daily_return * np.sqrt(trading_days_per_year)

            if metrics['Annualized Volatility (Daily)'] is not None and metrics['Annualized Volatility (Daily)'] > 1e-9:
                metrics['Sharpe Ratio'] = (metrics['Annualized Return'] - risk_free_rate) / metrics['Annualized Volatility (Daily)']
            else:
                metrics['Sharpe Ratio'] = 0.0

            metrics['Max Drawdown'] = calculate_max_drawdown(portfolio_daily_returns)
            metrics['Skewness (Daily)'] = portfolio_daily_returns.skew()
            metrics['Kurtosis (Daily)'] = portfolio_daily_returns.kurt()

        # Metrics based on Weekly Returns
        if not portfolio_weekly_returns.empty:
            std_dev_weekly_return = portfolio_weekly_returns.std()
            metrics['Annualized Volatility (Weekly)'] = std_dev_weekly_return * np.sqrt(trading_weeks_per_year)

        # ADDED: Calculate Weighted Annualized Security Log Return
        if security_annualized_log_returns is not None and not aligned_weights_daily.empty:
            # Ensure security_annualized_log_returns is aligned with the portfolio's weights
            # Use aligned_weights_daily as it represents the weights for the assets in the current portfolio context
            common_tickers_for_log = aligned_weights_daily.index.intersection(security_annualized_log_returns.index)
            
            if not common_tickers_for_log.empty:
                aligned_sec_log_returns = security_annualized_log_returns.loc[common_tickers_for_log]
                current_portfolio_weights = aligned_weights_daily.loc[common_tickers_for_log]
                
                # Renormalize these specific weights if their sum isn't 1, to ensure the sum-product is meaningful for this subset
                if not np.isclose(current_portfolio_weights.sum(),1.0) and current_portfolio_weights.sum() !=0:
                    renorm_weights = current_portfolio_weights / current_portfolio_weights.sum()
                else:
                    renorm_weights = current_portfolio_weights

                metrics['Weighted Ann. Sec. Log Return'] = (renorm_weights * aligned_sec_log_returns).sum()
            else:
                print(f"Warning: No common tickers between aligned_weights_daily and security_annualized_log_returns for portfolio {portfolio_name}")


        print(f"Performance calculated for portfolio: {portfolio_name}")
        return metrics

    except KeyError as ke:
        print(f"Error calculating performance for portfolio '{portfolio_name}' (KeyError): {ke}. Check ticker alignment.")
        traceback.print_exc()
        return None
    except Exception as e:
        print(f"Error calculating performance for portfolio '{portfolio_name}': {e}")
        traceback.print_exc()
        return None

 

def run_performance_analysis(daily_returns, weekly_returns, optimized_weights_df,
                             analysis_start_date, analysis_end_date, risk_free_rate,
                             security_analysis_returns_df=None): # MODIFIED: Added new optional arg
    """
    Runs performance analysis for all optimized portfolios using the extended calculation.

    Args:
        daily_returns (pd.DataFrame): Full DataFrame of daily returns.
        weekly_returns (pd.DataFrame): Full DataFrame of weekly returns.
        optimized_weights_df (pd.DataFrame): DataFrame with weights for all strategies
                                             (Columns: Strategy Names, Index: Tickers).
        analysis_start_date (pd.Timestamp): Start date for analysis.
        analysis_end_date (pd.Timestamp): End date for analysis.
        risk_free_rate (float): Annual risk-free rate.
        security_analysis_returns_df (pd.DataFrame, optional): DataFrame with 'Ticker' and
                                                              'Annualized_Total_Log_Return'.
                                                              Defaults to None.

    Returns:
        pd.DataFrame: DataFrame containing performance metrics for each portfolio strategy.
                      Index is Metric Name, Columns are Strategy Names.
                      Returns None if the process fails for all portfolios.
    """
    print("\n=== Running Performance Analysis (Extended Metrics) ===")
    performance_results = {}

    if daily_returns is None or daily_returns.empty:
         print("Error: Cannot run performance analysis without daily returns data.")
         return None
    if weekly_returns is None or weekly_returns.empty:
         print("Error: Cannot run performance analysis without weekly returns data for weekly volatility.")
         return None
    if optimized_weights_df is None or optimized_weights_df.empty:
         print("Error: Cannot run performance analysis without optimized weights.")
         return None

    # Prepare security_annualized_log_returns Series if data is available
    security_log_returns_series = None
    if security_analysis_returns_df is not None and \
       'Ticker' in security_analysis_returns_df.columns and \
       'Annualized_Total_Log_Return' in security_analysis_returns_df.columns:
        security_log_returns_series = security_analysis_returns_df.set_index('Ticker')['Annualized_Total_Log_Return']
    elif security_analysis_returns_df is not None:
        print("Warning: security_analysis_returns_df passed to run_performance_analysis is missing expected columns ('Ticker', 'Annualized_Total_Log_Return'). Cannot calculate 'Weighted Ann. Sec. Log Return'.")


    for portfolio_name in optimized_weights_df.columns:
        weights_series = optimized_weights_df[portfolio_name]

        portfolio_metrics = calculate_portfolio_performance(
                                          daily_returns,
                                          weekly_returns,
                                          weights_series,
                                          analysis_start_date,
                                          analysis_end_date,
                                          risk_free_rate,
                                          security_annualized_log_returns=security_log_returns_series) # MODIFIED: Pass it here

        if portfolio_metrics is not None:
            performance_results[portfolio_name] = portfolio_metrics
        else:
            print(f"Critical error calculating performance for portfolio: {portfolio_name}. Results for this portfolio will be missing.")

    if not performance_results:
        print("Error: Performance analysis failed critically for all portfolios.")
        return None

    performance_df = pd.DataFrame(performance_results)
    print("\n--- Performance Analysis Finished ---")
    return performance_df

def calculate_correlation_matrix(daily_returns, analysis_start_date, analysis_end_date, relevant_tickers):
    """
    Calculates the correlation matrix of asset returns over the analysis period.

    Args:
        daily_returns (pd.DataFrame): DataFrame of daily returns for all assets.
        analysis_start_date (pd.Timestamp): Start date for the analysis period.
        analysis_end_date (pd.Timestamp): End date for the analysis period.
        relevant_tickers (list): List of tickers present in the optimized portfolios.

    Returns:
        pd.DataFrame: Correlation matrix, or None if error.
    """
    print("\nCalculating asset correlation matrix for the analysis period...")
    try:
        # Filter returns for the analysis period and relevant tickers
        returns_period = daily_returns.loc[analysis_start_date:analysis_end_date, relevant_tickers]


        if returns_period.empty or returns_period.shape[1] < 2:
             print("Warning: Insufficient data or assets to calculate correlation matrix for the analysis period.")
             return None


        # Calculate correlation matrix
        correlation_matrix = returns_period.corr()


        print("Correlation matrix calculation complete.")
        return correlation_matrix


    except KeyError:
         print("Error: Some relevant tickers not found in daily returns columns.")
         return None
    except Exception as e:
        print(f"Error calculating correlation matrix: {e}")
        return None
    
