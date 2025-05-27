# portfolio_optimization.py
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.covariance import LedoitWolf

def calculate_robust_covariance(weekly_returns, optimization_date):
    """
    Calculates the annualized Ledoit-Wolf shrunk covariance matrix.

    Args:
        weekly_returns (pd.DataFrame): DataFrame of weekly returns for all assets.
        optimization_date (pd.Timestamp): The date up to which returns are considered.

    Returns:
        pd.DataFrame: Annualized robust covariance matrix, or None if error.
    """
    try:
        # Filter returns up to the optimization date
        returns_for_cov = weekly_returns[weekly_returns.index <= optimization_date]

        if returns_for_cov.empty or returns_for_cov.isnull().all().any():
             print(f"Error: No valid weekly returns data found up to {optimization_date.date()} for covariance calculation.")
             # Check which assets have issues
             if not returns_for_cov.empty:
                 print("Assets with all NaN returns in period:", returns_for_cov.columns[returns_for_cov.isnull().all()].tolist())
             return None

        # Drop assets with insufficient data (e.g., all NaNs) within the period
        returns_for_cov = returns_for_cov.dropna(axis=1, how='all')
        if returns_for_cov.shape[1] < 2: # Need at least 2 assets
             print("Error: Need at least two assets with valid data for covariance calculation.")
             return None

        print(f"\nCalculating robust covariance matrix using data up to {optimization_date.date()} for {returns_for_cov.shape[1]} assets...")

        # Fit Ledoit-Wolf estimator
        lw = LedoitWolf().fit(returns_for_cov)
        cov_matrix_robust = lw.covariance_

        # Annualize (assuming 52 weeks)
        cov_matrix_robust_annualized = cov_matrix_robust * 52

        # Convert back to DataFrame for clarity
        cov_matrix_df = pd.DataFrame(cov_matrix_robust_annualized,
                                     index=returns_for_cov.columns,
                                     columns=returns_for_cov.columns)

        print("Robust covariance matrix calculation complete.")
        return cov_matrix_df

    except Exception as e:
        print(f"Error calculating robust covariance matrix: {e}")
        return None

def calculate_expected_returns(weekly_returns, optimization_date):
    """
    Calculates annualized expected returns (mean of historical weekly returns).

    Args:
        weekly_returns (pd.DataFrame): DataFrame of weekly returns.
        optimization_date (pd.Timestamp): The date up to which returns are considered.

    Returns:
        pd.Series: Annualized expected returns for each asset, or None if error.
    """
    try:
        returns_for_mean = weekly_returns[weekly_returns.index <= optimization_date]

        if returns_for_mean.empty:
             print(f"Error: No weekly returns data found up to {optimization_date.date()} for expected return calculation.")
             return None

        # Calculate mean weekly return
        mean_weekly_returns = returns_for_mean.mean()

        # Annualize
        expected_returns_annualized = mean_weekly_returns * 52

        print("Annualized expected returns calculation complete.")
        # Return only for assets that had data
        return expected_returns_annualized.dropna()

    except Exception as e:
         print(f"Error calculating expected returns: {e}")
         return None


def calculate_portfolio_variance(weights, cov_matrix):
    """Calculates the annualized portfolio variance."""
    return np.dot(weights.T, np.dot(cov_matrix, weights))

def calculate_portfolio_volatility(weights, cov_matrix):
    """Calculates the annualized portfolio volatility."""
    return np.sqrt(calculate_portfolio_variance(weights, cov_matrix))

def calculate_portfolio_return(weights, expected_returns):
    """Calculates the expected annualized portfolio return."""
    return np.sum(weights * expected_returns)

def neg_sharpe_ratio(weights, expected_returns, cov_matrix, risk_free_rate):
    """Calculates the negative Sharpe ratio (for minimization)."""
    portfolio_return = calculate_portfolio_return(weights, expected_returns)
    portfolio_volatility = calculate_portfolio_volatility(weights, cov_matrix)

    if portfolio_volatility == 0: # Avoid division by zero
        return -np.inf # Or some other large negative number if appropriate
    else:
        sharpe = (portfolio_return - risk_free_rate) / portfolio_volatility
        return -sharpe # Return negative Sharpe ratio
    
def optimize_portfolio(expected_returns, cov_matrix, constraints_df, risk_free_rate, target='min_volatility'):
    """
    Optimizes the portfolio for the given target (Min Volatility or Max Sharpe).

    Args:
        expected_returns (pd.Series): Annualized expected returns.
        cov_matrix (pd.DataFrame): Annualized covariance matrix.
        constraints_df (pd.DataFrame): DataFrame with 'Ticker', 'min_weight', 'max_weight'.
        risk_free_rate (float): Annual risk-free rate.
        target (str): 'min_volatility' or 'max_sharpe'.

    Returns:
        np.array: Optimized portfolio weights, or None if optimization fails.
    """
    num_assets = len(expected_returns)
    if num_assets == 0 or cov_matrix.empty:
         print("Error: Cannot optimize with no assets or no covariance matrix.")
         return None

    print(f"\nOptimizing portfolio for: {target}")

    # Initial guess (equal weight)
    init_guess = np.array(num_assets * [1. / num_assets])

    # Bounds (min/max constraints for each asset)
    # Ensure constraints match the assets available in expected_returns/cov_matrix
    constraints_aligned = constraints_df.set_index('Ticker').reindex(expected_returns.index)
    if constraints_aligned.isnull().any().any():
        print("Warning: Some assets in returns/covariance matrix are missing constraints. Using default bounds (0, 1).")
        # Find missing tickers and apply default bounds
        missing_tickers = constraints_aligned[constraints_aligned.isnull().any(axis=1)].index
        constraints_aligned.loc[missing_tickers, ['min_weight', 'max_weight']] = [0.0, 1.0] # Example default

    bounds = tuple((row['min_weight'], row['max_weight']) for index, row in constraints_aligned.iterrows())

    # Constraints: Weights sum to 1
    constraints = ({'type': 'eq', 'fun': lambda weights: np.sum(weights) - 1})

    # Define objective function based on target
    if target == 'min_volatility':
        objective_func = calculate_portfolio_variance # Minimize variance
        args = (cov_matrix,)
    elif target == 'max_sharpe':
        objective_func = neg_sharpe_ratio # Minimize negative Sharpe ratio
        args = (expected_returns, cov_matrix, risk_free_rate)
    else:
        print(f"Error: Unknown optimization target '{target}'")
        return None

    # Perform optimization
    try:
        result = minimize(objective_func,
                          init_guess,
                          args=args,
                          method='SLSQP', # As required
                          bounds=bounds,
                          constraints=constraints)

        if result.success:
            optimized_weights = result.x
            # Optional: Clean weights (set very small weights to zero, renormalize)
            optimized_weights[np.abs(optimized_weights) < 1e-6] = 0
            optimized_weights /= np.sum(optimized_weights) # Renormalize
            print(f"Optimization successful for {target}.")
            return optimized_weights
        else:
            print(f"Warning: Optimization failed for {target}. Reason: {result.message}")
            return None # Indicate failure

    except ValueError as ve:
         print(f"Error during optimization setup for {target} (check bounds/constraints): {ve}")
         return None
    except Exception as e:
        print(f"Error during optimization for {target}: {e}")
        return None
    
def calculate_equal_weights(tickers):
    """Calculates equal weights for the given tickers."""
    num_assets = len(tickers)
    if num_assets == 0:
        return None
    weights = np.array(num_assets * [1. / num_assets])
    print("\nCalculated equal weights.")
    return weights

def run_optimizations(weekly_returns, optimization_date, constraints_df, risk_free_rate):
    """
    Runs all portfolio optimization strategies.

    Args:
        weekly_returns (pd.DataFrame): Full DataFrame of weekly returns.
        optimization_date (pd.Timestamp): Date for calculating inputs.
        constraints_df (pd.DataFrame): Asset constraints.
        risk_free_rate (float): Annual risk-free rate.

    Returns:
        pd.DataFrame: DataFrame containing weights for Min Vol, Max Sharpe, and Equal Weight portfolios.
                      Returns None if critical errors occur.
    """
    # Calculate inputs for optimization
    cov_matrix = calculate_robust_covariance(weekly_returns, optimization_date)
    expected_returns = calculate_expected_returns(weekly_returns, optimization_date)

    if cov_matrix is None or expected_returns is None:
        print("Error: Cannot proceed with optimization due to missing covariance matrix or expected returns.")
        return None

    # Align assets - Ensure all inputs use the same set of assets
    # (those present in both cov_matrix and expected_returns)
    common_tickers = cov_matrix.index.intersection(expected_returns.index)
    if len(common_tickers) == 0:
         print("Error: No common assets between covariance matrix and expected returns.")
         return None
    if len(common_tickers) < len(expected_returns):
        print(f"Warning: Optimizing only for assets common to returns and covariance: {common_tickers.tolist()}")

    expected_returns = expected_returns.loc[common_tickers]
    cov_matrix = cov_matrix.loc[common_tickers, common_tickers]
    constraints_aligned_df = constraints_df[constraints_df['Ticker'].isin(common_tickers)].copy()


    # --- Run Optimizations ---
    min_vol_weights = optimize_portfolio(expected_returns, cov_matrix, constraints_aligned_df, risk_free_rate, target='min_volatility')
    max_sharpe_weights = optimize_portfolio(expected_returns, cov_matrix, constraints_aligned_df, risk_free_rate, target='max_sharpe')
    equal_weights = calculate_equal_weights(common_tickers.tolist())

    # --- Compile Results ---
    results = {}
    if min_vol_weights is not None:
        results['Min Volatility'] = pd.Series(min_vol_weights, index=common_tickers)
    else:
         print("Min Volatility optimization failed, results will be incomplete.")

    if max_sharpe_weights is not None:
        results['Max Sharpe'] = pd.Series(max_sharpe_weights, index=common_tickers)
    else:
         print("Max Sharpe optimization failed, results will be incomplete.")

    if equal_weights is not None:
        results['Equal Weight'] = pd.Series(equal_weights, index=common_tickers)

    if not results: # If all optimizations failed
         print("Error: All optimization strategies failed.")
         return None

    results_df = pd.DataFrame(results)
    # Fill NaN weights with 0 (e.g., if an optimization failed for some reason but others succeeded)
    # Although currently if one fails, we return None for its weights.
    results_df = results_df.fillna(0.0)

    print("\n--- Optimization Process Finished ---")
    return results_df

