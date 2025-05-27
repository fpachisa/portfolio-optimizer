# run_portfolio_tool.py
import sys
import os
import configparser # For reading config file
import pandas as pd
import matplotlib # To ensure backend is usable if run non-interactively

# Import your custom modules
import data_acquisition as da
import portfolio_optimization as po
import performance_analysis as pa
import visualization as viz

# Use a non-interactive backend suitable for saving files without displaying
try:
    matplotlib.use('Agg') # 'Agg' is good for writing to file without GUI
    import matplotlib.pyplot as plt
except ImportError:
    print("Warning: Matplotlib could not be imported or backend could not be set.")
    plt = None

# --- Helper Functions for Path Handling ---

def get_base_path():
    """ Get the base path for BUNDLED data files (like config.ini in MEIPASS) """
    if getattr(sys, 'frozen', False):
        base_path = sys._MEIPASS
    else:
        base_path = os.path.dirname(os.path.abspath(__file__))
    return base_path

def get_app_directory():
    """ Get the directory of the running executable or script itself """
    if getattr(sys, 'frozen', False):
        return os.path.dirname(sys.executable)
    else:
        return os.path.dirname(os.path.abspath(__file__))

# --- Configuration Loading (UPDATED) ---

def load_configuration(config_file='config.ini'):
    """
    Loads configuration from INI file.
    Priority:
    1. Looks for config_file in the application directory (next to exe).
    2. If not found, looks for config_file in the bundled base path (MEIPASS).
    """
    config = configparser.ConfigParser()
    config_loaded_from = None

    # Path 1: Application directory (next to executable)
    app_dir = get_app_directory()
    config_in_app_dir = os.path.join(app_dir, config_file)

    # Path 2: Bundled base path (MEIPASS)
    base_path = get_base_path()
    config_in_base_path = os.path.join(base_path, config_file)

    if os.path.exists(config_in_app_dir):
        print(f"Found configuration file in application directory: '{config_in_app_dir}'")
        config.read(config_in_app_dir)
        config_loaded_from = config_in_app_dir
    elif os.path.exists(config_in_base_path):
        print(f"Configuration file not found in app directory. Using bundled version: '{config_in_base_path}'")
        config.read(config_in_base_path)
        config_loaded_from = config_in_base_path
    else:
        # Error if not found in either location
        raise FileNotFoundError(
            f"Configuration file '{config_file}' not found.\n"
            f"Checked application directory: '{config_in_app_dir}'\n"
            f"Checked bundled path: '{config_in_base_path}'\n"
            f"Please ensure 'config.ini' exists in one of these locations or was bundled correctly."
        )

    # Basic validation (check if sections exist)
    required_sections = ['Files', 'Parameters', 'Settings']
    for section in required_sections:
        if section not in config:
            raise ValueError(f"Missing required section '[{section}]' in loaded config file: '{config_loaded_from}'.")

    print(f"Configuration loaded successfully.")
    return config

def get_config_value(config, section, key, is_float=False, is_int=False):
    """Helper function to get config value with error checking."""
    # (This function remains the same as before)
    try:
        value = config.get(section, key)
        if not value:
             raise ValueError(f"Configuration value for '{key}' in section '[{section}]' is empty.")
        if is_float:
            return float(value)
        if is_int:
            return int(value)
        return value
    except (configparser.NoOptionError, configparser.NoSectionError):
        raise ValueError(f"Missing configuration value: section='[{section}]', key='{key}'")
    except ValueError as e:
         raise ValueError(f"Invalid format or empty value for configuration: section='[{section}]', key='{key}'. Error: {e}")


# --- Main Pipeline Function ---

def run_full_pipeline(config):
    """Orchestrates the entire portfolio optimization pipeline."""
    # (This function remains the same as the previous complete version)
    # It will now use the config object loaded by the updated load_configuration
    print("--- Starting Portfolio Optimization Pipeline ---")

    # Initialize results to None
    daily_returns, weekly_returns = None, None
    optimized_weights_df = None
    performance_df = None
    correlation_matrix = None

    # --- Get Configuration Values ---
    try:
        print("Reading configuration settings...")
        app_dir = get_app_directory()
        input_excel_filename = get_config_value(config, 'Files', 'InputExcelFile')
        input_excel_file_path = os.path.join(app_dir, input_excel_filename)
        output_dir_name = get_config_value(config, 'Files', 'OutputDirectory')
        output_dir = os.path.join(app_dir, output_dir_name)

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            print(f"Created output directory: '{output_dir}'")

        csv_prices_path = os.path.join(output_dir, get_config_value(config, 'Files', 'PricesCSV'))
        csv_daily_returns_path = os.path.join(output_dir, get_config_value(config, 'Files', 'DailyReturnsCSV'))
        csv_weekly_returns_path = os.path.join(output_dir, get_config_value(config, 'Files', 'WeeklyReturnsCSV'))
        excel_optimized_weights_path = os.path.join(output_dir, get_config_value(config, 'Files', 'OptimizedWeightsExcel'))
        excel_performance_risk_path = os.path.join(output_dir, get_config_value(config, 'Files', 'PerformanceRiskExcel'))
        pdf_analysis_charts_path = os.path.join(output_dir, get_config_value(config, 'Files', 'AnalysisPDF'))
        pdf_pie_charts_path = os.path.join(output_dir, get_config_value(config, 'Files', 'PieChartsPDF'))

        risk_free_rate = get_config_value(config, 'Parameters', 'RiskFreeRate', is_float=True)
        pie_chart_threshold = get_config_value(config, 'Settings', 'PieChartWeightThreshold', is_float=True)
        print("Configuration settings processed.")

    except (FileNotFoundError, ValueError) as e:
         print(f"*** Configuration Error ***")
         print(e)
         print("Pipeline cannot continue due to configuration issues.")
         return # Stop execution

    # === Step 1 & 2: Data Acquisition ===
    try:
        print("\n=== Running Data Acquisition ===")
        print(f"Attempting to read input file: {input_excel_file_path}")
        if not os.path.exists(input_excel_file_path):
             raise FileNotFoundError(f"Input Excel file '{input_excel_filename}' not found in the application directory: '{app_dir}'. Please place it next to the executable/script.")
        tickers, start_date, end_date, optimization_date, analysis_start_date, analysis_end_date, constraints_df = da.read_input_data(input_excel_file_path)
        if tickers is None: raise ValueError("Failed to read valid data from input Excel file (check format/content/sheet names).")
        print(f"Successfully read {len(tickers)} tickers and date ranges.")
        print(f"Downloading price data for {len(tickers)} tickers...")
        price_data = da.download_price_data(tickers, start_date, end_date)
        if price_data is None: raise ValueError("Failed to download price data (check internet connection and ticker validity).")
        print(f"Successfully downloaded price data for {price_data.shape[1]} tickers.")
        print("Calculating returns...")
        daily_returns, weekly_returns = da.calculate_returns(price_data)
        if daily_returns is None or weekly_returns is None: raise ValueError("Failed to calculate returns.")
        print("Returns calculated.")
        print(f"Saving price data to: {csv_prices_path}")
        price_data.to_csv(csv_prices_path)
        print(f"Saving daily returns to: {csv_daily_returns_path}")
        daily_returns.to_csv(csv_daily_returns_path)
        print(f"Saving weekly returns to: {csv_weekly_returns_path}")
        weekly_returns.to_csv(csv_weekly_returns_path)
        print(f"Saved intermediate CSV files to '{output_dir}'")
        print("=== Data Acquisition Finished ===")

        if daily_returns is not None and analysis_start_date is not None and analysis_end_date is not None:
            security_analysis_returns_df = pa.calculate_security_analysis_returns( # This line already exists
                daily_returns,
                analysis_start_date,
                analysis_end_date
            )
            # Ensure the column name is what we expect, or adjust if it was changed by user manually
            if security_analysis_returns_df is not None and 'Annualized_Total_Log_Return' not in security_analysis_returns_df.columns:
                 if 'Annualized_Mean_Arithmetic_Return' in security_analysis_returns_df.columns: # If user tried previous fix
                     security_analysis_returns_df = security_analysis_returns_df.rename(columns={'Annualized_Mean_Arithmetic_Return': 'Annualized_Total_Log_Return'})
                     print("Info: Renamed 'Annualized_Mean_Arithmetic_Return' to 'Annualized_Total_Log_Return' in security_analysis_returns_df for consistency with current request.")
                 # Add more checks if other names are possible for the log return column.
        else:
             print("Warning: Could not calculate security analysis returns due to missing daily returns or analysis dates.")

    except Exception as e:
        print(f"*** Error during Data Acquisition ***"); print(e)
        print("Pipeline stopped due to error in Data Acquisition."); return


    # === Step 3: Portfolio Optimization ===
    try:
        print("\n=== Running Portfolio Optimization ===")
        if weekly_returns is None or optimization_date is None or constraints_df is None: raise ValueError("Prerequisite data for optimization is missing.")
        optimized_weights_df = po.run_optimizations(weekly_returns, optimization_date, constraints_df, risk_free_rate)
        if optimized_weights_df is None: raise ValueError("Portfolio optimization process failed (check constraints, data quality, or optimization logs).")
        print("\nFinal Optimized Weights:"); print(optimized_weights_df.round(4))
        print(f"Saving optimized weights and security analysis returns to: {excel_optimized_weights_path}")
        try: # Add a try-except block for robustness during file writing
             with pd.ExcelWriter(excel_optimized_weights_path, engine='openpyxl') as writer:
                 if optimized_weights_df is not None:
                     optimized_weights_df.to_excel(writer, sheet_name='Optimized_Weights')
                 else:
                      print("Warning: Optimized weights data is missing, not writing this sheet.")

                 if security_analysis_returns_df is not None:
                     security_analysis_returns_df.to_excel(writer, sheet_name='Security_Analysis_Returns', index=False) # index=False for cleaner output
                 else:
                      print("Warning: Security analysis returns data is missing, not writing this sheet.")
             print(f"Saved data successfully to '{excel_optimized_weights_path}'.")
        except Exception as write_error:
             print(f"*** Error writing to Excel file '{excel_optimized_weights_path}' ***: {write_error}")
        print("=== Portfolio Optimization Finished ===")
    except Exception as e:
        print(f"*** Error during Portfolio Optimization ***"); print(e)
        print("Pipeline stopped due to error in Portfolio Optimization."); return

    # === Step 4: Performance Analysis ===
    try:
        print("\n=== Running Performance Analysis ===")
        if daily_returns is None or optimized_weights_df is None or analysis_start_date is None or analysis_end_date is None: raise ValueError("Prerequisite data for performance analysis is missing.")
        print("Calculating portfolio performance metrics...")
        performance_df = pa.run_performance_analysis(daily_returns, weekly_returns, optimized_weights_df,
                                                     analysis_start_date, analysis_end_date, risk_free_rate,
                                                     security_analysis_returns_df=security_analysis_returns_df)
        print("Calculating asset correlation matrix for analysis period...")
        relevant_tickers = optimized_weights_df.index.tolist()
        correlation_matrix = pa.calculate_correlation_matrix(daily_returns, analysis_start_date, analysis_end_date, relevant_tickers)
        if performance_df is not None: print("\nCalculated Performance Metrics:"); print(performance_df.round(4))
        else: print("Warning: Performance analysis metrics calculation failed or produced no results.")
        if correlation_matrix is not None: print(f"\nCalculated Correlation Matrix (Analysis Period) Shape: {correlation_matrix.shape}")
        else: print("Warning: Correlation matrix calculation failed or produced no results.")
        if performance_df is not None or correlation_matrix is not None:
            print(f"Saving performance metrics and correlation matrix to: {excel_performance_risk_path}")
            with pd.ExcelWriter(excel_performance_risk_path, engine='openpyxl') as writer:
                if performance_df is not None: performance_df.to_excel(writer, sheet_name='Performance_Metrics')
                if correlation_matrix is not None: correlation_matrix.to_excel(writer, sheet_name='Correlation_Matrix')
            print(f"Saved performance/risk data successfully.")
        else: print("\nSkipping saving performance/risk Excel file as no relevant data was generated.")
        print("=== Performance Analysis Finished ===")
    except Exception as e:
        print(f"*** Error during Performance Analysis ***"); print(e)
        print("Attempting to continue to Visualization despite potential errors in Performance Analysis...")

    # === Step 5: Visualization ===
    try:
        print("\n=== Running Visualization ===")
        if plt is None: print("Skipping visualization because Matplotlib could not be initialized.")
        elif performance_df is None and correlation_matrix is None and optimized_weights_df is None: print("Skipping visualization as no data is available.")
        else:
            analysis_figures = []
            if performance_df is not None:
                 print("Generating performance comparison plots...")
                 fig_perf = viz.plot_performance_comparison(performance_df);
                 if fig_perf: analysis_figures.append(fig_perf)
                 else: print("Warning: Failed to generate performance comparison plot.")
            if correlation_matrix is not None:
                 print("Generating correlation heatmap...")
                 fig_corr = viz.plot_correlation_heatmap(correlation_matrix);
                 if fig_corr: analysis_figures.append(fig_corr)
                 else: print("Warning: Failed to generate correlation heatmap.")
            if analysis_figures: print(f"Saving analysis plots to: {pdf_analysis_charts_path}"); viz.save_plots_to_pdf(analysis_figures, pdf_analysis_charts_path)
            else: print("No analysis plots were generated to save.")
            pie_figures = []
            if optimized_weights_df is not None:
                 print("Generating allocation pie charts...")
                 fig_pie = viz.plot_allocation_pie_charts(optimized_weights_df, weight_threshold=pie_chart_threshold)
                 if fig_pie: pie_figures.append(fig_pie)
                 else: print("Warning: Failed to generate allocation pie charts.")
            if pie_figures: print(f"Saving allocation pie charts to: {pdf_pie_charts_path}"); viz.save_plots_to_pdf(pie_figures, pdf_pie_charts_path)
            else: print("No allocation pie charts were generated to save.")
            print("=== Visualization Finished ===")
    except Exception as e:
        print(f"*** Error during Visualization ***"); print(e)

    print("\n--- Portfolio Optimization Pipeline Finished ---")
    print(f"Please check the '{output_dir}' folder for all generated CSV, Excel, and PDF files.")


# --- Main Execution Block ---
if __name__ == "__main__":
    print("=============================================")
    print("= Starting Portfolio Optimization Tool =")
    print("=============================================")
    try:
        config = load_configuration() # Looks for config.ini by default
        run_full_pipeline(config)
    except (FileNotFoundError, ValueError, configparser.Error) as cfg_error:
         print(f"*** CRITICAL CONFIGURATION ERROR ***"); print(cfg_error)
         print("Pipeline cannot start. Please check your 'config.ini' file.")
    except Exception as general_error:
         print(f"*** AN UNEXPECTED ERROR OCCURRED ***"); print(general_error)
         import traceback; traceback.print_exc()
         print("Pipeline execution stopped prematurely.")

    print("\n=============================================")
    print("= Pipeline Execution Complete              =")
    print("=============================================")

    if getattr(sys, 'frozen', False): # Only pause if running as frozen executable
         input("\nPress Enter to exit...")