# Portfolio Optimizer

## Description

This command-line tool performs portfolio optimization based on historical stock data. It downloads price data, calculates returns, runs various optimization strategies, analyzes portfolio performance, and generates reports including CSVs, Excel files, and PDF charts.

## Features

- **Data Acquisition:** Downloads historical stock prices from Yahoo Finance.
- **Portfolio Optimization:**
    - Minimum Volatility
    - Maximum Sharpe Ratio
    - Equal Weight
- **Performance Analysis:** Calculates key metrics such as:
    - Annualized Return
    - Annualized Volatility (Daily and Weekly)
    - Sharpe Ratio
    - Maximum Drawdown
    - Skewness & Kurtosis
    - Weighted Annualized Security Log Return
    - Asset Correlation Matrix
- **Reporting:** Generates:
    - CSV files for prices, daily returns, and weekly returns.
    - Excel files for optimized weights, security analysis returns, performance metrics, and correlation matrix.
    - PDF files for performance comparison charts and asset allocation pie charts.
- **Configuration:** Uses a `config.ini` file for flexible parameter settings.

## Input Excel File Format

The tool requires an input Excel file (e.g., `input_data.xlsx`) with the following structure:

- **Sheet3:**
    - Column A (from row 2): List of stock tickers.
    - Cell B2: Historical start date for data download (YYYY-MM-DD).
    - Cell C2: Historical end date for data download (YYYY-MM-DD).
    - Cell E2: Optimization date (YYYY-MM-DD) - data up to this date is used for calculating expected returns and covariance.
    - Cell G2: Analysis start date (YYYY-MM-DD).
    - Cell H2: Analysis end date (YYYY-MM-DD).
- **Sheet2:**
    - Column "Ticker": Stock tickers (must match Sheet3).
    - Column "Min Weight": Minimum allocation constraint for the ticker (e.g., 0.0).
    - Column "Max Weight": Maximum allocation constraint for the ticker (e.g., 1.0).

## Configuration (`config.ini`)

The `config.ini` file allows you to customize various parameters. It should be placed in the same directory as the executable or `run_portfolio_tool.py`.

```ini
[Files]
InputExcelFile = input_data.xlsx
OutputDirectory = output_results
PricesCSV = price_data.csv
DailyReturnsCSV = daily_returns.csv
WeeklyReturnsCSV = weekly_returns.csv
OptimizedWeightsExcel = optimized_portfolios.xlsx
PerformanceRiskExcel = performance_analysis.xlsx
AnalysisPDF = analysis_charts.pdf
PieChartsPDF = allocation_pie_charts.pdf

[Parameters]
RiskFreeRate = 0.01

[Settings]
PieChartWeightThreshold = 0.01
```

- **`InputExcelFile`**: Name of your input Excel file.
- **`OutputDirectory`**: Folder where all output files will be saved.
- **`RiskFreeRate`**: Annual risk-free rate for Sharpe Ratio calculation.
- **`PieChartWeightThreshold`**: Minimum weight for a security to be shown as a separate slice in pie charts; smaller weights are grouped into "Other".

## Output Files

The tool generates the following files in the specified `OutputDirectory`:

- **`price_data.csv`**: Historical adjusted closing prices.
- **`daily_returns.csv`**: Calculated daily percentage returns.
- **`weekly_returns.csv`**: Calculated weekly (Friday-to-Friday) percentage returns.
- **`optimized_portfolios.xlsx`**:
    - `Optimized_Weights` sheet: Weights for Min Volatility, Max Sharpe, and Equal Weight portfolios.
    - `Security_Analysis_Returns` sheet: Annualized total logarithmic return for each security.
- **`performance_analysis.xlsx`**:
    - `Performance_Metrics` sheet: Detailed performance and risk metrics for each strategy.
    - `Correlation_Matrix` sheet: Asset correlation matrix for the analysis period.
- **`analysis_charts.pdf`**: PDF containing:
    - Bar charts comparing annualized return and volatility of different strategies.
    - Heatmap of the asset correlation matrix.
- **`allocation_pie_charts.pdf`**: PDF containing pie charts showing asset allocation for each optimized portfolio.

## Setup and Dependencies

To run this tool, you need Python and the following libraries:

- pandas
- yfinance
- openpyxl
- matplotlib
- seaborn
- scikit-learn
- scipy

You can typically install these using pip:
```bash
pip install pandas yfinance openpyxl matplotlib seaborn scikit-learn scipy
```
It's recommended to use a virtual environment.

## Running the Tool

1.  Ensure Python and the required dependencies are installed.
2.  Prepare your `input_data.xlsx` file according to the specified format.
3.  Create or verify the `config.ini` file in the same directory.
4.  Run the script from your terminal:
    ```bash
    python run_portfolio_tool.py
    ```
5.  Check the `OutputDirectory` (e.g., `output_results`) for the generated files.

## Building the Executable (Optional)

The script uses `sys._MEIPASS` and `sys.frozen`, which suggests it's designed to be bundled into an executable using a tool like PyInstaller. If you want to create a standalone executable:

1.  Install PyInstaller:
    ```bash
    pip install pyinstaller
    ```
2.  Navigate to the script's directory and run:
    ```bash
    pyinstaller --onefile --add-data "config.ini:." run_portfolio_tool.py
    ```
    *(Note: The `--add-data` flag ensures `config.ini` is bundled. Adjust if your `config.ini` is located elsewhere relative to the script or if you have other data files.)*

    This will create a distributable executable in the `dist` folder.

## Disclaimer

This tool is for educational and informational purposes only. It is NOT financial advice. The results generated by this tool should not be taken as recommendations to buy or sell any financial instrument. Always do your own research and consult with a qualified financial advisor before making investment decisions.

## License

(To be added - currently no license file found. Consider adding an MIT License or similar.)
