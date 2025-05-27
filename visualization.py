# visualization.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages # For saving multi-page PDFs

def plot_performance_comparison(performance_df):
    """
    Generates bar charts comparing annualized return and volatility.

    Args:
        performance_df (pd.DataFrame): DataFrame with 'Annualized Return',
                                       'Annualized Volatility', etc., as rows
                                       and portfolio strategies as columns.

    Returns:
        matplotlib.figure.Figure: The figure object containing the plots, or None if error.
    """
    if performance_df is None or performance_df.empty or \
       'Annualized Return' not in performance_df.index or \
       'Annualized Volatility' not in performance_df.index:
        print("Error: Performance data is missing or incomplete for plotting.")
        return None

    try:
        # Ensure data is numeric and handle potential NaNs if necessary
        plot_data = performance_df.loc[['Annualized Return', 'Annualized Volatility']]. T # Transpose for plotting
        plot_data = plot_data.astype(float) # Ensure numeric type

        fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12, 5), sharey=False) # Create 2 subplots
        fig.suptitle('Portfolio Performance Comparison (Analysis Period)', fontsize=14)

        # Plot Annualized Return
        plot_data['Annualized Return'].plot(kind='bar', ax=axes[0], rot=0,
                                           title='Annualized Return',
                                           ylabel='Return (%)', # Assuming values are percentages or format them
                                           color=['skyblue', 'lightcoral', 'lightgreen']) # Example colors
        axes[0].yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: '{:.2%}'.format(y))) # Format y-axis as percentage

        # Plot Annualized Volatility
        plot_data['Annualized Volatility'].plot(kind='bar', ax=axes[1], rot=0,
                                              title='Annualized Volatility (Risk)',
                                              ylabel='Volatility (%)',
                                              color=['skyblue', 'lightcoral', 'lightgreen'])
        axes[1].yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: '{:.2%}'.format(y))) # Format y-axis as percentage

        plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust layout to prevent title overlap

        print("Generated performance comparison bar charts.")
        return fig

    except Exception as e:
        print(f"Error generating performance comparison plot: {e}")
        return None
    
def plot_correlation_heatmap(correlation_matrix):
    """
    Generates a heatmap for the asset correlation matrix.

    Args:
        correlation_matrix (pd.DataFrame): Asset correlation matrix.

    Returns:
        matplotlib.figure.Figure: The figure object containing the heatmap, or None if error.
    """
    if correlation_matrix is None or correlation_matrix.empty:
        print("Error: Correlation matrix is missing or empty for plotting.")
        return None

    try:
        fig, ax = plt.subplots(figsize=(10, 8)) # Adjust size as needed
        sns.heatmap(correlation_matrix, annot=False, cmap='coolwarm', # annot=True can be slow/cluttered for many assets
                    linewidths=.5, linecolor='black', ax=ax, fmt=".2f")
        ax.set_title('Asset Correlation Matrix (Analysis Period)', fontsize=14)
        plt.xticks(rotation=45, ha='right') # Rotate x-axis labels if needed
        plt.yticks(rotation=0)
        plt.tight_layout()

        print("Generated correlation matrix heatmap.")
        return fig

    except Exception as e:
        print(f"Error generating correlation heatmap: {e}")
        return None
    
def plot_allocation_pie_charts(optimized_weights_df, weight_threshold=0.01):
    """
    Generates pie charts for portfolio asset allocations.

    Args:
        optimized_weights_df (pd.DataFrame): DataFrame of optimized weights
                                              (Columns: Strategies, Index: Tickers).
        weight_threshold (float): Minimum weight to display as a separate slice.
                                  Smaller weights are grouped into 'Other'.

    Returns:
        matplotlib.figure.Figure: The figure object containing the pie charts, or None if error.
    """
    if optimized_weights_df is None or optimized_weights_df.empty:
        print("Error: Optimized weights data is missing for plotting pie charts.")
        return None

    try:
        num_portfolios = len(optimized_weights_df.columns)
        # Adjust layout based on number of portfolios
        ncols = min(num_portfolios, 3) # Max 3 columns
        nrows = (num_portfolios + ncols - 1) // ncols
        fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(ncols * 5, nrows * 5), squeeze=False) # Create subplots
        fig.suptitle('Portfolio Asset Allocation', fontsize=14)
        axes = axes.flatten() # Flatten to easily iterate

        for i, portfolio_name in enumerate(optimized_weights_df.columns):
            weights = optimized_weights_df[portfolio_name].copy()
            # Filter small weights and group into 'Other'
            significant_weights = weights[weights >= weight_threshold]
            other_weight = weights[weights < weight_threshold].sum()
            if other_weight > 1e-6: # Add 'Other' slice if it's non-negligible
                significant_weights['Other (<{:.0%})'.format(weight_threshold)] = other_weight

            # Create Pie Chart
            wedges, texts, autotexts = axes[i].pie(significant_weights,
                                                   labels=significant_weights.index,
                                                   autopct='%1.1f%%', # Format percentages
                                                   startangle=90,
                                                   pctdistance=0.85) # Distance of percentage labels from center
            axes[i].set_title(portfolio_name)
            # Optional: Improve label appearance
            plt.setp(autotexts, size=8, weight="bold", color="white")


        # Hide any unused subplots if num_portfolios doesn't fill the grid
        for j in range(i + 1, len(axes)):
             fig.delaxes(axes[j])


        plt.tight_layout(rect=[0, 0.03, 1, 0.93]) # Adjust layout


        print("Generated allocation pie charts.")
        return fig


    except Exception as e:
        print(f"Error generating allocation pie charts: {e}")
        return None
    
def save_plots_to_pdf(figures, output_pdf_path):
    """
    Saves a list of matplotlib figures to a single PDF file.

    Args:
        figures (list): A list of matplotlib Figure objects.
        output_pdf_path (str): The path to save the output PDF file.
    """
    valid_figures = [fig for fig in figures if fig is not None] # Filter out None values if plotting failed
    if not valid_figures:
         print(f"No valid figures to save to {output_pdf_path}.")
         return

    try:
        with PdfPages(output_pdf_path) as pdf:
            for fig in valid_figures:
                pdf.savefig(fig, bbox_inches='tight') # Use bbox_inches='tight' to prevent cutting off labels
                plt.close(fig) # Close the figure after saving to free memory
        print(f"Successfully saved plots to {output_pdf_path}")
    except Exception as e:
        print(f"Error saving plots to PDF {output_pdf_path}: {e}")

