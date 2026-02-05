"""
visualization module for dengue forecasting results.
handles plotting of predictions, comparisons, and trends.
"""

import os
import matplotlib.pyplot as plt
import pandas as pd
from typing import Optional


class Visualizer:
    """
    visualizer class following single responsibility principle.
    responsible only for creating visualizations.
    """
    
    def __init__(self, figsize: tuple = (12, 5), show_plots: bool = False):
        """
        initialize visualizer with default figure size.
        
        args:
            figsize: default figure size for plots
            show_plots: whether to display plots (False for automated training)
        """
        self.figsize = figsize
        self.show_plots = show_plots
        self.output_dir = os.path.join("outputs", "plots")
    
    def _save_fig(self, filename: str) -> None:
        os.makedirs(self.output_dir, exist_ok=True)
        path = os.path.join(self.output_dir, filename)
        plt.savefig(path, dpi=150, bbox_inches="tight")
    
    def add_quarter_date(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        convert year_quarter to datetime for plotting.
        
        args:
            df: dataframe with year_quarter column
            
        returns:
            dataframe with added date column
        """
        df_copy = df.copy()
        df_copy['date'] = pd.PeriodIndex(
            df_copy['year_quarter'], freq='Q'
        ).to_timestamp()
        return df_copy
    
    def plot_actual_vs_predicted(
        self,
        test_df: pd.DataFrame,
        predictions: pd.Series,
        model_name: str,
        year: int,
        historical_df: Optional[pd.DataFrame] = None,
        hist_years: int = 2
    ) -> None:
        """
        plot actual vs predicted values for a test year.
        
        args:
            test_df: test dataframe with actual values
            predictions: predicted values
            model_name: name of the model
            year: test year
            historical_df: optional full historical dataframe
            hist_years: number of historical years to show
        """
        # prepare test data
        plot_df = self.add_quarter_date(test_df.copy())
        plot_df['predicted'] = predictions
        
        fig, ax = plt.subplots(figsize=self.figsize)
        
        # plot historical context if provided
        if historical_df is not None:
            hist_df = self.add_quarter_date(historical_df.copy())
            train_hist = hist_df[
                (hist_df['year'] <= year - 1) & 
                (hist_df['year'] >= year - hist_years)
            ]
            ax.plot(
                train_hist['date'],
                train_hist['casos_est'],
                'b-',
                linewidth=2,
                label='Historical (train)'
            )
        
        # plot actual and predicted
        ax.plot(
            plot_df['date'],
            plot_df['casos_est'],
            'g-o',
            linewidth=2,
            markersize=7,
            label=f'Actual ({year})'
        )
        ax.plot(
            plot_df['date'],
            plot_df['predicted'],
            'r--s',
            linewidth=2,
            markersize=7,
            label=f'Predicted ({year})'
        )
        
        # add forecast start line
        ax.axvline(
            x=plot_df['date'].min(),
            color='gray',
            linestyle=':',
            alpha=0.8,
            label='Forecast start'
        )
        
        # formatting
        ax.set_title(
            f'{year}: Actual vs Predicted ({model_name})',
            fontsize=14,
            fontweight='bold'
        )
        ax.set_xlabel('Quarter')
        ax.set_ylabel('Quarterly Cases')
        ax.legend()
        ax.grid(alpha=0.3)
        plt.xticks(rotation=45)
        plt.tight_layout()
        self._save_fig(f"actual_vs_predicted_{year}_{model_name}.png")
        if self.show_plots:
            plt.show()
        else:
            plt.close()
    
    def plot_forecast(
        self,
        forecast_df: pd.DataFrame,
        historical_df: pd.DataFrame,
        forecast_year: int,
        model_name: str,
        hist_start_year: int = None
    ) -> None:
        """
        plot future forecast with historical context.
        
        args:
            forecast_df: dataframe with forecast values
            historical_df: full historical dataframe
            forecast_year: year being forecast
            model_name: name of the model used
            hist_start_year: starting year for historical plot
        """
        # prepare data
        forecast_plot = self.add_quarter_date(forecast_df.copy())
        hist_plot = self.add_quarter_date(historical_df.copy())
        
        # filter historical data
        if hist_start_year:
            hist = hist_plot[hist_plot['year'] >= hist_start_year].copy()
        else:
            hist = hist_plot.copy()
        
        # get last historical point for connection
        last_hist_date = hist['date'].iloc[-1]
        last_hist_value = hist['casos_est'].iloc[-1]
        
        # create connected forecast
        forecast_connected = pd.concat([
            pd.DataFrame({
                'date': [last_hist_date],
                'predicted_casos_est': [last_hist_value]
            }),
            forecast_plot[['date', 'predicted_casos_est']]
        ], ignore_index=True)
        
        # create plot
        fig, ax = plt.subplots(figsize=self.figsize)
        
        # plot historical
        ax.plot(
            hist['date'],
            hist['casos_est'],
            color='blue',
            linewidth=2.5,
            label='Historical'
        )
        
        # plot forecast
        ax.plot(
            forecast_connected['date'],
            forecast_connected['predicted_casos_est'],
            'r--o',
            linewidth=2.5,
            markersize=8,
            label=f'Forecast {forecast_year}'
        )
        
        # add forecast start line
        ax.axvline(
            x=forecast_plot['date'].min(),
            color='black',
            linestyle=':',
            linewidth=1.5,
            label='Forecast start'
        )
        
        # formatting
        ax.set_title(
            f'{forecast_year} Forecast ({model_name})',
            fontsize=15,
            fontweight='bold'
        )
        ax.set_xlabel('Quarter')
        ax.set_ylabel('Quarterly Cases')
        ax.grid(alpha=0.3)
        ax.legend()
        plt.xticks(rotation=45)
        plt.tight_layout()
        self._save_fig(f"forecast_{forecast_year}_{model_name}.png")
        if self.show_plots:
            plt.show()
        else:
            plt.close()
    
    def plot_feature_importance(
        self,
        importance_df: pd.DataFrame,
        title: str = "Feature Importance"
    ) -> None:
        """
        plot feature importance as horizontal bar chart.
        
        args:
            importance_df: dataframe with Feature and Importance columns
            title: plot title
        """
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # create horizontal bar chart
        ax.barh(
            importance_df['Feature'],
            importance_df['Importance'],
            color='steelblue'
        )
        
        # formatting
        ax.set_xlabel('Importance')
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.invert_yaxis()  # highest importance on top
        ax.grid(alpha=0.3, axis='x')
        plt.tight_layout()
        safe_title = "".join(
            c if c.isalnum() or c in ("-", "_") else "_"
            for c in title.strip().lower()
        )
        self._save_fig(f"feature_importance_{safe_title}.png")
        if self.show_plots:
            plt.show()
        else:
            plt.close()
    
    def print_feature_importance(
        self,
        importance_df: pd.DataFrame,
        title: str = "Top 10 Feature Importance"
    ) -> None:
        """
        print feature importance as text visualization.
        
        args:
            importance_df: dataframe with Feature and Importance columns
            title: section title
        """
        print("\n" + "=" * 80)
        print(f"[SEARCH] {title}")
        print("=" * 80)
        
        for _, row in importance_df.iterrows():
            bar = '' * int(row['Importance'] * 50)
            print(
                f"   {row['Feature']:<25} "
                f"{row['Importance']:.4f} {bar}"
            )
    