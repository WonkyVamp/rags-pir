from typing import List, Dict, Any
import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

class PerformanceAnalyzer:
    def __init__(self, portfolio_data: pd.DataFrame):
        """
        Initialize performance analyzer
        
        Args:
            portfolio_data: DataFrame with columns ['date', 'portfolio_value', 'benchmark_value']
        """
        self.portfolio_data = portfolio_data
        self.returns = None
        self._calculate_returns()
        
    def _calculate_returns(self):
        """Calculate daily returns for portfolio and benchmark"""
        self.returns = pd.DataFrame()
        self.returns['portfolio'] = self.portfolio_data['portfolio_value'].pct_change()
        self.returns['benchmark'] = self.portfolio_data['benchmark_value'].pct_change()
        self.returns = self.returns.dropna()
        
    def calculate_metrics(self) -> Dict[str, float]:
        """Calculate key performance metrics"""
        if self.returns is None:
            self._calculate_returns()
            
        # Calculate basic metrics
        total_return = (self.portfolio_data['portfolio_value'].iloc[-1] / 
                       self.portfolio_data['portfolio_value'].iloc[0] - 1)
        
        # Calculate annualized metrics
        days = (self.portfolio_data['date'].iloc[-1] - self.portfolio_data['date'].iloc[0]).days
        years = days / 365.25
        
        annualized_return = (1 + total_return) ** (1 / years) - 1
        annualized_volatility = self.returns['portfolio'].std() * np.sqrt(252)
        
        # Calculate risk-adjusted metrics
        risk_free_rate = 0.02  # Assuming 2% risk-free rate
        excess_returns = self.returns['portfolio'] - risk_free_rate/252
        sharpe_ratio = np.sqrt(252) * excess_returns.mean() / excess_returns.std()
        
        # Calculate drawdown metrics
        portfolio_values = self.portfolio_data['portfolio_value']
        rolling_max = portfolio_values.expanding().max()
        drawdowns = (portfolio_values - rolling_max) / rolling_max
        max_drawdown = drawdowns.min()
        
        # Calculate information ratio
        tracking_error = (self.returns['portfolio'] - self.returns['benchmark']).std() * np.sqrt(252)
        information_ratio = (annualized_return - 
                           (self.returns['benchmark'].mean() * 252)) / tracking_error
        
        return {
            'total_return': total_return,
            'annualized_return': annualized_return,
            'annualized_volatility': annualized_volatility,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'information_ratio': information_ratio,
            'tracking_error': tracking_error
        }
    
    def analyze_risk_factors(self) -> Dict[str, Any]:
        """Analyze portfolio risk factors"""
        if self.returns is None:
            self._calculate_returns()
            
        # Calculate Value at Risk (VaR)
        var_95 = np.percentile(self.returns['portfolio'], 5)
        var_99 = np.percentile(self.returns['portfolio'], 1)
        
        # Calculate Expected Shortfall (ES)
        es_95 = self.returns['portfolio'][self.returns['portfolio'] <= var_95].mean()
        es_99 = self.returns['portfolio'][self.returns['portfolio'] <= var_99].mean()
        
        # Calculate downside deviation
        downside_returns = self.returns['portfolio'][self.returns['portfolio'] < 0]
        downside_deviation = np.sqrt(np.mean(downside_returns ** 2))
        
        # Calculate beta
        covariance = np.cov(self.returns['portfolio'], self.returns['benchmark'])[0,1]
        benchmark_variance = np.var(self.returns['benchmark'])
        beta = covariance / benchmark_variance
        
        return {
            'var_95': var_95,
            'var_99': var_99,
            'expected_shortfall_95': es_95,
            'expected_shortfall_99': es_99,
            'downside_deviation': downside_deviation,
            'beta': beta
        }
    
    def generate_performance_report(self) -> Dict[str, Any]:
        """Generate comprehensive performance report"""
        metrics = self.calculate_metrics()
        risk_factors = self.analyze_risk_factors()
        
        # Calculate rolling metrics
        rolling_returns = self.returns['portfolio'].rolling(window=252).mean() * 252
        rolling_vol = self.returns['portfolio'].rolling(window=252).std() * np.sqrt(252)
        
        # Calculate rolling Sharpe ratio
        rolling_sharpe = (rolling_returns - 0.02) / rolling_vol
        
        return {
            'performance_metrics': metrics,
            'risk_factors': risk_factors,
            'rolling_metrics': {
                'returns': rolling_returns.to_dict(),
                'volatility': rolling_vol.to_dict(),
                'sharpe_ratio': rolling_sharpe.to_dict()
            }
        }
    
    def plot_performance(self, save_path: str = None):
        """Plot performance charts"""
        if self.returns is None:
            self._calculate_returns()
            
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Plot cumulative returns
        cum_returns = (1 + self.returns).cumprod()
        axes[0,0].plot(cum_returns.index, cum_returns['portfolio'], label='Portfolio')
        axes[0,0].plot(cum_returns.index, cum_returns['benchmark'], label='Benchmark')
        axes[0,0].set_title('Cumulative Returns')
        axes[0,0].legend()
        
        # Plot rolling volatility
        rolling_vol = self.returns['portfolio'].rolling(window=252).std() * np.sqrt(252)
        axes[0,1].plot(rolling_vol.index, rolling_vol)
        axes[0,1].set_title('Rolling Volatility (1 Year)')
        
        # Plot drawdowns
        portfolio_values = self.portfolio_data['portfolio_value']
        rolling_max = portfolio_values.expanding().max()
        drawdowns = (portfolio_values - rolling_max) / rolling_max
        axes[1,0].fill_between(drawdowns.index, drawdowns, 0, color='red', alpha=0.3)
        axes[1,0].set_title('Drawdowns')
        
        # Plot rolling Sharpe ratio
        rolling_returns = self.returns['portfolio'].rolling(window=252).mean() * 252
        rolling_sharpe = (rolling_returns - 0.02) / rolling_vol
        axes[1,1].plot(rolling_sharpe.index, rolling_sharpe)
        axes[1,1].set_title('Rolling Sharpe Ratio (1 Year)')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
        else:
            plt.show() 