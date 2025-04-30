from typing import List, Dict, Any
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

class Backtester:
    def __init__(self, historical_data: pd.DataFrame):
        """
        Initialize backtester with historical data
        
        Args:
            historical_data: DataFrame with columns ['date', 'company', 'price', 'volume', 'sentiment', 'risk_score']
        """
        self.historical_data = historical_data
        self.results = None
        
    def run_backtest(self, 
                    strategy: Dict[str, Any],
                    start_date: datetime,
                    end_date: datetime,
                    initial_capital: float = 100000.0) -> Dict[str, Any]:
        """
        Run backtest for a given strategy
        
        Args:
            strategy: Dictionary containing strategy parameters
            start_date: Start date for backtesting
            end_date: End date for backtesting
            initial_capital: Initial capital for simulation
        """
        # Filter data for backtest period
        mask = (self.historical_data['date'] >= start_date) & (self.historical_data['date'] <= end_date)
        test_data = self.historical_data[mask].copy()
        
        # Initialize results
        portfolio_value = initial_capital
        positions = {}
        trades = []
        
        # Run simulation
        for date, group in test_data.groupby('date'):
            # Get predictions for current date
            predictions = self._get_predictions(group, strategy)
            
            # Execute trades
            for company, pred in predictions.items():
                if pred['action'] == 'buy' and company not in positions:
                    # Calculate position size
                    position_size = portfolio_value * strategy['position_size']
                    price = group[group['company'] == company]['price'].iloc[0]
                    shares = position_size / price
                    
                    positions[company] = {
                        'shares': shares,
                        'entry_price': price,
                        'entry_date': date
                    }
                    
                    trades.append({
                        'date': date,
                        'company': company,
                        'action': 'buy',
                        'shares': shares,
                        'price': price,
                        'value': position_size
                    })
                    
                elif pred['action'] == 'sell' and company in positions:
                    price = group[group['company'] == company]['price'].iloc[0]
                    position = positions[company]
                    value = position['shares'] * price
                    
                    trades.append({
                        'date': date,
                        'company': company,
                        'action': 'sell',
                        'shares': position['shares'],
                        'price': price,
                        'value': value
                    })
                    
                    portfolio_value += value
                    del positions[company]
        
        # Calculate final portfolio value
        for company, position in positions.items():
            final_price = test_data[test_data['company'] == company]['price'].iloc[-1]
            portfolio_value += position['shares'] * final_price
        
        # Calculate performance metrics
        performance = self._calculate_performance_metrics(trades, initial_capital, portfolio_value)
        
        self.results = {
            'trades': trades,
            'final_portfolio_value': portfolio_value,
            'performance_metrics': performance
        }
        
        return self.results
    
    def _get_predictions(self, data: pd.DataFrame, strategy: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
        """Generate predictions based on strategy rules"""
        predictions = {}
        
        for _, row in data.iterrows():
            company = row['company']
            sentiment = row['sentiment']
            risk_score = row['risk_score']
            
            # Apply strategy rules
            if (sentiment > strategy['sentiment_threshold'] and 
                risk_score < strategy['risk_threshold']):
                predictions[company] = {'action': 'buy', 'confidence': sentiment}
            elif (sentiment < -strategy['sentiment_threshold'] or 
                  risk_score > strategy['risk_threshold']):
                predictions[company] = {'action': 'sell', 'confidence': abs(sentiment)}
                
        return predictions
    
    def _calculate_performance_metrics(self, 
                                    trades: List[Dict[str, Any]],
                                    initial_capital: float,
                                    final_value: float) -> Dict[str, float]:
        """Calculate performance metrics"""
        if not trades:
            return {
                'total_return': 0.0,
                'sharpe_ratio': 0.0,
                'max_drawdown': 0.0,
                'win_rate': 0.0
            }
            
        # Calculate returns
        returns = []
        portfolio_values = [initial_capital]
        current_value = initial_capital
        
        for trade in trades:
            if trade['action'] == 'buy':
                current_value -= trade['value']
            else:
                current_value += trade['value']
            portfolio_values.append(current_value)
            
            if len(portfolio_values) > 1:
                returns.append((portfolio_values[-1] - portfolio_values[-2]) / portfolio_values[-2])
        
        returns = np.array(returns)
        
        # Calculate metrics
        total_return = (final_value - initial_capital) / initial_capital
        sharpe_ratio = np.mean(returns) / np.std(returns) if len(returns) > 0 else 0
        
        # Calculate maximum drawdown
        peak = portfolio_values[0]
        max_drawdown = 0
        for value in portfolio_values:
            if value > peak:
                peak = value
            drawdown = (peak - value) / peak
            max_drawdown = max(max_drawdown, drawdown)
        
        # Calculate win rate
        profitable_trades = sum(1 for trade in trades if trade['action'] == 'sell' and 
                              trade['value'] > trade['shares'] * trade['price'])
        total_trades = sum(1 for trade in trades if trade['action'] == 'sell')
        win_rate = profitable_trades / total_trades if total_trades > 0 else 0
        
        return {
            'total_return': total_return,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'win_rate': win_rate
        }
    
    def get_strategy_analysis(self) -> Dict[str, Any]:
        """Get detailed analysis of strategy performance"""
        if self.results is None:
            raise ValueError("No backtest results available. Run backtest first.")
            
        trades_df = pd.DataFrame(self.results['trades'])
        
        # Calculate trade statistics
        trade_stats = {
            'total_trades': len(trades_df),
            'buy_trades': len(trades_df[trades_df['action'] == 'buy']),
            'sell_trades': len(trades_df[trades_df['action'] == 'sell']),
            'avg_trade_value': trades_df['value'].mean(),
            'max_trade_value': trades_df['value'].max(),
            'min_trade_value': trades_df['value'].min()
        }
        
        # Calculate company-specific statistics
        company_stats = {}
        for company in trades_df['company'].unique():
            company_trades = trades_df[trades_df['company'] == company]
            company_stats[company] = {
                'total_trades': len(company_trades),
                'total_value': company_trades['value'].sum(),
                'avg_trade_value': company_trades['value'].mean()
            }
        
        return {
            'trade_statistics': trade_stats,
            'company_statistics': company_stats,
            'performance_metrics': self.results['performance_metrics']
        } 