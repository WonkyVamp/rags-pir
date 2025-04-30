from typing import List, Dict, Any, Callable
import pandas as pd
import numpy as np
from scipy.optimize import minimize
from datetime import datetime, timedelta
from concurrent.futures import ProcessPoolExecutor
from .backtesting import Backtester

class StrategyOptimizer:
    def __init__(self, 
                 historical_data: pd.DataFrame,
                 backtester: Backtester,
                 param_bounds: Dict[str, tuple]):
        """
        Initialize strategy optimizer
        
        Args:
            historical_data: DataFrame with historical data
            backtester: Backtester instance
            param_bounds: Dictionary of parameter bounds for optimization
        """
        self.historical_data = historical_data
        self.backtester = backtester
        self.param_bounds = param_bounds
        self.best_params = None
        self.best_score = float('-inf')
        
    def optimize(self, 
                objective_func: Callable,
                n_trials: int = 100,
                n_jobs: int = -1) -> Dict[str, Any]:
        """
        Optimize strategy parameters
        
        Args:
            objective_func: Function to optimize (e.g., Sharpe ratio, returns)
            n_trials: Number of optimization trials
            n_jobs: Number of parallel jobs (-1 for all cores)
        """
        # Generate random parameter combinations
        param_combinations = self._generate_param_combinations(n_trials)
        
        # Run optimization in parallel
        with ProcessPoolExecutor(max_workers=n_jobs) as executor:
            results = list(executor.map(
                lambda params: self._evaluate_params(params, objective_func),
                param_combinations
            ))
        
        # Find best parameters
        best_idx = np.argmax([r['score'] for r in results])
        self.best_params = results[best_idx]['params']
        self.best_score = results[best_idx]['score']
        
        return {
            'best_params': self.best_params,
            'best_score': self.best_score,
            'all_results': results
        }
    
    def _generate_param_combinations(self, n_trials: int) -> List[Dict[str, float]]:
        """Generate random parameter combinations within bounds"""
        param_combinations = []
        
        for _ in range(n_trials):
            params = {}
            for param, (lower, upper) in self.param_bounds.items():
                params[param] = np.random.uniform(lower, upper)
            param_combinations.append(params)
            
        return param_combinations
    
    def _evaluate_params(self, 
                        params: Dict[str, float],
                        objective_func: Callable) -> Dict[str, Any]:
        """Evaluate a set of parameters"""
        try:
            # Run backtest with parameters
            results = self.backtester.run_backtest(
                strategy=params,
                start_date=self.historical_data['date'].min(),
                end_date=self.historical_data['date'].max()
            )
            
            # Calculate objective score
            score = objective_func(results)
            
            return {
                'params': params,
                'score': score,
                'results': results
            }
        except Exception as e:
            return {
                'params': params,
                'score': float('-inf'),
                'error': str(e)
            }
    
    def optimize_with_constraints(self,
                                objective_func: Callable,
                                constraints: List[Dict[str, Any]],
                                initial_params: Dict[str, float] = None) -> Dict[str, Any]:
        """
        Optimize strategy parameters with constraints using scipy.optimize
        
        Args:
            objective_func: Function to optimize
            constraints: List of constraint dictionaries
            initial_params: Initial parameter values
        """
        if initial_params is None:
            # Generate random initial parameters
            initial_params = {
                param: np.random.uniform(bounds[0], bounds[1])
                for param, bounds in self.param_bounds.items()
            }
        
        # Convert parameters to array for optimization
        param_names = list(self.param_bounds.keys())
        initial_values = [initial_params[name] for name in param_names]
        bounds = [self.param_bounds[name] for name in param_names]
        
        # Define objective function for scipy
        def objective(x):
            params = dict(zip(param_names, x))
            results = self._evaluate_params(params, objective_func)
            return -results['score']  # Negative because we want to maximize
        
        # Run optimization
        result = minimize(
            objective,
            initial_values,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints
        )
        
        # Convert results back to dictionary
        optimized_params = dict(zip(param_names, result.x))
        
        return {
            'optimized_params': optimized_params,
            'optimization_success': result.success,
            'final_score': -result.fun,  # Convert back to positive
            'message': result.message
        }
    
    def analyze_parameter_sensitivity(self,
                                    param_name: str,
                                    n_points: int = 10) -> Dict[str, Any]:
        """
        Analyze sensitivity of strategy to a specific parameter
        
        Args:
            param_name: Name of parameter to analyze
            n_points: Number of points to evaluate
        """
        if self.best_params is None:
            raise ValueError("Run optimization first to get best parameters")
            
        # Generate parameter values
        param_range = np.linspace(
            self.param_bounds[param_name][0],
            self.param_bounds[param_name][1],
            n_points
        )
        
        # Evaluate each parameter value
        results = []
        for value in param_range:
            params = self.best_params.copy()
            params[param_name] = value
            
            # Run backtest
            backtest_results = self.backtester.run_backtest(
                strategy=params,
                start_date=self.historical_data['date'].min(),
                end_date=self.historical_data['date'].max()
            )
            
            results.append({
                'parameter_value': value,
                'performance': backtest_results['performance_metrics']
            })
            
        return {
            'parameter': param_name,
            'sensitivity_analysis': results
        }
    
    def generate_optimization_report(self) -> Dict[str, Any]:
        """Generate comprehensive optimization report"""
        if self.best_params is None:
            raise ValueError("Run optimization first to get best parameters")
            
        # Run final backtest with best parameters
        final_results = self.backtester.run_backtest(
            strategy=self.best_params,
            start_date=self.historical_data['date'].min(),
            end_date=self.historical_data['date'].max()
        )
        
        # Get strategy analysis
        strategy_analysis = self.backtester.get_strategy_analysis()
        
        return {
            'best_parameters': self.best_params,
            'best_score': self.best_score,
            'final_performance': final_results['performance_metrics'],
            'strategy_analysis': strategy_analysis
        } 