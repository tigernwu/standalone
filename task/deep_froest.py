import copy
from datetime import datetime
import importlib
import json
from pathlib import Path
import sys
from typing import Optional, Tuple, List, Dict, Any
import concurrent.futures

import joblib
import numpy as np
import pandas as pd
from deepforest import CascadeForestRegressor
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import optuna

from core.utils.log import logger
from core.stock.tushare_data_provider import get_technical_factor

def lazy(fullname):
    if fullname in sys.modules:
        return sys.modules[fullname]
    try:
        return importlib.import_module(fullname)
    except ImportError:
        return None

class StockPricePredictor:
    def __init__(self, 
                 predict_days: int = 5,  
                 lookback_days: int = 20,  
                 test_size: int = 60,  
                 target_col: str = 'close',
                 model_dir: str = './output/models/stock_predictors'): 
        self.predict_days = predict_days
        self.lookback_days = lookback_days
        self.test_size = test_size
        self.target_col = target_col
        self.model = None
        self.feature_cols = []
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)
        
        # GPU detection
        try:
            torch = lazy('torch')
            if torch:
                self.use_gpu = torch.cuda.is_available()
            else:
                self.use_gpu = False
        except Exception as e:
            self.use_gpu = False
        
        self.default_params = {
            'n_estimators': 500,           # Number of trees in each forest
            'n_trees': 100,                # Number of trees in each cascade layer
            'n_layers': 3,                 # Number of cascade layers
            'max_depth': 8,                # Maximum depth of trees
            'min_samples_leaf': 2,         # Minimum samples in leaf nodes
            'random_state': 42,
            'n_jobs': -1 if not self.use_gpu else 1  # Use all CPU cores if not using GPU
        }

    def prepare_data(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Prepare training, validation and test data with memory optimization"""
        df = df.sort_values('date', ascending=False).reset_index(drop=True)
        
        # Remove unnecessary columns
        exclude_cols = ['date', 'open', 'high', 'low', 'close', 'vol', 'code']
        self.feature_cols = [col for col in df.columns if col not in exclude_cols]
        
        if not self.feature_cols:
            raise ValueError(f"No feature columns found after excluding: {exclude_cols}")
        
        # Create features and target arrays efficiently
        X = df[self.feature_cols].values
        y = df[self.target_col].values
        
        # Create time series sequences using numpy operations
        n_samples = len(df) - self.lookback_days - self.predict_days + 1
        X_sequences = np.zeros((n_samples, self.lookback_days, len(self.feature_cols)))
        y_sequences = np.zeros(n_samples)
        
        for i in range(n_samples):
            X_sequences[i] = X[i:i+self.lookback_days][::-1]
            y_sequences[i] = y[i+self.predict_days-1]
        
        # Flatten feature matrix
        X_flat = X_sequences.reshape(X_sequences.shape[0], -1)
        
        # Split data
        train_size = len(X_flat) - self.test_size
        val_size = int(train_size * 0.2)
        
        if train_size - val_size <= 0:
            raise ValueError(f"Insufficient data for train-val split")
        
        return (X_flat[:train_size-val_size], X_flat[train_size-val_size:train_size], X_flat[train_size:],
                y_sequences[:train_size-val_size], y_sequences[train_size-val_size:train_size], y_sequences[train_size:])

    def optimize_params(self, X_train: np.ndarray, y_train: np.ndarray,
                       X_val: np.ndarray, y_val: np.ndarray,
                       n_trials: int = 100) -> Dict[str, Any]:
        """Optimize model parameters using Optuna"""
        def objective(trial):
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 200, 1000),
                'n_trees': trial.suggest_int('n_trees', 50, 200),
                'n_layers': trial.suggest_int('n_layers', 2, 5),
                'max_depth': trial.suggest_int('max_depth', 4, 12),
                'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
                'random_state': 42,
                'n_jobs': -1 if not self.use_gpu else 1
            }
            
            model = CascadeForestRegressor(**params)
            model.fit(X_train, y_train)
            
            val_pred = model.predict(X_val)
            return mean_squared_error(y_val, val_pred, squared=False)

        study = optuna.create_study(direction='minimize')
        study.optimize(objective, n_trials=n_trials)
        
        best_params = study.best_params
        best_params.update({k: v for k, v in self.default_params.items() 
                          if k not in best_params})
        return best_params
    
    def train(self, X_train: np.ndarray, y_train: np.ndarray, params: Dict[str, Any] = None):
        """Train the model with the given parameters"""
        if params is None:
            params = self.default_params
        
        self.model = CascadeForestRegressor(**params)
        self.model.fit(X_train, y_train)
    
    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, float]:
        """Evaluate model performance"""
        predictions = self.model.predict(X_test)
        
        return {
            'rmse': mean_squared_error(y_test, predictions, squared=False),
            'mae': mean_absolute_error(y_test, predictions),
            'r2': r2_score(y_test, predictions)
        }
    
    def predict_next(self, df: pd.DataFrame) -> float:
        """Predict the next price"""
        df = df.sort_values('date', ascending=False)
        recent_data = df.head(self.lookback_days)
        
        if len(recent_data) < self.lookback_days:
            raise ValueError(f"Insufficient data: need {self.lookback_days} days")
        
        features = recent_data[self.feature_cols].values[::-1]
        features = features.reshape(1, -1)
        
        predicted_price = self.model.predict(features)[0]
        
        # Apply price limits
        latest_close = df.iloc[0][self.target_col]
        max_up = latest_close * 1.1
        max_down = latest_close * 0.9
        predicted_price = max(min(predicted_price, max_up), max_down)
        
        return predicted_price

    def save_model(self, symbol: str) -> str:
        """Save the model and configuration"""
        if self.model is None:
            raise ValueError("No model to save")
        
        model_path = self.model_dir / f"{symbol}"
        model_path.mkdir(exist_ok=True)
        
        joblib.dump(self.model, model_path / "model.joblib")
        
        config = {
            'predict_days': self.predict_days,
            'lookback_days': self.lookback_days,
            'test_size': self.test_size,
            'target_col': self.target_col,
            'feature_cols': self.feature_cols,
        }
        joblib.dump(config, model_path / "config.joblib")
        
        return str(model_path)
    
    def load_model(self, model_path: str, config: Dict[str, Any] = None):
        """Load a saved model and configuration"""
        model_path = Path(model_path)
        if not model_path.exists():
            raise ValueError(f"Model path {model_path} does not exist")
        
        if config is None and (model_path / "config.joblib").exists():
            config = joblib.load(model_path / "config.joblib")
        elif config is None:
            config = {
                'predict_days': self.predict_days,
                'lookback_days': self.lookback_days,
                'test_size': self.test_size,
                'target_col': self.target_col,
                'feature_cols': self.feature_cols,
            }
        
        self.model = joblib.load(model_path / "model.joblib")

def run_single_stock_prediction(symbol: str, days: int = 400, 
                              optimize: bool = False,
                              save_model: bool = True) -> Tuple[StockPricePredictor, Dict[str, Any]]:
    """Run prediction for a single stock"""
    try:
        data = get_technical_factor(symbol, days=days)
        actual_days = len(data)
        
        lookback_days = max(min(int(actual_days * 0.12), 30), 10)
        test_size = max(min(int(actual_days * 0.18), 60), 20)
        
        predictor = StockPricePredictor(
            predict_days=5,
            lookback_days=lookback_days,
            test_size=test_size
        )
        
        X_train, X_val, X_test, y_train, y_val, y_test = predictor.prepare_data(data)
        
        if optimize:
            best_params = predictor.optimize_params(X_train, y_train, X_val, y_val)
            predictor.train(X_train, y_train, best_params)
        else:
            predictor.train(X_train, y_train)
        
        metrics = predictor.evaluate(X_test, y_test)
        next_price = predictor.predict_next(data)
        latest_price = data.iloc[0]['close']
        price_change = (next_price - latest_price) / latest_price * 100
        
        model_path = predictor.save_model(symbol) if save_model else None
        
        results = {
            'symbol': symbol,
            'metrics': metrics,
            'next_price': next_price,
            'latest_price': latest_price,
            'price_change': price_change,
            'model_path': model_path
        }
        
        return predictor, results
        
    except Exception as e:
        logger.error(f"Error predicting {symbol}: {str(e)}")
        raise

def run_stock_prediction(symbols: List[str], 
                        days: int = 400,
                        optimize: bool = False,
                        max_workers: int = None,
                        save_models: bool = True) -> Dict[str, Dict[str, Any]]:
    """Run parallel prediction for multiple stocks"""
    results = {}
    
    def process_stock(symbol: str) -> Tuple[str, Dict[str, Any]]:
        try:
            _, result = run_single_stock_prediction(
                symbol, days, optimize, save_models
            )
            logger.info(f"Successfully predicted {symbol}")
            return symbol, result
        except Exception as e:
            logger.error(f"Failed to predict {symbol}: {str(e)}")
            return symbol, {'error': str(e)}
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_symbol = {
            executor.submit(process_stock, symbol): symbol 
            for symbol in symbols
        }
        
        for future in concurrent.futures.as_completed(future_to_symbol):
            symbol = future_to_symbol[future]
            try:
                symbol, result = future.result()
                results[symbol] = result
            except Exception as e:
                logger.error(f"Error processing {symbol}: {str(e)}")
                results[symbol] = {'error': str(e)}
    
    success_count = sum(1 for r in results.values() if 'error' not in r)
    print(f"\nPrediction Summary:")
    print(f"Total Stocks: {len(symbols)}")
    print(f"Successful: {success_count}")
    print(f"Failed: {len(symbols) - success_count}")
    
    print("\nIndividual Results:")
    for symbol, result in results.items():
        if 'error' in result:
            print(f"{symbol}: Failed - {result['error']}")
        else:
            print(f"{symbol}: Predicted {result['next_price']:.2f} "
                  f"({result['price_change']:+.2f}%), "
                  f"RMSE: {result['metrics']['rmse']:.4f}")
    
    return results

def runner(symbol: str = "601127"):
    predictor, result = run_single_stock_prediction(symbol, optimize=True)
    print(f"\nSingle Stock Prediction:")
    print(f"Predicted Price: {result['next_price']:.2f}")
    print(f"Change: {result['price_change']:+.2f}%")
    print(f"Model Path: {result['model_path']}")

if __name__ == "__main__":
    # Single stock prediction
    symbol = "601127"
    predictor, result = run_single_stock_prediction(symbol, optimize=True)
    print(f"\nSingle Stock Prediction:")
    print(f"Predicted Price: {result['next_price']:.2f}")
    print(f"Change: {result['price_change']:+.2f}%")
    print(f"Model Path: {result['model_path']}")
    
    # Multiple stock parallel prediction
    symbols = ["601127", "600519", "000858"]
    results = run_stock_prediction(
        symbols,
        optimize=True,
        max_workers=3,
        save_models=True
    )