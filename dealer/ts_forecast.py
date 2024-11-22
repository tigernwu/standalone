import pandas as pd
import requests
import json
from typing import Optional, List, Union
from core.utils.config_setting import Config


class TSForecast:
    def __init__(self, base_url: str=None, api_key: str=None):
        self.base_url = Config.get_config( "AZURE_TIMEGEN_ENDPOINT","forecast")
        self.api_key = Config.get_config("AZURE_TIMEGEN_API_KEY","forecast")
        self.df = None
        self.forecast_df = None
        self.time_col = None
        self.target_col = None

    def set_data(self, df: pd.DataFrame, time_col: str="date", target_col: str="close"):
        """Set the data using a DataFrame."""
        self.df = df.copy()
        self.time_col = time_col
        self.target_col = target_col
        self.df[self.time_col] = pd.to_datetime(self.df[self.time_col])

    def forecast(self, h: int, freq: Optional[str] = None, 
                 level: Optional[List[Union[int, float]]] = None,
                 clean_ex_first: bool = True, finetune_steps: int = 0,
                 finetune_loss: str = "default") -> pd.DataFrame:
        """Make a forecast using the TimeGEN-1 model."""
        if self.df is None:
            raise ValueError("No data set. Please use set_data() first.")

        forecast_params = {
            "h": h,
            "freq": freq,
            "level": level,
            "clean_ex_first": clean_ex_first,
            "finetune_steps": finetune_steps,
            "finetune_loss": finetune_loss
        }

        # Remove None values from params
        forecast_params = {k: v for k, v in forecast_params.items() if v is not None}

        try:
            self.forecast_df = self.client.forecast(
                df=self.df,
                time_col=self.time_col,
                target_col=self.target_col,
                **forecast_params
            )
            return self.forecast_df
        except Exception as e:
            raise Exception(f"Forecast failed: {str(e)}")

    def get_forecast(self) -> Optional[pd.DataFrame]:
        """Return the forecast DataFrame if available."""
        return self.forecast_df


# Example usage
if __name__ == "__main__":
    import numpy as np

    # Assume df is a pandas DataFrame with your time series data
    df = pd.DataFrame({
        'timestamp': pd.date_range(start='2023-01-01', periods=100, freq='D'),
        'value': np.random.randn(100).cumsum()  # Example random data
    })

    forecaster = TSForecast(base_url="your_azure_ai_endpoint", api_key="your_api_key")
    forecaster.set_data(df, time_col="timestamp", target_col="value")
    forecast_result = forecaster.forecast(h=12)
    print(forecast_result)