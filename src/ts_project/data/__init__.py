"""Shared data loading and forecasting-window utilities."""

from ts_project.data.etth1 import (
    BENCHMARK_ROWS,
    FEATURE_COLUMNS,
    TEST_ROWS,
    TRAIN_ROWS,
    VALIDATION_ROWS,
    ETTh1Data,
    load_etth1,
    prepare_etth1,
)
from ts_project.data.windows import ForecastWindowDataset, build_window_datasets
from ts_project.data.weather import (
    EXPECTED_CHANNELS as WEATHER_CHANNELS,
    EXPECTED_ROWS as WEATHER_ROWS,
    WeatherData,
    WeatherSplitBounds,
    build_weather_window_datasets,
    load_weather,
    prepare_weather,
    weather_split_bounds,
)

__all__ = [
    "BENCHMARK_ROWS",
    "ETTh1Data",
    "FEATURE_COLUMNS",
    "ForecastWindowDataset",
    "TEST_ROWS",
    "TRAIN_ROWS",
    "VALIDATION_ROWS",
    "WEATHER_CHANNELS",
    "WEATHER_ROWS",
    "WeatherData",
    "WeatherSplitBounds",
    "build_window_datasets",
    "build_weather_window_datasets",
    "load_etth1",
    "load_weather",
    "prepare_etth1",
    "prepare_weather",
    "weather_split_bounds",
]
