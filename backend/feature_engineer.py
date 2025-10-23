"""
Complete Feature Engineering Pipeline - FIXED VERSION
Matches your training code exactly with robust error handling
"""

import pandas as pd
import numpy as np
from typing import Dict, List
import logging

logger = logging.getLogger(__name__)


class FeatureEngineer:
    """
    Creates features matching training pipeline for PM2.5, PM10, NO2, OZONE
    """
    
    # Weather features (raw)
    WEATHER_FEATURES = [
        'temperature', 'humidity', 'dewPoint', 'apparentTemperature',
        'precipIntensity', 'pressure', 'surfacePressure',
        'cloudCover', 'windSpeed', 'windBearing', 'windGust'
    ]
    
    # Lag windows
    LAG_WINDOWS = [1, 2, 3, 6, 12, 24, 48, 72, 96]
    
    # Rolling windows
    ROLLING_WINDOWS = [6, 12, 24, 48]
    WEATHER_ROLLING_WINDOWS = [6, 12, 24]
    
    def __init__(self):
        self.feature_cache = {}
    
    def engineer_features(self, 
                         current_data: pd.DataFrame, 
                         historical_data: pd.DataFrame,
                         pollutant: str,
                         horizon: str) -> pd.DataFrame:
        """
        Create features for prediction
        
        Args:
            current_data: Current timestamp data (single row)
            historical_data: Historical data sorted by timestamp (for lags)
            pollutant: 'PM25' | 'PM10' | 'NO2' | 'OZONE'
            horizon: '1h' | '6h' | '12h' | '24h'
        
        Returns:
            DataFrame with engineered features (single row)
        """
        try:
            # Start with current data
            features = current_data.copy()
            
            # Ensure we have a clean copy
            if len(features) > 1:
                features = features.iloc[[-1]].copy()
            
            # 1. Create lag features
            features = self._add_lag_features(
                features, historical_data, pollutant, horizon
            )
            
            # 2. Create pollutant rolling features
            features = self._add_pollutant_rolling(
                features, historical_data, pollutant, horizon
            )
            
            # 3. Create weather rolling features
            features = self._add_weather_rolling(
                features, historical_data, horizon
            )
            
            # 4. Add enhanced interaction features
            features = self._add_enhanced_features(features, pollutant)
            
            # 5. Select features based on horizon
            features = self._select_horizon_features(
                features, pollutant, horizon
            )
            
            # 6. Fill nulls and cast to float32
            features = self._finalize_features(features)
            
            logger.info(f"Engineered {len(features.columns)} features for {pollutant} {horizon}")
            
            return features
            
        except Exception as e:
            logger.error(f"Feature engineering failed: {e}")
            raise
    
    def _add_lag_features(self, 
                         current: pd.DataFrame,
                         historical: pd.DataFrame,
                         pollutant: str,
                         horizon: str) -> pd.DataFrame:
        """
        Add lag features based on horizon
        
        1h: lags at 1, 2, 3, 6, 12, 24
        6h: lags at 6, 12, 24, 48
        12h: lags at 12, 24, 48, 72
        24h: lags at 24, 48, 72, 96
        """
        lag_map = {
            '1h': [1, 2, 3, 6, 12, 24],
            '6h': [6, 12, 24, 48],
            '12h': [12, 24, 48, 72],
            '24h': [24, 48, 72, 96]
        }
        
        lags = lag_map[horizon]
        
        # Ensure pollutant column exists
        if pollutant not in historical.columns:
            logger.warning(f"{pollutant} not found in historical data")
            for lag in lags:
                current.loc[current.index[0], f'{pollutant}_lag_{lag}h'] = np.nan
            return current
        
        for lag in lags:
            lag_col = f'{pollutant}_lag_{lag}h'
            if len(historical) >= lag:
                current.loc[current.index[0], lag_col] = historical[pollutant].iloc[-lag]
            else:
                current.loc[current.index[0], lag_col] = np.nan
        
        return current
    
    def _add_pollutant_rolling(self,
                              current: pd.DataFrame,
                              historical: pd.DataFrame,
                              pollutant: str,
                              horizon: str) -> pd.DataFrame:
        """
        Add rolling statistics for pollutant
        
        1h: No rolling
        6h: rolling 6h, 12h
        12h: rolling 12h, 24h
        24h: rolling 24h, 48h
        """
        rolling_map = {
            '1h': [],
            '6h': [6, 12],
            '12h': [12, 24],
            '24h': [24, 48]
        }
        
        windows = rolling_map[horizon]
        
        if pollutant not in historical.columns:
            for window in windows:
                current.loc[current.index[0], f'{pollutant}_rolling_{window}h_mean'] = np.nan
                current.loc[current.index[0], f'{pollutant}_rolling_{window}h_std'] = np.nan
                current.loc[current.index[0], f'{pollutant}_rolling_{window}h_min'] = np.nan
                current.loc[current.index[0], f'{pollutant}_rolling_{window}h_max'] = np.nan
            return current
        
        for window in windows:
            if len(historical) >= window:
                recent = historical[pollutant].tail(window)
                
                current.loc[current.index[0], f'{pollutant}_rolling_{window}h_mean'] = recent.mean()
                current.loc[current.index[0], f'{pollutant}_rolling_{window}h_std'] = recent.std()
                current.loc[current.index[0], f'{pollutant}_rolling_{window}h_min'] = recent.min()
                current.loc[current.index[0], f'{pollutant}_rolling_{window}h_max'] = recent.max()
            else:
                current.loc[current.index[0], f'{pollutant}_rolling_{window}h_mean'] = np.nan
                current.loc[current.index[0], f'{pollutant}_rolling_{window}h_std'] = np.nan
                current.loc[current.index[0], f'{pollutant}_rolling_{window}h_min'] = np.nan
                current.loc[current.index[0], f'{pollutant}_rolling_{window}h_max'] = np.nan
        
        return current
    
    def _add_weather_rolling(self,
                            current: pd.DataFrame,
                            historical: pd.DataFrame,
                            horizon: str) -> pd.DataFrame:
        """
        Add weather rolling statistics
        """
        window_map = {
            '1h': None,
            '6h': 6,
            '12h': 12,
            '24h': 24
        }
        
        window = window_map[horizon]
        
        if window is not None:
            for feature in self.WEATHER_FEATURES:
                if feature in historical.columns:
                    if len(historical) >= window:
                        recent = historical[feature].tail(window)
                        current.loc[current.index[0], f'{feature}_rolling_{window}h_mean'] = recent.mean()
                        current.loc[current.index[0], f'{feature}_rolling_{window}h_std'] = recent.std()
                    else:
                        current.loc[current.index[0], f'{feature}_rolling_{window}h_mean'] = np.nan
                        current.loc[current.index[0], f'{feature}_rolling_{window}h_std'] = np.nan
        
        return current
    
    def _add_enhanced_features(self,
                              features: pd.DataFrame,
                              pollutant: str) -> pd.DataFrame:
        """
        Add interaction features computed during training
        """
        # Short/long change from lags
        lag_cols = [col for col in features.columns if f'{pollutant}_lag_' in col]
        
        if len(lag_cols) >= 2:
            features.loc[features.index[0], f'{pollutant}_short_change'] = (
                features[lag_cols[0]].iloc[0] - features[lag_cols[1]].iloc[0]
            )
            
            if len(lag_cols) > 2:
                features.loc[features.index[0], f'{pollutant}_long_change'] = (
                    features[lag_cols[0]].iloc[0] - features[lag_cols[-1]].iloc[0]
                )
        
        # Weather deviations
        weather_vars = ['temperature', 'humidity', 'pressure', 'windSpeed']
        
        for var in weather_vars:
            if var in features.columns:
                rolling_cols = [col for col in features.columns 
                               if f'{var}_rolling_' in col and '_mean' in col]
                
                if rolling_cols:
                    features.loc[features.index[0], f'{var}_deviation'] = (
                        features[var].iloc[0] - features[rolling_cols[0]].iloc[0]
                    )
        
        # Wind-humidity interaction (common to all pollutants)
        if 'windSpeed' in features.columns and 'humidity' in features.columns:
            features.loc[features.index[0], 'wind_humidity_interaction'] = (
                features['windSpeed'].iloc[0] * features['humidity'].iloc[0]
            )
        
        # Pollutant-specific interactions
        if pollutant == 'OZONE':
            if 'temperature' in features.columns and 'humidity' in features.columns:
                features.loc[features.index[0], 'temp_humidity_interaction'] = (
                    features['temperature'].iloc[0] * (100 - features['humidity'].iloc[0])
                )
            
            if 'windSpeed' in features.columns and 'temperature' in features.columns:
                features.loc[features.index[0], 'wind_temp_interaction'] = (
                    features['windSpeed'].iloc[0] * features['temperature'].iloc[0]
                )
            
            if 'cloudCover' in features.columns and 'temperature' in features.columns:
                features.loc[features.index[0], 'cloud_temp_interaction'] = (
                    (1 - features['cloudCover'].iloc[0]) * features['temperature'].iloc[0]
                )
        
        elif pollutant == 'NO2':
            if 'windSpeed' in features.columns and 'temperature' in features.columns:
                features.loc[features.index[0], 'wind_temp_interaction'] = (
                    features['windSpeed'].iloc[0] * features['temperature'].iloc[0]
                )
        
        return features
    
    def _select_horizon_features(self,
                                features: pd.DataFrame,
                                pollutant: str,
                                horizon: str) -> pd.DataFrame:
        """
        Select only features used for this horizon
        """
        # Exclude columns that should never be in model
        exclude_cols = {
            'location', 'location_aq', 'exposure_category',
            'region', 'season', 'timestamp', 'location_id',
            'date', 'lat', 'lng', 'lon',
            # Exclude contemporaneous pollutants (data leakage)
            'PM25', 'PM10', 'NO2', 'OZONE', 'CO', 'SO2', 'AQI',
            # Exclude all targets
            'target_1h', 'target_6h', 'target_12h', 'target_24h'
        }
        
        # Keep only numeric columns not in exclude list
        numeric_cols = [
            col for col in features.columns
            if col not in exclude_cols and 
            pd.api.types.is_numeric_dtype(features[col])
        ]
        
        return features[numeric_cols]
    
    def _finalize_features(self, features: pd.DataFrame) -> pd.DataFrame:
        """
        Final processing: fill nulls + cast to float32
        """
        # Fill nulls with column median
        for col in features.columns:
            if features[col].isna().any():
                median_val = features[col].median()
                if pd.isna(median_val):
                    median_val = 0.0
                features[col].fillna(median_val, inplace=True)
        
        # If still nulls, fill with 0
        features.fillna(0, inplace=True)
        
        # Cast to float32
        features = features.astype(np.float32)
        
        return features


class FeatureAligner:
    """
    Aligns features to match model's expected input
    """
    
    def align_features(self,
                      features: pd.DataFrame,
                      model_features: List[str]) -> pd.DataFrame:
        """
        Reindex features to match model's training features
        
        Args:
            features: Engineered features
            model_features: Feature names from model artifact
        
        Returns:
            Aligned DataFrame with exact columns in exact order
        """
        # Reindex with fill_value=0.0 for missing features
        aligned = features.reindex(columns=model_features, fill_value=0.0)
        
        # Ensure float32
        aligned = aligned.astype(np.float32)
        
        logger.info(f"Aligned to {len(aligned.columns)} model features")
        
        return aligned
