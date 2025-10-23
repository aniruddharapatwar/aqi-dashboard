"""
Complete Model Predictor with Comprehensive AQI Calculator - FIXED FILE PATHS
Handles flat file structure: model_artifacts_POLLUTANT_HORIZON.pkl
"""

import pickle
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import logging

# Import from same directory
try:
    from data.feature_engineer import FeatureEngineer, FeatureAligner
except ImportError:
    from feature_engineer import FeatureEngineer, FeatureAligner

logger = logging.getLogger(__name__)


# ============================================================================
# AQI BREAKPOINTS AND STANDARDS
# ============================================================================

INDIAN_AQI_BREAKPOINTS = {
    'PM25': {
        'Good': (0, 30), 'Satisfactory': (31, 60), 'Moderate': (61, 90),
        'Poor': (91, 120), 'Very_Poor': (121, 250), 'Severe': (251, 500)
    },
    'PM10': {
        'Good': (0, 50), 'Satisfactory': (51, 100), 'Moderate': (101, 250),
        'Poor': (251, 350), 'Very_Poor': (351, 430), 'Severe': (431, 600)
    },
    'NO2': {
        'Good': (0, 40), 'Satisfactory': (41, 80), 'Moderate': (81, 180),
        'Poor': (181, 280), 'Very_Poor': (281, 400), 'Severe': (401, 500)
    },
    'OZONE': {
        'Good': (0, 50), 'Satisfactory': (51, 100), 'Moderate': (101, 168),
        'Poor': (169, 208), 'Very_Poor': (209, 748), 'Severe': (749, 1000)
    }
}

INDIAN_AQI_INDEX = {
    'Good': (0, 50), 'Satisfactory': (51, 100), 'Moderate': (101, 200),
    'Poor': (201, 300), 'Very_Poor': (301, 400), 'Severe': (401, 500)
}

US_AQI_BREAKPOINTS = {
    'PM25': {
        'Good': (0.0, 12.0), 'Moderate': (12.1, 35.4), 'Unhealthy_for_Sensitive': (35.5, 55.4),
        'Unhealthy': (55.5, 150.4), 'Very_Unhealthy': (150.5, 250.4), 'Hazardous': (250.5, 500.4)
    },
    'PM10': {
        'Good': (0, 54), 'Moderate': (55, 154), 'Unhealthy_for_Sensitive': (155, 254),
        'Unhealthy': (255, 354), 'Very_Unhealthy': (355, 424), 'Hazardous': (425, 604)
    },
    'NO2': {
        'Good': (0, 53), 'Moderate': (54, 100), 'Unhealthy_for_Sensitive': (101, 360),
        'Unhealthy': (361, 649), 'Very_Unhealthy': (650, 1249), 'Hazardous': (1250, 2049)
    },
    'OZONE': {
        'Good': (0.000, 0.054), 'Moderate': (0.055, 0.070), 'Unhealthy_for_Sensitive': (0.071, 0.085),
        'Unhealthy': (0.086, 0.105), 'Very_Unhealthy': (0.106, 0.200), 'Hazardous': (0.201, 0.604)
    }
}

US_AQI_INDEX = {
    'Good': (0, 50), 'Moderate': (51, 100), 'Unhealthy_for_Sensitive': (101, 150),
    'Unhealthy': (151, 200), 'Very_Unhealthy': (201, 300), 'Hazardous': (301, 500)
}

CATEGORY_MAPPING = {
    'Good': {'us': 'Good'},
    'Satisfactory': {'us': 'Moderate'},
    'Moderate': {'us': 'Unhealthy_for_Sensitive'},
    'Poor': {'us': 'Unhealthy'},
    'Very_Poor': {'us': 'Very_Unhealthy'},
    'Severe': {'us': 'Hazardous'}
}


# ============================================================================
# MODEL MANAGER - FIXED FOR FLAT FILE STRUCTURE
# ============================================================================

class ModelManager:
    """
    Loads and manages trained models for all pollutants and horizons
    FIXED: Handles flat file structure model_artifacts_POLLUTANT_HORIZON.pkl
    """
    
    def __init__(self, model_path: Path):
        self.model_path = Path(model_path)
        self.models = {}  # Cache loaded models
    
    def load_model(self, pollutant: str, horizon: str) -> Dict:
        """
        Load model artifact for specific pollutant and horizon
        
        File naming: model_artifacts_POLLUTANT_HORIZON.pkl
        Example: model_artifacts_PM25_1h.pkl
        """
        cache_key = f"{pollutant}_{horizon}"
        
        if cache_key in self.models:
            return self.models[cache_key]
        
        # FIXED: New file naming convention
        model_file = self.model_path / f"model_artifacts_{pollutant}_{horizon}.pkl"
        
        if not model_file.exists():
            raise FileNotFoundError(f"Model not found: {model_file}")
        
        logger.info(f"Loading model: {pollutant} {horizon}")
        
        with open(model_file, 'rb') as f:
            model_artifact = pickle.load(f)
        
        # Cache for future use
        self.models[cache_key] = model_artifact
        
        logger.info(f"  ✓ Loaded model with {len(model_artifact['feature_names'])} features")
        
        return model_artifact


# ============================================================================
# AQI CALCULATOR
# ============================================================================

class AQICalculator:
    """Calculate detailed AQI with ranges for both Indian and US standards"""
    
    def __init__(self):
        self.indian_breakpoints = INDIAN_AQI_BREAKPOINTS
        self.us_breakpoints = US_AQI_BREAKPOINTS
        self.indian_index = INDIAN_AQI_INDEX
        self.us_index = US_AQI_INDEX
    
    def category_to_aqi_range_indian(self, category: str) -> Tuple[int, int]:
        normalized_category = category.replace(' ', '_')
        if normalized_category not in self.indian_index:
            logger.warning(f"Unknown category: {category}, defaulting to Severe")
            normalized_category = 'Severe'
        return self.indian_index[normalized_category]
    
    def category_to_aqi_range_us(self, category: str) -> Tuple[int, int]:
        normalized_category = category.replace(' ', '_')
        us_category = CATEGORY_MAPPING.get(normalized_category, {}).get('us', 'Hazardous')
        if us_category not in self.us_index:
            logger.warning(f"Unknown US category: {us_category}, defaulting to Hazardous")
            us_category = 'Hazardous'
        return self.us_index[us_category]
    
    def get_concentration_range(self, pollutant: str, category: str, standard: str = 'IN') -> Tuple[float, float]:
        normalized_category = category.replace(' ', '_')
        breakpoints = self.indian_breakpoints if standard == 'IN' else self.us_breakpoints
        
        if pollutant in breakpoints:
            if standard == 'IN' and normalized_category in breakpoints[pollutant]:
                return breakpoints[pollutant][normalized_category]
            elif standard == 'US':
                us_category = CATEGORY_MAPPING.get(normalized_category, {}).get('us', 'Hazardous')
                if us_category in breakpoints[pollutant]:
                    return breakpoints[pollutant][us_category]
        
        return (0, 0)
    
    def calculate_sub_index_indian(self, pollutant: str, category: str, confidence: float) -> Dict:
        aqi_min, aqi_max = self.category_to_aqi_range_indian(category)
        conc_min, conc_max = self.get_concentration_range(pollutant, category, 'IN')
        
        return {
            'pollutant': pollutant,
            'category': category,
            'aqi_range': (aqi_min, aqi_max),
            'aqi_min': aqi_min,
            'aqi_max': aqi_max,
            'aqi_mid': (aqi_min + aqi_max) / 2,
            'concentration_range': (conc_min, conc_max),
            'confidence': confidence
        }
    
    def calculate_sub_index_us(self, pollutant: str, category: str, confidence: float) -> Dict:
        normalized_category = category.replace(' ', '_')
        us_category = CATEGORY_MAPPING.get(normalized_category, {}).get('us', 'Hazardous')
        aqi_min, aqi_max = self.category_to_aqi_range_us(category)
        conc_min, conc_max = self.get_concentration_range(pollutant, category, 'US')
        
        return {
            'pollutant': pollutant,
            'indian_category': category,
            'us_category': us_category,
            'aqi_range': (aqi_min, aqi_max),
            'aqi_min': aqi_min,
            'aqi_max': aqi_max,
            'aqi_mid': (aqi_min + aqi_max) / 2,
            'concentration_range': (conc_min, conc_max),
            'confidence': confidence
        }
    
    def calculate_overall_aqi(self, predictions: Dict[str, Tuple[str, float]], standard: str = 'IN') -> Dict:
        sub_indices = []
        
        for pollutant, (category, confidence) in predictions.items():
            if pollutant in ['PM25', 'PM10', 'NO2', 'OZONE']:
                try:
                    if standard == 'IN':
                        sub_index = self.calculate_sub_index_indian(pollutant, category, confidence)
                    else:
                        sub_index = self.calculate_sub_index_us(pollutant, category, confidence)
                    sub_indices.append(sub_index)
                except Exception as e:
                    logger.error(f"Failed to calculate sub-index for {pollutant}: {e}")
        
        if not sub_indices:
            return {'error': 'No valid predictions'}
        
        # Overall AQI is the maximum (worst) sub-index
        max_sub_index = max(sub_indices, key=lambda x: x['aqi_mid'])
        
        result = {
            'standard': standard,
            'aqi_range': max_sub_index['aqi_range'],
            'aqi_min': max_sub_index['aqi_min'],
            'aqi_max': max_sub_index['aqi_max'],
            'aqi_mid': max_sub_index['aqi_mid'],
            'dominant_pollutant': max_sub_index['pollutant'],
            'dominant_category': max_sub_index.get('us_category', max_sub_index['category']),
            'category': max_sub_index.get('us_category', max_sub_index['category']),
            'confidence': max_sub_index['confidence'],
            'sub_indices': sub_indices,
            'health_implications': self._get_health_implications(
                max_sub_index.get('us_category', max_sub_index['category']), 
                standard
            )
        }
        
        return result
    
    def _get_health_implications(self, category: str, standard: str) -> str:
        normalized_category = category.replace(' ', '_')
        
        if standard == 'IN':
            implications = {
                'Good': 'Minimal Impact',
                'Satisfactory': 'Minor breathing discomfort to sensitive people',
                'Moderate': 'Breathing discomfort to people with lung/heart disease, children, older adults',
                'Poor': 'Breathing discomfort to most people on prolonged exposure',
                'Very_Poor': 'Respiratory illness on prolonged exposure',
                'Severe': 'Affects healthy people and seriously impacts those with existing conditions'
            }
        else:
            implications = {
                'Good': 'Air quality is satisfactory, and air pollution poses little or no risk',
                'Moderate': 'Air quality is acceptable. However, there may be a risk for some people',
                'Unhealthy_for_Sensitive': 'Members of sensitive groups may experience health effects',
                'Unhealthy': 'Some members of the general public may experience health effects',
                'Very_Unhealthy': 'Health alert: The risk of health effects is increased for everyone',
                'Hazardous': 'Health warning of emergency conditions: everyone is more likely to be affected'
            }
        
        return implications.get(normalized_category, 'Unknown')


# ============================================================================
# POLLUTANT PREDICTOR
# ============================================================================

class PollutantPredictor:
    """Complete prediction pipeline for all pollutants"""
    
    def __init__(self, model_path: Path):
        self.model_manager = ModelManager(model_path)
        self.feature_engineer = FeatureEngineer()
        self.feature_aligner = FeatureAligner()
        self.aqi_calculator = AQICalculator()
        
        self.pollutants = ['PM25', 'PM10', 'NO2', 'OZONE']
    
    def predict_single_pollutant(self,
                                current_data: pd.DataFrame,
                                historical_data: pd.DataFrame,
                                pollutant: str,
                                horizon: str) -> Tuple[str, float]:
        """Predict category for a single pollutant at specific horizon"""
        try:
            # 1. Load model
            model_artifact = self.model_manager.load_model(pollutant, horizon)
            
            # 2. Engineer features
            features = self.feature_engineer.engineer_features(
                current_data=current_data,
                historical_data=historical_data,
                pollutant=pollutant,
                horizon=horizon
            )
            
            # 3. Align features to model's expected input
            aligned_features = self.feature_aligner.align_features(
                features=features,
                model_features=model_artifact['feature_names']
            )
            
            # 4. Make prediction using calibrated model
            calibrated_model = model_artifact['calibrated_model']
            
            # Get probabilities
            probabilities = calibrated_model.predict_proba(aligned_features)[0]
            
            # Apply health-aware thresholds
            predicted_class_idx = self._apply_health_thresholds(
                probabilities=probabilities,
                thresholds=model_artifact['thresholds'],
                classes=model_artifact['classes']
            )
            
            # Get category name
            predicted_category = model_artifact['classes'][predicted_class_idx]
            
            # Confidence is the probability of predicted class
            confidence = probabilities[predicted_class_idx]
            
            logger.info(f"  ✓ {pollutant} {horizon}: {predicted_category} ({confidence:.2%})")
            
            return predicted_category, confidence
            
        except Exception as e:
            logger.error(f"Prediction failed for {pollutant} {horizon}: {e}")
            raise
    
    def _apply_health_thresholds(self,
                                probabilities: np.ndarray,
                                thresholds: Dict[str, float],
                                classes: List[str]) -> int:
        """Apply health-aware thresholds"""
        candidate_classes = []
        
        for i, class_name in enumerate(classes):
            if probabilities[i] >= thresholds[class_name]:
                candidate_classes.append((i, probabilities[i]))
        
        if candidate_classes:
            predicted_class_idx = max(candidate_classes, key=lambda x: x[1])[0]
        else:
            predicted_class_idx = np.argmax(probabilities)
        
        return predicted_class_idx
    
    def predict_all_horizons(self,
                           current_data: pd.DataFrame,
                           historical_data: pd.DataFrame,
                           pollutant: str,
                           standard: str = 'IN') -> Dict:
        """Predict for all horizons (1h, 6h, 12h, 24h)"""
        results = {}
        
        for horizon in ['1h', '6h', '12h', '24h']:
            try:
                category, confidence = self.predict_single_pollutant(
                    current_data=current_data,
                    historical_data=historical_data,
                    pollutant=pollutant,
                    horizon=horizon
                )
                
                # Calculate AQI details for this pollutant
                if standard == 'IN':
                    sub_index = self.aqi_calculator.calculate_sub_index_indian(
                        pollutant, category, confidence
                    )
                else:
                    sub_index = self.aqi_calculator.calculate_sub_index_us(
                        pollutant, category, confidence
                    )
                
                results[horizon] = {
                    'category': category,
                    'confidence': confidence,
                    'aqi_range': sub_index['aqi_range'],
                    'aqi_min': sub_index['aqi_min'],
                    'aqi_max': sub_index['aqi_max'],
                    'aqi_mid': sub_index['aqi_mid'],
                    'concentration_range': sub_index['concentration_range']
                }
                
            except Exception as e:
                logger.error(f"Failed to predict {pollutant} {horizon}: {e}")
                results[horizon] = {
                    'category': 'Unknown',
                    'confidence': 0.0,
                    'aqi_range': (0, 0),
                    'aqi_min': 0,
                    'aqi_max': 0,
                    'aqi_mid': 0,
                    'concentration_range': (0, 0),
                    'error': str(e)
                }
        
        return results
    
    def predict_all_pollutants_all_horizons(self,
                                           current_data: pd.DataFrame,
                                           historical_data: pd.DataFrame,
                                           standard: str = 'IN') -> Dict:
        """Comprehensive prediction for all pollutants across all horizons"""
        results = {}
        
        # Predict each pollutant
        for pollutant in self.pollutants:
            logger.info(f"\n🔍 Predicting {pollutant}...")
            results[pollutant] = self.predict_all_horizons(
                current_data=current_data,
                historical_data=historical_data,
                pollutant=pollutant,
                standard=standard
            )
        
        # Calculate overall AQI for each horizon
        overall_results = {}
        
        for horizon in ['1h', '6h', '12h', '24h']:
            # Gather predictions for this horizon
            predictions = {
                pollutant: (
                    results[pollutant][horizon]['category'],
                    results[pollutant][horizon]['confidence']
                )
                for pollutant in self.pollutants
                if 'error' not in results[pollutant][horizon]
            }
            
            # Calculate overall AQI
            overall = self.aqi_calculator.calculate_overall_aqi(predictions, standard)
            
            overall_results[horizon] = overall
        
        results['overall'] = overall_results
        
        return results
