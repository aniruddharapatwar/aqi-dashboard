"""
AQI Dashboard - FastAPI Backend (FINAL FIX - NUMPY + OZONE NAMING)
Complete REST API for air quality predictions and AI assistance
Fixed: OZONE naming (Ozone) and numpy compatibility
"""

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, List, Optional
import pandas as pd
import numpy as np
import pickle
from pathlib import Path
import logging
import json
import os
import sys

# Import Gemini AI with proper error handling
try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False
    logging.warning("google-generativeai not installed. Install with: pip install google-generativeai")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Log numpy version for debugging
logger.info(f"NumPy version: {np.__version__}")
logger.info(f"Python version: {sys.version}")

# ============================================================================
# CONFIGURATION
# ============================================================================

class Config:
    BASE_DIR = Path(__file__).parent
    MODEL_PATH = BASE_DIR / "Classification_trained_models"
    DATA_PATH = BASE_DIR / "data" / "inference_data.csv"
    WHITELIST_PATH = BASE_DIR / "region_wise_popular_places_from_inference.csv"
    
    # Railway uses environment variables, not Streamlit secrets
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
    
    AQI_COLORS_IN = {
        'Good': '#00E400', 'Satisfactory': '#FFFF00', 'Moderate': '#FF7E00',
        'Poor': '#FF0000', 'Very_Poor': '#8F3F97', 'Severe': '#7E0023'
    }
    
    AQI_COLORS_US = {
        'Good': '#00E400', 'Moderate': '#FFFF00', 'Unhealthy_for_Sensitive': '#FF7E00',
        'Unhealthy': '#FF0000', 'Very_Unhealthy': '#8F3F97', 'Hazardous': '#7E0023'
    }

# ============================================================================
# MODELS
# ============================================================================

class PredictionRequest(BaseModel):
    location: str
    standard: str = "IN"

class ChatRequest(BaseModel):
    message: str
    location: Optional[str] = None
    aqi_data: Optional[Dict] = None
    user_profile: Optional[Dict] = None

# ============================================================================
# FASTAPI APP
# ============================================================================

app = FastAPI(
    title="AQI Dashboard API",
    description="Air Quality Index Prediction and Advisory System",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS middleware - Allow all origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================================================
# DATA MANAGER - FIXED VERSION
# ============================================================================

class DataManager:
    def __init__(self):
        self.data = self.load_data()
        self.whitelist = self.load_whitelist()
        self.models = {}
    
    def load_data(self):
        """Load data with mixed date format support"""
        try:
            if not os.path.exists(Config.DATA_PATH):
                logger.warning(f"Data file not found: {Config.DATA_PATH}")
                return pd.DataFrame(columns=['lat', 'lon', 'timestamp', 'location'])
            
            logger.info(f"Loading data from: {Config.DATA_PATH}")
            df = pd.read_csv(Config.DATA_PATH)
            
            # Handle mixed date formats
            if 'date' in df.columns:
                df['timestamp'] = pd.to_datetime(df['date'], format='mixed', errors='coerce')
            elif 'timestamp' in df.columns:
                df['timestamp'] = pd.to_datetime(df['timestamp'], format='mixed', errors='coerce')
            else:
                logger.warning("Data missing date/timestamp column, using current time")
                df['timestamp'] = pd.Timestamp.now()
            
            # Handle lng -> lon mapping
            if 'lng' in df.columns and 'lon' not in df.columns:
                df['lon'] = df['lng']
            
            # Verify required columns
            if 'lat' not in df.columns or 'lon' not in df.columns:
                logger.error("Data missing lat/lon columns")
                return pd.DataFrame(columns=['lat', 'lon', 'timestamp', 'location'])
            
            df = df.sort_values(['lat', 'lon', 'timestamp'])
            logger.info(f"✓ Loaded {len(df)} data rows")
            return df
            
        except Exception as e:
            logger.error(f"Failed to load data: {e}")
            return pd.DataFrame(columns=['lat', 'lon', 'timestamp', 'location'])
    
    def load_whitelist(self):
        """Load whitelist and augment with actual data locations"""
        try:
            whitelist = {}
            
            # Load original whitelist if exists
            if os.path.exists(Config.WHITELIST_PATH):
                try:
                    df = pd.read_csv(Config.WHITELIST_PATH)
                    for _, row in df.iterrows():
                        whitelist[row['Place']] = {
                            'region': row['Region'],
                            'lat': row['Latitude'],
                            'lon': row['Longitude'],
                            'pin': row.get('PIN Code', ''),
                            'area': row.get('Area/Locality', row['Place'])
                        }
                    logger.info(f"✓ Loaded {len(whitelist)} locations from whitelist file")
                except Exception as e:
                    logger.warning(f"Could not load whitelist file: {e}")
            
            # Add locations from actual data
            if len(self.data) > 0 and 'location' in self.data.columns:
                location_groups = self.data.groupby(['lat', 'lon']).agg({
                    'location': 'first',
                    'pincode': lambda x: x.dropna().iloc[0] if len(x.dropna()) > 0 else '',
                    'region': lambda x: x.dropna().iloc[0] if len(x.dropna()) > 0 else 'Unknown'
                }).reset_index()
                
                added = 0
                for _, row in location_groups.iterrows():
                    loc_name = row['location']
                    if pd.notna(loc_name) and str(loc_name).strip():
                        whitelist[loc_name] = {
                            'region': row['region'] if pd.notna(row['region']) else 'Central Delhi',
                            'lat': row['lat'],
                            'lon': row['lon'],
                            'pin': row['pincode'] if pd.notna(row['pincode']) else '',
                            'area': loc_name
                        }
                        added += 1
                
                logger.info(f"✓ Added {added} locations from actual data")
            
            if len(whitelist) == 0:
                logger.error("⚠️ No locations loaded! Check data files.")
            
            return whitelist
            
        except Exception as e:
            logger.error(f"Failed to load whitelist: {e}")
            return {}
    
    def get_regions(self):
        """Get unique regions from whitelist"""
        regions = set(loc['region'] for loc in self.whitelist.values())
        return sorted([r for r in regions if r and str(r).strip()])
    
    def get_locations_by_region(self, region):
        """Get locations for a specific region"""
        locations = [name for name, info in self.whitelist.items() 
                    if info['region'] == region]
        return sorted(locations)
    
    def get_location_data(self, location_name):
        """Get current and historical data for a location - WITH BETTER ERROR HANDLING"""
        if location_name not in self.whitelist:
            raise ValueError(f"Location '{location_name}' not found in whitelist")
        
        loc = self.whitelist[location_name]
        lat, lon = loc['lat'], loc['lon']
        
        logger.info(f"Searching data for {location_name} at ({lat}, {lon})")
        
        # Get data for this location with tolerance for floating point
        location_mask = (
            (self.data['lat'].round(4) == round(lat, 4)) & 
            (self.data['lon'].round(4) == round(lon, 4))
        )
        location_data = self.data[location_mask].copy()
        
        if len(location_data) == 0:
            # Try broader search with tolerance
            lat_tolerance = 0.01
            lon_tolerance = 0.01
            location_mask = (
                (self.data['lat'] >= lat - lat_tolerance) & 
                (self.data['lat'] <= lat + lat_tolerance) &
                (self.data['lon'] >= lon - lon_tolerance) & 
                (self.data['lon'] <= lon + lon_tolerance)
            )
            location_data = self.data[location_mask].copy()
            
            if len(location_data) == 0:
                logger.warning(f"No data found for {location_name} even with tolerance")
                raise ValueError(f"No data available for location: {location_name}")
        
        # Get most recent data as "current"
        location_data = location_data.sort_values('timestamp')
        current = location_data.iloc[-1:].copy()
        
        # Get historical data (last 30 days)
        cutoff_date = current['timestamp'].iloc[0] - pd.Timedelta(days=30)
        historical = location_data[location_data['timestamp'] >= cutoff_date].copy()
        
        logger.info(f"Found {len(historical)} historical records for {location_name}")
        
        return current, historical
    
    def load_model(self, pollutant: str, horizon: str = '6h'):
        """Load ML model - FIXED: OZONE naming and numpy compatibility"""
        model_key = f"{pollutant}_{horizon}"
        
        if model_key in self.models:
            return self.models[model_key]
        
        # FIXED: Handle OZONE vs Ozone naming
        # Your files use "Ozone" (capital O, then lowercase)
        file_pollutant = "Ozone" if pollutant == "OZONE" else pollutant
        model_file = Config.MODEL_PATH / f"model_artifacts_{file_pollutant}_{horizon}.pkl"
        
        if not model_file.exists():
            logger.warning(f"Model not found: {model_file}")
            raise FileNotFoundError(f"Model not found: {model_file}")
        
        try:
            with open(model_file, 'rb') as f:
                # Load with explicit encoding for numpy compatibility
                self.models[model_key] = pickle.load(f)
            logger.info(f"✓ Loaded model: {pollutant}_{horizon}")
            return self.models[model_key]
        except ModuleNotFoundError as e:
            if 'numpy._core' in str(e):
                logger.error(f"NumPy version mismatch for {pollutant}_{horizon}. Models need numpy>=2.0")
                raise Exception(f"NumPy compatibility error: {e}")
            raise
        except Exception as e:
            logger.error(f"Failed to load model {pollutant}_{horizon}: {e}")
            raise

# Initialize global data manager
data_manager = DataManager()

# ============================================================================
# PREDICTION UTILITIES
# ============================================================================

class AQICalculator:
    """AQI Calculator for Indian and US standards"""
    
    # Indian AQI breakpoints
    IN_BREAKPOINTS = {
        'PM25': [(0, 30, 'Good'), (31, 60, 'Satisfactory'), (61, 90, 'Moderate'),
                 (91, 120, 'Poor'), (121, 250, 'Very_Poor'), (251, 500, 'Severe')],
        'PM10': [(0, 50, 'Good'), (51, 100, 'Satisfactory'), (101, 250, 'Moderate'),
                 (251, 350, 'Poor'), (351, 430, 'Very_Poor'), (431, 500, 'Severe')],
        'NO2': [(0, 40, 'Good'), (41, 80, 'Satisfactory'), (81, 180, 'Moderate'),
                (181, 280, 'Poor'), (281, 400, 'Very_Poor'), (401, 500, 'Severe')],
        'OZONE': [(0, 50, 'Good'), (51, 100, 'Satisfactory'), (101, 168, 'Moderate'),
                  (169, 208, 'Poor'), (209, 748, 'Very_Poor'), (749, 1000, 'Severe')]
    }
    
    # US AQI breakpoints
    US_BREAKPOINTS = {
        'PM25': [(0, 12, 'Good'), (12.1, 35.4, 'Moderate'), (35.5, 55.4, 'Unhealthy_for_Sensitive'),
                 (55.5, 150.4, 'Unhealthy'), (150.5, 250.4, 'Very_Unhealthy'), (250.5, 500, 'Hazardous')],
        'PM10': [(0, 54, 'Good'), (55, 154, 'Moderate'), (155, 254, 'Unhealthy_for_Sensitive'),
                 (255, 354, 'Unhealthy'), (355, 424, 'Very_Unhealthy'), (425, 604, 'Hazardous')],
        'NO2': [(0, 53, 'Good'), (54, 100, 'Moderate'), (101, 360, 'Unhealthy_for_Sensitive'),
                (361, 649, 'Unhealthy'), (650, 1249, 'Very_Unhealthy'), (1250, 2049, 'Hazardous')],
        'OZONE': [(0, 54, 'Good'), (55, 70, 'Moderate'), (71, 85, 'Unhealthy_for_Sensitive'),
                  (86, 105, 'Unhealthy'), (106, 200, 'Very_Unhealthy'), (201, 604, 'Hazardous')]
    }
    
    def get_category(self, value: float, pollutant: str, standard: str = 'IN'):
        """Get AQI category for a pollutant value"""
        breakpoints = self.IN_BREAKPOINTS if standard == 'IN' else self.US_BREAKPOINTS
        
        if pollutant not in breakpoints:
            return 'Unknown'
        
        for low, high, category in breakpoints[pollutant]:
            if low <= value <= high:
                return category
        
        # If value exceeds all breakpoints
        return breakpoints[pollutant][-1][2]
    
    def calculate_aqi_range(self, value: float, pollutant: str, standard: str = 'IN'):
        """Calculate AQI range (min, max, mid) for a pollutant"""
        breakpoints = self.IN_BREAKPOINTS if standard == 'IN' else self.US_BREAKPOINTS
        
        if pollutant not in breakpoints:
            return 0, 0, 0
        
        for low, high, _ in breakpoints[pollutant]:
            if low <= value <= high:
                aqi_low = low
                aqi_high = high
                aqi_mid = (low + high) / 2
                return aqi_low, aqi_high, aqi_mid
        
        # Value exceeds breakpoints
        last = breakpoints[pollutant][-1]
        return last[0], last[1], (last[0] + last[1]) / 2
    
    def calculate_overall(self, predictions: Dict, standard: str = 'IN'):
        """Calculate overall AQI from all pollutant predictions"""
        max_aqi = 0
        dominant_pollutant = 'None'
        dominant_category = 'Good'
        avg_confidence = 0
        
        for pollutant, (category, confidence, value) in predictions.items():
            aqi_min, aqi_max, aqi_mid = self.calculate_aqi_range(value, pollutant, standard)
            
            if aqi_max > max_aqi:
                max_aqi = aqi_max
                dominant_pollutant = pollutant
                dominant_category = category
            
            avg_confidence += confidence
        
        avg_confidence = avg_confidence / len(predictions) if predictions else 0
        
        aqi_min, aqi_max, aqi_mid = 0, max_aqi, max_aqi / 2
        
        return {
            'aqi_min': round(aqi_min, 1),
            'aqi_max': round(aqi_max, 1),
            'aqi_mid': round(aqi_mid, 1),
            'category': dominant_category,
            'dominant_pollutant': dominant_pollutant,
            'confidence': round(avg_confidence, 2)
        }

aqi_calculator = AQICalculator()

def extract_weather_data(current_data):
    """Extract real weather data from current observation"""
    if len(current_data) == 0:
        return {}
    
    row = current_data.iloc[0]
    
    weather_data = {
        'temperature': float(row.get('temp', 0)) if pd.notna(row.get('temp')) else None,
        'humidity': float(row.get('humidity', 0)) if pd.notna(row.get('humidity')) else None,
        'wind_speed': float(row.get('wind_speed', 0)) if pd.notna(row.get('wind_speed')) else None,
        'wind_direction': float(row.get('wind_dir', 0)) if pd.notna(row.get('wind_dir')) else None,
        'pressure': float(row.get('pressure', 0)) if pd.notna(row.get('pressure')) else None,
        'precipitation': float(row.get('precip', 0)) if pd.notna(row.get('precip')) else None,
        'visibility': float(row.get('vis', 0)) if pd.notna(row.get('vis')) else None,
        'cloud_cover': float(row.get('clouds', 0)) if pd.notna(row.get('clouds')) else None
    }
    
    return weather_data

def predict_pollutant(pollutant: str, current_data, historical_data, horizon: str):
    """Predict a single pollutant for a specific time horizon"""
    try:
        # Map horizons to model file names
        horizon_map = {
            'current': '6h',  # Use 6h model for current
            '6h': '6h',
            '24h': '24h'
        }
        
        model_horizon = horizon_map.get(horizon, '6h')
        model = data_manager.load_model(pollutant, model_horizon)
        
        # Create features
        feature_cols = ['temp', 'humidity', 'wind_speed', 'wind_dir', 
                       'pressure', 'precip', 'vis', 'clouds']
        
        # Check if all required features exist
        missing_cols = [col for col in feature_cols if col not in current_data.columns]
        if missing_cols:
            logger.warning(f"Missing features for {pollutant}: {missing_cols}")
            return {'error': f"Missing features: {missing_cols}"}
        
        features = current_data[feature_cols].copy()
        
        # Handle any NaN values
        features = features.fillna(0)
        
        # Make prediction
        prediction = model.predict(features)[0]
        
        # Get confidence if model supports it
        try:
            probabilities = model.predict_proba(features)[0]
            confidence = float(max(probabilities))
        except:
            confidence = 0.75  # Default confidence if model doesn't support predict_proba
        
        return {
            'value': float(prediction),
            'confidence': confidence,
            'error': None
        }
    
    except FileNotFoundError as e:
        logger.error(f"Model file not found for {pollutant} ({horizon}): {e}")
        return {'error': f"Model not available: {pollutant}_{horizon}"}
    except Exception as e:
        logger.error(f"Prediction error for {pollutant} ({horizon}): {e}")
        return {'error': str(e)}

def predict_all(current_data, historical_data, standard: str = 'IN'):
    """Make predictions for all pollutants and time horizons"""
    horizons = ['current', '6h', '24h']
    pollutants = ['PM25', 'PM10', 'NO2', 'OZONE']
    
    results = {p: {} for p in pollutants}
    results['overall'] = {}
    
    # Make predictions for each horizon
    for horizon in horizons:
        for pollutant in pollutants:
            pred = predict_pollutant(pollutant, current_data, historical_data, horizon)
            
            if 'error' not in pred:
                category = aqi_calculator.get_category(pred['value'], pollutant, standard)
                aqi_min, aqi_max, aqi_mid = aqi_calculator.calculate_aqi_range(
                    pred['value'], pollutant, standard
                )
                
                results[pollutant][horizon] = {
                    'value': round(pred['value'], 2),
                    'category': category,
                    'aqi_min': round(aqi_min, 1),
                    'aqi_max': round(aqi_max, 1),
                    'aqi_mid': round(aqi_mid, 1),
                    'confidence': pred['confidence']
                }
            else:
                results[pollutant][horizon] = pred
        
        # Calculate overall AQI
        preds = {p: (results[p][horizon]['category'], results[p][horizon]['confidence'], results[p][horizon]['value'])
                for p in pollutants if 'error' not in results[p][horizon]}
        
        if preds:
            results['overall'][horizon] = aqi_calculator.calculate_overall(preds, standard)
        else:
            results['overall'][horizon] = {
                'aqi_min': 0, 'aqi_max': 0, 'aqi_mid': 0,
                'category': 'Unknown', 'dominant_pollutant': 'None', 'confidence': 0.0
            }
    
    return results

# ============================================================================
# GEMINI AI ASSISTANT
# ============================================================================

class GeminiAssistant:
    def __init__(self):
        self.enabled = False
        if Config.GEMINI_API_KEY and GEMINI_AVAILABLE:
            try:
                genai.configure(api_key=Config.GEMINI_API_KEY)
                self.model = genai.GenerativeModel('gemini-pro')
                self.enabled = True
                logger.info("✓ Gemini AI initialized")
            except Exception as e:
                logger.error(f"Gemini init failed: {e}")
    
    def get_response(self, message: str, context: Dict) -> Dict:
        user_profile = context.get('user_profile', {})
        
        if not self.enabled:
            return {
                'response': self._static_response(context, user_profile),
                'updated_profile': None,
                'source': 'static_fallback'
            }
        
        try:
            # Extract user profile information
            profile_type = user_profile.get('profile_type', '')
            profile_label = user_profile.get('profile_label', 'general public')
            
            # Build profile context
            profile_context = ""
            if profile_type:
                profile_guidance = {
                    'child': 'Children are highly sensitive to air pollution. Lungs are still developing. Extra precautions needed.',
                    'teenager': 'Teenagers are active and breathe more air. Caution during sports/outdoor activities.',
                    'elderly': 'Elderly persons have weaker immune systems and existing health conditions. High risk group.',
                    'pregnant': 'Pregnant women need special care - pollution affects both mother and baby. Avoid exposure.',
                    'asthma': 'Asthma patients are extremely sensitive to air pollution. Even low pollution can trigger attacks.',
                    'heart_condition': 'Heart patients at high risk from air pollution. Can trigger cardiac events.',
                    'respiratory': 'People with respiratory issues highly vulnerable. Avoid outdoor exposure during poor AQI.'
                }
                profile_context = f"\n\nIMPORTANT: User is {profile_label}. {profile_guidance.get(profile_type, '')} Tailor your advice specifically for this group."
            
            prompt = f"""You are an expert AQI health advisor for Delhi NCR, providing personalized air quality advice.

Current Context:
- Location: {context.get('location', 'Delhi NCR')}
- Current AQI: {context.get('aqi_data', {}).get('aqi_mid', 'N/A')} ({context.get('aqi_data', {}).get('category', 'Unknown')})
- Dominant Pollutant: {context.get('aqi_data', {}).get('dominant_pollutant', 'Unknown')}
{profile_context}

User Question: {message}

Instructions:
1. Provide personalized health advice based on the user's profile
2. Be specific about precautions for their situation
3. Include actionable recommendations
4. Keep response 4-6 sentences
5. Use a warm, caring, but professional tone

Provide your response:"""
            
            response = self.model.generate_content(prompt)
            return {
                'response': response.text,
                'updated_profile': None,
                'source': 'gemini_ai'
            }
        except Exception as e:
            logger.error(f"Gemini error: {e}")
            return {
                'response': self._static_response(context, user_profile),
                'updated_profile': None,
                'source': 'static_fallback',
                'error': str(e)
            }
    
    def _static_response(self, context, user_profile=None):
        """Fallback static response when AI unavailable"""
        category = context.get('aqi_data', {}).get('category', 'Unknown')
        
        base_responses = {
            'Good': "✅ Air quality is excellent! Safe for all outdoor activities.",
            'Satisfactory': "😊 Air quality is acceptable for most people.",
            'Moderate': "⚠️ Moderate air quality. Sensitive individuals should be cautious.",
            'Poor': "🚨 Poor air quality. Limit outdoor activities.",
            'Very_Poor': "⛔ Very poor air quality! Stay indoors if possible.",
            'Severe': "🔴 SEVERE air quality! Avoid going outside."
        }
        
        response = base_responses.get(category, "How can I help you with air quality information?")
        
        # Add profile-specific advice
        if user_profile:
            profile_type = user_profile.get('profile_type', '')
            profile_advice = {
                'child': " Children should avoid outdoor play during poor air quality.",
                'elderly': " Elderly persons should take extra precautions and stay indoors.",
                'pregnant': " Pregnant women should minimize outdoor exposure.",
                'asthma': " Asthma patients should keep rescue inhalers ready.",
                'heart_condition': " Heart patients should avoid physical exertion outdoors.",
                'respiratory': " Use air purifiers indoors if available."
            }
            
            if profile_type in profile_advice:
                response += profile_advice[profile_type]
        
        return response

# Global instance
gemini_assistant = GeminiAssistant()

# ============================================================================
# API ENDPOINTS
# ============================================================================

@app.get("/")
async def root():
    """Root endpoint - API info"""
    return {
        "message": "AQI Dashboard API",
        "version": "1.0.0",
        "status": "running",
        "numpy_version": np.__version__,
        "endpoints": {
            "docs": "/docs",
            "health": "/api/health",
            "regions": "/api/regions",
            "predict": "/api/predict",
            "chat": "/api/chat"
        }
    }

@app.get("/health")
async def health_basic():
    """Basic health check"""
    return {"status": "ok", "numpy_version": np.__version__}

@app.get("/api/health")
async def health_check():
    """Detailed health check endpoint"""
    # Check which models are available
    available_models = []
    for pollutant in ['PM25', 'PM10', 'NO2', 'OZONE']:
        for horizon in ['6h', '24h']:
            # Use correct naming for OZONE
            file_pollutant = "Ozone" if pollutant == "OZONE" else pollutant
            model_file = Config.MODEL_PATH / f"model_artifacts_{file_pollutant}_{horizon}.pkl"
            if model_file.exists():
                available_models.append(f"{pollutant}_{horizon}")
    
    return {
        "status": "healthy",
        "data_loaded": len(data_manager.data) > 0,
        "data_rows": len(data_manager.data),
        "locations": len(data_manager.whitelist),
        "regions": len(data_manager.get_regions()),
        "gemini_enabled": gemini_assistant.enabled,
        "models_path_exists": Config.MODEL_PATH.exists(),
        "available_models": available_models,
        "total_models": len(available_models),
        "numpy_version": np.__version__,
        "python_version": sys.version
    }

@app.get("/api/regions")
async def get_regions():
    """Get all available regions - Returns direct array"""
    try:
        regions = data_manager.get_regions()
        logger.info(f"Returning {len(regions)} regions")
        return regions  # Return direct array
    except Exception as e:
        logger.error(f"Error getting regions: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/locations/{region}")
async def get_locations(region: str):
    """Get locations for a specific region - Returns direct array"""
    try:
        locations = data_manager.get_locations_by_region(region)
        logger.info(f"Returning {len(locations)} locations for region: {region}")
        return locations  # Return direct array
    except Exception as e:
        logger.error(f"Error getting locations: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/predict")
async def predict_endpoint(request: PredictionRequest):
    """Get AQI predictions with real weather data for a location"""
    try:
        logger.info(f"Prediction request for: {request.location}")
        
        # Validate location
        if request.location not in data_manager.whitelist:
            available_locations = list(data_manager.whitelist.keys())[:10]
            raise HTTPException(
                status_code=404,
                detail=f"Location '{request.location}' not found. Available: {available_locations}"
            )
        
        # Get location data
        try:
            current, historical = data_manager.get_location_data(request.location)
        except ValueError as e:
            # Location exists in whitelist but has no data
            logger.warning(f"No data for location: {request.location}")
            raise HTTPException(
                status_code=404,
                detail=f"No data available for location '{request.location}'. This location may not have monitoring stations."
            )
        
        if len(current) == 0:
            raise HTTPException(
                status_code=404,
                detail=f"No current data available for location: {request.location}"
            )
        
        # Make predictions
        predictions = predict_all(current, historical, request.standard)
        
        # Extract real weather data
        weather_data = extract_weather_data(current)
        
        # Build complete response
        response = {
            "success": True,
            "location": request.location,
            "standard": request.standard,
            "timestamp": pd.Timestamp.now().isoformat(),
            "predictions": predictions,
            "current_data": weather_data
        }
        
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Prediction error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@app.post("/api/chat")
async def chat(request: ChatRequest):
    """Chat with AI assistant"""
    try:
        context = {
            'location': request.location,
            'aqi_data': request.aqi_data,
            'user_profile': request.user_profile
        }
        result = gemini_assistant.get_response(request.message, context)
        return {
            "success": True,
            **result
        }
    except Exception as e:
        logger.error(f"Chat error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# ============================================================================
# STARTUP EVENT
# ============================================================================

@app.on_event("startup")
async def startup_event():
    """Log startup information"""
    logger.info("=" * 60)
    logger.info("AQI Dashboard API Starting on Railway...")
    logger.info(f"Python version: {sys.version}")
    logger.info(f"NumPy version: {np.__version__}")
    logger.info(f"Data loaded: {len(data_manager.data)} rows")
    logger.info(f"Locations available: {len(data_manager.whitelist)}")
    logger.info(f"Regions available: {len(data_manager.get_regions())}")
    logger.info(f"Gemini AI: {'✓ Enabled' if gemini_assistant.enabled else '✗ Disabled (Set GEMINI_API_KEY env variable)'}")
    
    # Check available models
    available_models = []
    for pollutant in ['PM25', 'PM10', 'NO2', 'OZONE']:
        for horizon in ['6h', '24h']:
            # Use correct naming for OZONE
            file_pollutant = "Ozone" if pollutant == "OZONE" else pollutant
            model_file = Config.MODEL_PATH / f"model_artifacts_{file_pollutant}_{horizon}.pkl"
            if model_file.exists():
                available_models.append(f"{pollutant}_{horizon}")
    
    logger.info(f"Available models: {len(available_models)}/8")
    if len(available_models) < 8:
        missing = 8 - len(available_models)
        logger.warning(f"⚠️ Missing {missing} model files - predictions may not work for all pollutants/horizons")
    
    logger.info(f"Port: {os.environ.get('PORT', '8000')}")
    logger.info("=" * 60)

# ============================================================================
# RUN SERVER
# ============================================================================

if __name__ == "__main__":
    import uvicorn
    
    # Railway automatically provides the PORT environment variable
    port = int(os.environ.get("PORT", 8000))
    
    # Run server
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=port,
        log_level="info"
    )