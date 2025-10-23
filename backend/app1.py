"""
AQI Dashboard - FastAPI Backend (COMPLETE VERSION WITH WEATHER DATA)
Complete REST API for air quality predictions with real weather data
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

# Import Gemini AI
try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ============================================================================
# CONFIGURATION
# ============================================================================

class Config:
    BASE_DIR = Path(__file__).parent
    MODEL_PATH = BASE_DIR / "Classification_trained_models"
    DATA_PATH = BASE_DIR / "data" / "inference_data.csv"
    WHITELIST_PATH = BASE_DIR / "region_wise_popular_places_from_inference.csv"
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
    version="1.0.0"
)

# CORS middleware
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
        self.data = self.load_data()  # Load data FIRST
        self.whitelist = self.load_whitelist()  # Then whitelist
        self.models = {}
    
    def load_data(self):
        """Load data with mixed date format support"""
        try:
            if not os.path.exists(Config.DATA_PATH):
                raise FileNotFoundError(f"Data file not found: {Config.DATA_PATH}")
            
            logger.info(f"Loading data from: {Config.DATA_PATH}")
            df = pd.read_csv(Config.DATA_PATH)
            
            # Handle mixed date formats
            if 'date' in df.columns:
                df['timestamp'] = pd.to_datetime(df['date'], format='mixed', errors='coerce')
            elif 'timestamp' in df.columns:
                df['timestamp'] = pd.to_datetime(df['timestamp'], format='mixed', errors='coerce')
            else:
                raise ValueError("Data must have 'date' or 'timestamp' column")
            
            # Handle lng -> lon mapping
            if 'lng' in df.columns:
                df['lon'] = df['lng']
            
            # Verify required columns
            if 'lat' not in df.columns or 'lon' not in df.columns:
                raise ValueError("Data must have 'lat' and 'lon' columns")
            
            df = df.sort_values(['lat', 'lon', 'timestamp'])
            logger.info(f"✓ Loaded {len(df)} data rows")
            return df
            
        except Exception as e:
            logger.error(f"Failed to load data: {e}")
            return pd.DataFrame(columns=['lat', 'lon', 'timestamp'])
    
    def load_whitelist(self):
        """Load whitelist and augment with actual data locations"""
        try:
            whitelist = {}
            
            # Load original whitelist if exists
            if os.path.exists(Config.WHITELIST_PATH):
                df = pd.read_csv(Config.WHITELIST_PATH)
                for _, row in df.iterrows():
                    whitelist[row['Place']] = {
                        'region': row['Region'],
                        'lat': row['Latitude'],
                        'lon': row['Longitude'],
                        'pin': row.get('PIN Code', ''),
                        'area': row.get('Area/Locality', row['Place'])
                    }
                logger.info(f"✓ Loaded {len(whitelist)} locations from whitelist")
            
            # Add locations from actual data
            if len(self.data) > 0 and 'location' in self.data.columns:
                # Get unique locations with their coordinates
                location_groups = self.data.groupby(['lat', 'lon']).agg({
                    'location': 'first',
                    'pincode': lambda x: x.dropna().iloc[0] if len(x.dropna()) > 0 else '',
                    'region': lambda x: x.dropna().iloc[0] if len(x.dropna()) > 0 else 'Unknown'
                }).reset_index()
                
                added = 0
                for _, row in location_groups.iterrows():
                    loc_name = row['location']
                    if pd.notna(loc_name) and str(loc_name).strip():
                        # Use actual coordinates from data
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
                logger.error("No locations loaded!")
            
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
        """Get current and historical data for a location"""
        if location_name not in self.whitelist:
            raise ValueError(f"Location '{location_name}' not found in whitelist")
        
        loc = self.whitelist[location_name]
        lat, lon = loc['lat'], loc['lon']
        
        logger.info(f"Searching data for {location_name} at ({lat}, {lon})")
        
        # Use exact coordinates from whitelist (which now matches data)
        mask = ((np.abs(self.data['lat'] - lat) < 0.0001) & 
               (np.abs(self.data['lon'] - lon) < 0.0001))
        loc_data = self.data[mask].copy()
        
        # If no exact match, expand search
        if len(loc_data) == 0:
            logger.warning(f"No exact match, expanding search radius")
            mask = ((np.abs(self.data['lat'] - lat) < 0.01) & 
                   (np.abs(self.data['lon'] - lon) < 0.01))
            loc_data = self.data[mask].copy()
        
        if len(loc_data) == 0:
            # Even wider search
            mask = ((np.abs(self.data['lat'] - lat) < 0.05) & 
                   (np.abs(self.data['lon'] - lon) < 0.05))
            loc_data = self.data[mask].copy()
        
        if len(loc_data) == 0:
            raise ValueError(f"No data found for {location_name} at ({lat}, {lon})")
        
        logger.info(f"✓ Found {len(loc_data)} data rows for {location_name}")
        
        loc_data = loc_data.sort_values('timestamp')
        
        # Return most recent row as current, last 96 as historical
        return loc_data.iloc[[-1]].copy(), loc_data.tail(96).copy()
    
    def load_model(self, pollutant: str, horizon: str):
        """Load ML model for specific pollutant and horizon"""
        cache_key = f"{pollutant}_{horizon}"
        if cache_key in self.models:
            return self.models[cache_key]
        
        model_file = Config.MODEL_PATH / f"model_artifacts_{pollutant}_{horizon}.pkl"
        if not model_file.exists():
            raise FileNotFoundError(f"Model not found: {model_file}")
        
        with open(model_file, 'rb') as f:
            model_artifact = pickle.load(f)
        
        self.models[cache_key] = model_artifact
        return model_artifact

# Initialize data manager
data_manager = DataManager()

# ============================================================================
# FEATURE ENGINEERING
# ============================================================================

class SimpleFeatureEngineer:
    WEATHER_FEATURES = [
        'temperature', 'humidity', 'dewPoint', 'apparentTemperature',
        'precipIntensity', 'pressure', 'surfacePressure',
        'cloudCover', 'windSpeed', 'windBearing', 'windGust'
    ]
    
    def engineer_features(self, current_data: pd.DataFrame, historical_data: pd.DataFrame,
                         pollutant: str, horizon: str) -> pd.DataFrame:
        features = current_data.copy()
        
        # Add lag features
        lag_map = {'1h': [1, 2, 3], '6h': [6, 12], '12h': [12, 24], '24h': [24, 48]}
        for lag in lag_map.get(horizon, [1]):
            if len(historical_data) >= lag and pollutant in historical_data.columns:
                features.loc[features.index[0], f'{pollutant}_lag_{lag}h'] = historical_data[pollutant].iloc[-lag]
        
        # Select numeric features only
        exclude = {'location', 'timestamp', 'date', 'lat', 'lng', 'lon', 'region',
                  'PM25', 'PM10', 'NO2', 'OZONE', 'CO', 'SO2', 'AQI', 'pincode', 'loc_key', 'loc_id'}
        numeric_cols = [c for c in features.columns 
                       if c not in exclude and pd.api.types.is_numeric_dtype(features[c])]
        
        return features[numeric_cols].fillna(0).astype(np.float32)
    
    def align_features(self, features: pd.DataFrame, model_features: List[str]) -> pd.DataFrame:
        return features.reindex(columns=model_features, fill_value=0.0).astype(np.float32)

feature_engineer = SimpleFeatureEngineer()

# ============================================================================
# AQI CALCULATION
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

US_AQI_INDEX = {
    'Good': (0, 50), 'Moderate': (51, 100), 'Unhealthy_for_Sensitive': (101, 150),
    'Unhealthy': (151, 200), 'Very_Unhealthy': (201, 300), 'Hazardous': (301, 500)
}

CATEGORY_MAPPING = {
    'Good': {'us': 'Good'}, 'Satisfactory': {'us': 'Moderate'},
    'Moderate': {'us': 'Unhealthy_for_Sensitive'}, 'Poor': {'us': 'Unhealthy'},
    'Very_Poor': {'us': 'Very_Unhealthy'}, 'Severe': {'us': 'Hazardous'}
}

class AQICalculator:
    def calculate_sub_index(self, pollutant: str, category: str, confidence: float, standard: str):
        normalized = category.replace(' ', '_')
        if standard == 'IN':
            aqi_min, aqi_max = INDIAN_AQI_INDEX.get(normalized, (0, 0))
        else:
            us_cat = CATEGORY_MAPPING.get(normalized, {}).get('us', 'Hazardous')
            aqi_min, aqi_max = US_AQI_INDEX.get(us_cat, (0, 0))
        
        conc_min, conc_max = INDIAN_AQI_BREAKPOINTS.get(pollutant, {}).get(normalized, (0, 0))
        
        return {
            'pollutant': pollutant, 'category': category,
            'aqi_min': aqi_min, 'aqi_max': aqi_max,
            'aqi_mid': (aqi_min + aqi_max) / 2,
            'concentration_range': (conc_min, conc_max),
            'confidence': confidence
        }
    
    def calculate_overall(self, predictions: Dict, standard: str):
        sub_indices = []
        for pollutant, (category, confidence) in predictions.items():
            if pollutant in ['PM25', 'PM10', 'NO2', 'OZONE']:
                sub_idx = self.calculate_sub_index(pollutant, category, confidence, standard)
                sub_indices.append(sub_idx)
        
        if not sub_indices:
            return {'error': 'No valid predictions'}
        
        max_idx = max(sub_indices, key=lambda x: x['aqi_mid'])
        return {
            'aqi_min': max_idx['aqi_min'], 'aqi_max': max_idx['aqi_max'],
            'aqi_mid': max_idx['aqi_mid'], 'category': max_idx['category'],
            'dominant_pollutant': max_idx['pollutant'], 'confidence': max_idx['confidence']
        }

aqi_calculator = AQICalculator()

# ============================================================================
# WEATHER DATA EXTRACTOR
# ============================================================================

def extract_weather_data(current_data: pd.DataFrame) -> Dict:
    """
    Extract weather parameters from current data
    Returns: Dictionary with temperature, humidity, windSpeed
    """
    weather = {}
    
    # List of weather parameters to extract
    weather_params = {
        'temperature': 0.0,
        'humidity': 0.0,
        'windSpeed': 0.0
    }
    
    for param, default in weather_params.items():
        if param in current_data.columns:
            value = current_data[param].iloc[0]
            # Handle NaN values
            if pd.notna(value):
                weather[param] = float(value)
            else:
                weather[param] = default
                logger.warning(f"Missing {param}, using default: {default}")
        else:
            weather[param] = default
            logger.warning(f"{param} not in data columns, using default: {default}")
    
    logger.info(f"✓ Extracted weather: Temp={weather['temperature']}°C, Humidity={weather['humidity']}%, Wind={weather['windSpeed']} km/h")
    
    return weather

# ============================================================================
# PREDICTION ENGINE
# ============================================================================

def predict_single(current_data: pd.DataFrame, historical_data: pd.DataFrame,
                  pollutant: str, horizon: str):
    model = data_manager.load_model(pollutant, horizon)
    features = feature_engineer.engineer_features(current_data, historical_data, pollutant, horizon)
    aligned = feature_engineer.align_features(features, model['feature_names'])
    probs = model['calibrated_model'].predict_proba(aligned)[0]
    pred_idx = np.argmax(probs)
    return model['classes'][pred_idx], probs[pred_idx]

def predict_all(current_data: pd.DataFrame, historical_data: pd.DataFrame, standard: str = 'IN'):
    results = {}
    
    logger.info(f"Current data shape: {current_data.shape}, Historical: {historical_data.shape}")
    logger.info(f"Available columns: {list(current_data.columns)}")
    
    for pollutant in ['PM25', 'PM10', 'NO2', 'OZONE']:
        results[pollutant] = {}
        
        # Check if pollutant data exists
        if pollutant not in historical_data.columns:
            logger.warning(f"{pollutant} not in data columns")
            for horizon in ['1h', '6h', '12h', '24h']:
                results[pollutant][horizon] = {
                    'category': 'Unknown', 'confidence': 0.0,
                    'aqi_min': 0, 'aqi_max': 0, 'aqi_mid': 0,
                    'concentration_range': (0, 0), 'error': f'{pollutant} data not available'
                }
            continue
        
        for horizon in ['1h', '6h', '12h', '24h']:
            try:
                category, confidence = predict_single(current_data, historical_data, pollutant, horizon)
                sub_idx = aqi_calculator.calculate_sub_index(pollutant, category, confidence, standard)
                results[pollutant][horizon] = {
                    'category': category, 'confidence': confidence,
                    'aqi_min': sub_idx['aqi_min'], 'aqi_max': sub_idx['aqi_max'],
                    'aqi_mid': sub_idx['aqi_mid'],
                    'concentration_range': sub_idx['concentration_range']
                }
                logger.info(f"✓ {pollutant} {horizon}: {category} ({confidence:.2%})")
            except Exception as e:
                logger.error(f"Failed {pollutant} {horizon}: {e}")
                results[pollutant][horizon] = {
                    'category': 'Unknown', 'confidence': 0.0,
                    'aqi_min': 0, 'aqi_max': 0, 'aqi_mid': 0,
                    'concentration_range': (0, 0), 'error': str(e)
                }
    
    # Calculate overall AQI
    results['overall'] = {}
    for horizon in ['1h', '6h', '12h', '24h']:
        preds = {p: (results[p][horizon]['category'], results[p][horizon]['confidence'])
                for p in ['PM25', 'PM10', 'NO2', 'OZONE'] if 'error' not in results[p][horizon]}
        
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
                'updated_profile': None
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
                'updated_profile': None
            }
        except Exception as e:
            logger.error(f"Gemini error: {e}")
            return {
                'response': self._static_response(context, user_profile),
                'updated_profile': None
            }
    
    def _static_response(self, context, user_profile=None):
        category = context.get('aqi_data', {}).get('category', 'Unknown')
        
        # Base responses by AQI category
        base_responses = {
            'Good': "✅ Air quality is excellent! Safe for all outdoor activities.",
            'Satisfactory': "😊 Air quality is acceptable for most people.",
            'Moderate': "⚠️ Moderate air quality. Sensitive individuals should be cautious.",
            'Poor': "🚨 Poor air quality. Limit outdoor activities.",
            'Very_Poor': "⛔ Very poor air quality! Stay indoors.",
            'Severe': "🔴 SEVERE air quality! Do not go outside."
        }
        
        response = base_responses.get(category, "How can I help you with air quality information?")
        
        # Add profile-specific advice
        if user_profile:
            profile_type = user_profile.get('profile_type', '')
            profile_advice = {
                'child': " Children should avoid outdoor play during poor air quality.",
                'elderly': " Elderly persons should take extra precautions and stay indoors.",
                'pregnant': " Pregnant women should minimize outdoor exposure to protect the baby.",
                'asthma': " Asthma patients should keep rescue inhalers ready and avoid triggers.",
                'heart_condition': " Heart patients should avoid physical exertion outdoors.",
                'respiratory': " Those with respiratory issues should use air purifiers indoors."
            }
            
            if profile_type in profile_advice:
                response += profile_advice[profile_type]
        
        return response

gemini_assistant = GeminiAssistant()

# ============================================================================
# API ENDPOINTS
# ============================================================================

@app.get("/")
async def root():
    return {
        "message": "AQI Dashboard API",
        "version": "1.0.0",
        "status": "running"
    }

@app.get("/api/regions")
async def get_regions():
    """Get all available regions"""
    try:
        return data_manager.get_regions()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/locations/{region}")
async def get_locations(region: str):
    """Get locations for a specific region"""
    try:
        return data_manager.get_locations_by_region(region)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/predict")
async def predict(request: PredictionRequest):
    """Get AQI predictions with real weather data for a location"""
    try:
        logger.info(f"Prediction request for: {request.location}")
        
        # Get location data
        current, historical = data_manager.get_location_data(request.location)
        
        # Make predictions
        predictions = predict_all(current, historical, request.standard)
        
        # ✨ Extract real weather data from current_data
        weather_data = extract_weather_data(current)
        
        # Build complete response
        response = {
            **predictions,  # All pollutant predictions and overall AQI
            'current_data': weather_data,  # ✨ Real weather data
            'location': request.location,
            'standard': request.standard,
            'timestamp': pd.Timestamp.now().isoformat()
        }
        
        return response
        
    except Exception as e:
        logger.error(f"Prediction error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

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
        return result
    except Exception as e:
        logger.error(f"Chat error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "data_loaded": len(data_manager.data) > 0,
        "locations": len(data_manager.whitelist),
        "gemini_enabled": gemini_assistant.enabled
    }

# ============================================================================
# RUN SERVER
# ============================================================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)