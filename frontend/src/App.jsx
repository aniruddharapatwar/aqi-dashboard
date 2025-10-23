import React, { useState, useEffect } from 'react';
import { Wind, Cloud, Activity, MessageSquare, MapPin, Clock, TrendingUp, AlertTriangle, Leaf } from 'lucide-react';

// API Configuration
const API_BASE_URL = 'http://localhost:8000/api';

const AQIDashboard = () => {
  const [regions, setRegions] = useState([]);
  const [locations, setLocations] = useState([]);
  const [selectedRegion, setSelectedRegion] = useState('');
  const [selectedLocation, setSelectedLocation] = useState('');
  const [horizon, setHorizon] = useState('1h');
  const [standard, setStandard] = useState('IN');
  const [predictions, setPredictions] = useState(null);
  const [loading, setLoading] = useState(false);
  const [showAQIInfo, setShowAQIInfo] = useState(false);
  const [showAdvisory, setShowAdvisory] = useState(false);
  const [messages, setMessages] = useState([
    { role: 'assistant', content: '👋 Hello! I\'m your AI air quality assistant. Ask me about current air quality, health advice, or pollutant risks!' }
  ]);
  const [userInput, setUserInput] = useState('');
  const [userProfile, setUserProfile] = useState({
    age_group: null,
    health_condition: null,
    activity_level: null
  });

  // Fetch regions on mount
  useEffect(() => {
    fetchRegions();
  }, []);

  // Fetch locations when region changes
  useEffect(() => {
    if (selectedRegion) {
      fetchLocations(selectedRegion);
    }
  }, [selectedRegion]);

  // Fetch predictions when location or settings change
  useEffect(() => {
    if (selectedLocation) {
      fetchPredictions();
    }
  }, [selectedLocation, horizon, standard]);

  const fetchRegions = async () => {
    try {
      const response = await fetch(`${API_BASE_URL}/regions`);
      const data = await response.json();
      setRegions(data);
      if (data.length > 0) setSelectedRegion(data[0]);
    } catch (error) {
      console.error('Error fetching regions:', error);
    }
  };

  const fetchLocations = async (region) => {
    try {
      const response = await fetch(`${API_BASE_URL}/locations/${region}`);
      const data = await response.json();
      setLocations(data);
    } catch (error) {
      console.error('Error fetching locations:', error);
    }
  };

  const fetchPredictions = async () => {
    setLoading(true);
    try {
      const response = await fetch(`${API_BASE_URL}/predict`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          location: selectedLocation,
          standard: standard
        })
      });
      const data = await response.json();
      setPredictions(data);
    } catch (error) {
      console.error('Error fetching predictions:', error);
    } finally {
      setLoading(false);
    }
  };

  const sendMessage = async () => {
    if (!userInput.trim()) return;

    const newMessages = [...messages, { role: 'user', content: userInput }];
    setMessages(newMessages);
    setUserInput('');

    try {
      const response = await fetch(`${API_BASE_URL}/chat`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          message: userInput,
          location: selectedLocation,
          aqi_data: predictions?.overall?.[horizon],
          user_profile: userProfile
        })
      });
      const data = await response.json();
      setMessages([...newMessages, { role: 'assistant', content: data.response }]);
      if (data.updated_profile) {
        setUserProfile(data.updated_profile);
      }
    } catch (error) {
      console.error('Error sending message:', error);
      setMessages([...newMessages, { role: 'assistant', content: 'Sorry, I encountered an error. Please try again.' }]);
    }
  };

  const getAQIColor = (category) => {
    const colors = {
      'Good': '#00E400',
      'Satisfactory': '#FFFF00',
      'Moderate': '#FF7E00',
      'Poor': '#FF0000',
      'Very_Poor': '#8F3F97',
      'Severe': '#7E0023',
      'Unhealthy_for_Sensitive': '#FF7E00',
      'Unhealthy': '#FF0000',
      'Very_Unhealthy': '#8F3F97',
      'Hazardous': '#7E0023'
    };
    return colors[category] || '#cbd5e1';
  };

  const getAQIGradient = (category) => {
    const gradients = {
      'Good': 'linear-gradient(135deg, #e0f2fe 0%, #bae6fd 50%, #7dd3fc 100%)',
      'Satisfactory': 'linear-gradient(135deg, #fef3c7 0%, #fde68a 50%, #fcd34d 100%)',
      'Moderate': 'linear-gradient(135deg, #fed7aa 0%, #fdba74 50%, #fb923c 100%)',
      'Poor': 'linear-gradient(135deg, #fecaca 0%, #fca5a5 50%, #f87171 100%)',
      'Very_Poor': 'linear-gradient(135deg, #e9d5ff 0%, #d8b4fe 50%, #c084fc 100%)',
      'Severe': 'linear-gradient(135deg, #7f1d1d 0%, #991b1b 50%, #b91c1c 100%)'
    };
    return gradients[category] || gradients['Good'];
  };

  const renderPollutantCard = (pollutant, displayName) => {
    if (!predictions) return null;
    
    const data = predictions[pollutant]?.[horizon];
    if (!data) return null;

    const color = getAQIColor(data.category);

    return (
      <div className="bg-white rounded-2xl p-5 shadow-lg border-2 border-gray-200 hover:shadow-xl transition-all duration-200 hover:-translate-y-1">
        <div className="text-gray-600 text-sm font-bold uppercase tracking-wide mb-2">
          {displayName}
        </div>
        <div className="text-3xl font-black mb-2" style={{ color }}>
          {data.category.replace('_', ' ')}
        </div>
        <div className="text-gray-600 text-sm mb-1">
          AQI: {Math.round(data.aqi_min)} - {Math.round(data.aqi_max)}
        </div>
        <div className="text-gray-400 text-xs">
          {data.concentration_range[0].toFixed(0)} - {data.concentration_range[1].toFixed(0)} µg/m³
        </div>
        <div className="text-gray-500 text-xs mt-2">
          {Math.round(data.confidence * 100)}% confidence
        </div>
      </div>
    );
  };

  if (!predictions && selectedLocation) {
    return (
      <div className="min-h-screen bg-gradient-to-br from-blue-50 via-purple-50 to-pink-50 flex items-center justify-center">
        <div className="text-center">
          <div className="animate-spin rounded-full h-16 w-16 border-b-4 border-blue-600 mx-auto mb-4"></div>
          <p className="text-gray-600 text-lg">Loading predictions...</p>
        </div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 via-purple-50 to-pink-50">
      {/* Header */}
      <div className="bg-white border-b-2 border-gray-200 shadow-lg">
        <div className="max-w-7xl mx-auto px-6 py-5">
          <div className="flex items-center gap-3 mb-2">
            <Wind className="w-10 h-10 text-blue-600" />
            <h1 className="text-4xl font-black bg-gradient-to-r from-blue-600 to-purple-600 bg-clip-text text-transparent">
              AQI Dashboard — Delhi NCR
            </h1>
          </div>
          <p className="text-gray-600 text-lg">
            AI-Powered Air Quality Monitoring & Advisory System
          </p>
        </div>
      </div>

      <div className="max-w-7xl mx-auto px-6 py-8">
        {/* Controls */}
        <div className="bg-white rounded-2xl p-6 shadow-lg mb-8 border-2 border-gray-200">
          <div className="grid grid-cols-1 md:grid-cols-4 gap-6">
            {/* Region */}
            <div>
              <label className="block text-sm font-bold text-gray-700 mb-2">
                <MapPin className="w-4 h-4 inline mr-1" />
                Region
              </label>
              <select
                value={selectedRegion}
                onChange={(e) => setSelectedRegion(e.target.value)}
                className="w-full px-4 py-2 border-2 border-gray-300 rounded-lg focus:border-blue-500 focus:ring-2 focus:ring-blue-200 transition-all"
              >
                {regions.map(region => (
                  <option key={region} value={region}>{region}</option>
                ))}
              </select>
            </div>

            {/* Location */}
            <div>
              <label className="block text-sm font-bold text-gray-700 mb-2">
                <MapPin className="w-4 h-4 inline mr-1" />
                Location
              </label>
              <select
                value={selectedLocation}
                onChange={(e) => setSelectedLocation(e.target.value)}
                className="w-full px-4 py-2 border-2 border-gray-300 rounded-lg focus:border-blue-500 focus:ring-2 focus:ring-blue-200 transition-all"
              >
                <option value="">-- Choose --</option>
                {locations.map(loc => (
                  <option key={loc} value={loc}>{loc}</option>
                ))}
              </select>
            </div>

            {/* Horizon */}
            <div>
              <label className="block text-sm font-bold text-gray-700 mb-2">
                <Clock className="w-4 h-4 inline mr-1" />
                Forecast
              </label>
              <div className="grid grid-cols-4 gap-2">
                {['1h', '6h', '12h', '24h'].map(h => (
                  <button
                    key={h}
                    onClick={() => setHorizon(h)}
                    className={`px-3 py-2 rounded-lg font-semibold text-sm transition-all ${
                      horizon === h
                        ? 'bg-blue-600 text-white shadow-lg'
                        : 'bg-gray-100 text-gray-700 hover:bg-gray-200'
                    }`}
                  >
                    {h}
                  </button>
                ))}
              </div>
            </div>

            {/* Standard */}
            <div>
              <label className="block text-sm font-bold text-gray-700 mb-2">
                <Activity className="w-4 h-4 inline mr-1" />
                Standard
              </label>
              <div className="grid grid-cols-2 gap-2">
                {[
                  { value: 'IN', label: '🇮🇳 NAQI' },
                  { value: 'US', label: '🇺🇸 EPA' }
                ].map(s => (
                  <button
                    key={s.value}
                    onClick={() => setStandard(s.value)}
                    className={`px-3 py-2 rounded-lg font-semibold text-sm transition-all ${
                      standard === s.value
                        ? 'bg-blue-600 text-white shadow-lg'
                        : 'bg-gray-100 text-gray-700 hover:bg-gray-200'
                    }`}
                  >
                    {s.label}
                  </button>
                ))}
              </div>
            </div>
          </div>
        </div>

        {selectedLocation && predictions && (
          <>
            {/* Main AQI Display */}
            <div className="grid grid-cols-1 lg:grid-cols-3 gap-8 mb-8">
              {/* Large AQI Card */}
              <div
                className="rounded-3xl p-10 shadow-2xl text-center border-2 border-gray-300 relative overflow-hidden"
                style={{ background: getAQIGradient(predictions.overall[horizon]?.category) }}
              >
                <div className="relative z-10">
                  <div className="text-gray-700 text-sm font-semibold mb-2 tracking-wide">
                    AQI • {selectedLocation} • {horizon} Forecast
                  </div>
                  <div
                    className="text-7xl font-black mb-4"
                    style={{ color: getAQIColor(predictions.overall[horizon]?.category) }}
                  >
                    {Math.round(predictions.overall[horizon]?.aqi_min)} — {Math.round(predictions.overall[horizon]?.aqi_max)}
                  </div>
                  <div
                    className="text-3xl font-bold mb-4"
                    style={{ color: getAQIColor(predictions.overall[horizon]?.category) }}
                  >
                    {predictions.overall[horizon]?.category?.replace('_', ' ')}
                  </div>
                  <div className="text-gray-700 text-sm font-semibold">
                    Dominant: {predictions.overall[horizon]?.dominant_pollutant}<br />
                    Confidence: {Math.round(predictions.overall[horizon]?.confidence * 100)}%
                  </div>
                </div>
              </div>

              {/* Pollutant Cards */}
              <div className="lg:col-span-2 grid grid-cols-2 gap-6">
                {renderPollutantCard('PM25', 'PM2.5')}
                {renderPollutantCard('PM10', 'PM10')}
                {renderPollutantCard('NO2', 'NO₂')}
                {renderPollutantCard('OZONE', 'O₃')}
              </div>
            </div>

            {/* AQI Info Toggle */}
            <div className="mb-8">
              <button
                onClick={() => setShowAQIInfo(!showAQIInfo)}
                className="w-full md:w-auto px-6 py-3 bg-gradient-to-r from-blue-600 to-purple-600 text-white font-bold rounded-xl shadow-lg hover:shadow-xl transition-all hover:-translate-y-1"
              >
                📚 {showAQIInfo ? 'Hide' : 'Learn About'} AQI Information
              </button>
            </div>

            {showAQIInfo && (
              <div className="bg-white rounded-2xl p-8 shadow-lg mb-8 border-2 border-gray-200">
                <h2 className="text-3xl font-bold text-gray-800 mb-6">📚 Understanding Air Quality Index (AQI)</h2>
                <p className="text-gray-600 mb-6 leading-relaxed">
                  The Air Quality Index (AQI) tells you how clean or polluted your air is, and what health effects might be a concern.
                </p>
                
                <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-6">
                  <div className="bg-green-50 border-l-4 border-green-500 p-4 rounded">
                    <h3 className="font-bold text-green-800 mb-2">🟢 Good (0-50)</h3>
                    <p className="text-sm text-gray-700">Air quality is excellent. Perfect for outdoor activities!</p>
                  </div>
                  <div className="bg-yellow-50 border-l-4 border-yellow-500 p-4 rounded">
                    <h3 className="font-bold text-yellow-800 mb-2">🟡 Satisfactory (51-100)</h3>
                    <p className="text-sm text-gray-700">Generally acceptable. Sensitive people should be cautious.</p>
                  </div>
                  <div className="bg-orange-50 border-l-4 border-orange-500 p-4 rounded">
                    <h3 className="font-bold text-orange-800 mb-2">🟠 Moderate (101-200)</h3>
                    <p className="text-sm text-gray-700">Sensitive groups may experience health effects.</p>
                  </div>
                </div>

                <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                  <div className="bg-red-50 border-l-4 border-red-500 p-4 rounded">
                    <h3 className="font-bold text-red-800 mb-2">🔴 Poor (201-300)</h3>
                    <p className="text-sm text-gray-700">Everyone may begin to experience health effects.</p>
                  </div>
                  <div className="bg-purple-50 border-l-4 border-purple-500 p-4 rounded">
                    <h3 className="font-bold text-purple-800 mb-2">🟣 Very Poor (301-400)</h3>
                    <p className="text-sm text-gray-700">Health alert! Everyone may experience serious effects.</p>
                  </div>
                  <div className="bg-gray-900 border-l-4 border-gray-900 p-4 rounded">
                    <h3 className="font-bold text-white mb-2">⚫ Severe (401-500)</h3>
                    <p className="text-sm text-gray-200">Emergency conditions. Entire population affected.</p>
                  </div>
                </div>
              </div>
            )}

            {/* AI Chat Assistant */}
            <div className="bg-white rounded-2xl shadow-lg p-6 mb-8 border-2 border-gray-200">
              <div className="bg-gradient-to-r from-blue-600 via-purple-600 to-pink-600 rounded-xl p-5 mb-6 text-white">
                <div className="flex items-center gap-3">
                  <MessageSquare className="w-8 h-8" />
                  <div>
                    <h2 className="text-xl font-bold">AI Assistant</h2>
                    <p className="text-sm opacity-95">Get personalized air quality advice & health recommendations</p>
                  </div>
                </div>
              </div>

              {/* Chat Messages */}
              <div className="bg-gray-50 rounded-xl p-4 mb-4 max-h-96 overflow-y-auto space-y-3">
                {messages.map((msg, idx) => (
                  <div
                    key={idx}
                    className={`p-3 rounded-lg ${
                      msg.role === 'user'
                        ? 'bg-blue-100 ml-8'
                        : 'bg-white border border-gray-200 mr-8'
                    }`}
                  >
                    <div className="text-xs font-semibold text-gray-500 mb-1">
                      {msg.role === 'user' ? 'You' : '🤖 AI Assistant'}
                    </div>
                    <div className="text-sm text-gray-800 whitespace-pre-wrap">
                      {msg.content}
                    </div>
                  </div>
                ))}
              </div>

              {/* Chat Input */}
              <div className="flex gap-2">
                <input
                  type="text"
                  value={userInput}
                  onChange={(e) => setUserInput(e.target.value)}
                  onKeyPress={(e) => e.key === 'Enter' && sendMessage()}
                  placeholder="Ask about air quality..."
                  className="flex-1 px-4 py-3 border-2 border-gray-300 rounded-lg focus:border-blue-500 focus:ring-2 focus:ring-blue-200 transition-all"
                />
                <button
                  onClick={sendMessage}
                  className="px-6 py-3 bg-gradient-to-r from-blue-600 to-purple-600 text-white font-bold rounded-lg shadow-lg hover:shadow-xl transition-all hover:-translate-y-1"
                >
                  Send
                </button>
              </div>
            </div>

            {/* Advisory Button */}
            <button
              onClick={() => setShowAdvisory(!showAdvisory)}
              className="w-full px-8 py-4 bg-gradient-to-r from-blue-600 to-purple-600 text-white text-lg font-bold rounded-xl shadow-lg hover:shadow-xl transition-all hover:-translate-y-1 mb-8"
            >
              📋 {showAdvisory ? 'Hide' : 'Get'} Health & Policy Advisory
            </button>
          </>
        )}

        {!selectedLocation && (
          <div className="text-center py-20">
            <Leaf className="w-24 h-24 text-gray-400 mx-auto mb-6" />
            <h2 className="text-3xl font-bold text-gray-700 mb-4">Welcome to AQI Dashboard</h2>
            <p className="text-gray-600 text-lg">Select a region and location to get started</p>
          </div>
        )}
      </div>
    </div>
  );
};

export default AQIDashboard;