import React, { useState, useEffect } from 'react';
import Navbar from './components/Navbar';
import AudioUploader from './components/AudioUploader';
import AudioRecorder from './components/AudioRecorder';
import SampleAudios from './components/SampleAudios';
import PredictionCard from './components/PredictionCard';
import EmotionBarChart from './components/EmotionBarChart';
import TrajectoryViewer from './components/TrajectoryViewer';
import BenchmarkView from './components/BenchmarkView';
import ArchitectureView from './components/ArchitectureView';
import { Sparkles, AlertCircle } from 'lucide-react';

export default function App() {
  const [activeTab, setActiveTab] = useState('detector');
  const [predictionResult, setPredictionResult] = useState(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState(null);
  const [modelInfo, setModelInfo] = useState(null);
  const [backendStatus, setBackendStatus] = useState('connecting');

  useEffect(() => {
    // Health check & model metadata polling
    fetch('http://localhost:8000/api/model-info')
      .then((res) => res.json())
      .then((data) => {
        setModelInfo(data);
        setBackendStatus('connected');
      })
      .catch(() => {
        setBackendStatus('offline');
      });
  }, []);

  const handleAnalyzeAudio = async (audioFile) => {
    setIsLoading(true);
    setError(null);

    const formData = new FormData();
    formData.append('file', audioFile);

    try {
      const response = await fetch('http://localhost:8000/api/predict', {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        const errData = await response.json();
        throw new Error(errData.detail || 'Prediction failed');
      }

      const data = await response.json();
      setPredictionResult(data);
    } catch (err) {
      setError(err.message || 'Failed to connect to backend inference service.');
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="min-h-screen bg-navy-950 text-slate-100 flex flex-col font-sans selection:bg-blue-600 selection:text-white">
      {/* Navigation */}
      <Navbar
        activeTab={activeTab}
        setActiveTab={setActiveTab}
        backendStatus={backendStatus}
        modelInfo={modelInfo}
      />

      {/* Main Container */}
      <main className="flex-1 max-w-7xl w-full mx-auto px-4 sm:px-6 lg:px-8 py-8">
        {/* Error Notification */}
        {error && (
          <div className="mb-6 p-4 rounded-xl bg-red-500/10 border border-red-500/30 text-red-400 text-xs flex items-center gap-3">
            <AlertCircle className="w-5 h-5 shrink-0" />
            <span>{error}</span>
          </div>
        )}

        {/* Tab 1: Live Emotion Detector */}
        {activeTab === 'detector' && (
          <div className="space-y-6">
            {/* Input Cards: Upload & Recorder */}
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              <AudioUploader
                onAnalyze={handleAnalyzeAudio}
                isLoading={isLoading}
              />
              <AudioRecorder
                onAnalyze={handleAnalyzeAudio}
                isLoading={isLoading}
              />
            </div>

            {/* Instant Sample Audios */}
            <SampleAudios
              onSelectSample={handleAnalyzeAudio}
              isLoading={isLoading}
            />

            {/* Results Section */}
            {predictionResult ? (
              <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 pt-2">
                <PredictionCard result={predictionResult} />
                <EmotionBarChart probabilities={predictionResult.probabilities} />
              </div>
            ) : (
              <div className="bg-navy-900/40 border border-navy-800/60 rounded-2xl p-10 text-center">
                <div className="w-12 h-12 rounded-xl bg-blue-500/10 text-blue-400 flex items-center justify-center mx-auto mb-3">
                  <Sparkles className="w-6 h-6" />
                </div>
                <h4 className="text-sm font-semibold text-slate-300">Ready for Speech Emotion Inference</h4>
                <p className="text-xs text-slate-500 mt-1 max-w-md mx-auto">
                  Upload an audio clip, record with your microphone, or click any of the 8 RAVDESS benchmark samples above to view deep learning emotion probabilities.
                </p>
              </div>
            )}
          </div>
        )}

        {/* Tab 2: Temporal Trajectory Analysis */}
        {activeTab === 'trajectory' && <TrajectoryViewer />}

        {/* Tab 3: Model Benchmark & EDA */}
        {activeTab === 'benchmark' && <BenchmarkView />}

        {/* Tab 4: Architecture & Technical Specs */}
        {activeTab === 'architecture' && <ArchitectureView />}
      </main>

      {/* Footer */}
      <footer className="border-t border-navy-800/80 bg-navy-950 py-6 text-center text-xs text-slate-500 font-mono">
        <div className="max-w-7xl mx-auto px-4 flex flex-col sm:flex-row items-center justify-between gap-3">
          <div>VocalSense AI — Deep Learning Based Human Emotion Recognition from Speech</div>
          <div className="flex items-center gap-4 text-[11px]">
            <span>FastAPI Backend :8000</span>
            <span>•</span>
            <span>Vite Frontend :5173</span>
            <span>•</span>
            <span>RAVDESS Speech Benchmark</span>
          </div>
        </div>
      </footer>
    </div>
  );
}
