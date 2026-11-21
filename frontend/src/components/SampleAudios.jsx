import React, { useState, useEffect } from 'react';
import { Play, Pause, Zap, Disc3, Sparkles } from 'lucide-react';

const EMOTION_THEMES = {
  neutral: { color: '#94a3b8', bg: 'rgba(148, 163, 184, 0.1)', emoji: '😐' },
  calm: { color: '#38bdf8', bg: 'rgba(56, 189, 248, 0.1)', emoji: '😌' },
  happy: { color: '#facc15', bg: 'rgba(250, 204, 21, 0.1)', emoji: '😄' },
  sad: { color: '#60a5fa', bg: 'rgba(96, 165, 250, 0.1)', emoji: '😢' },
  angry: { color: '#ef4444', bg: 'rgba(239, 68, 68, 0.1)', emoji: '😠' },
  fearful: { color: '#a855f7', bg: 'rgba(168, 85, 247, 0.1)', emoji: '😨' },
  disgust: { color: '#10b981', bg: 'rgba(16, 185, 129, 0.1)', emoji: '🤢' },
  surprised: { color: '#f97316', bg: 'rgba(249, 115, 22, 0.1)', emoji: '😲' },
};

export default function SampleAudios({ onSelectSample, isLoading }) {
  const [samples, setSamples] = useState([]);
  const [playingAudio, setPlayingAudio] = useState(null);
  const [audioElement, setAudioElement] = useState(null);

  useEffect(() => {
    // Fetch available sample audios from backend
    fetch('http://localhost:8000/api/samples')
      .then((res) => res.json())
      .then((data) => {
        if (data.samples && data.samples.length > 0) {
          setSamples(data.samples);
        } else {
          // Fallback static samples
          generateFallbackSamples();
        }
      })
      .catch(() => {
        generateFallbackSamples();
      });
  }, []);

  const generateFallbackSamples = () => {
    const list = [
      { emotion: 'happy', display_name: 'Happy', filename: '03-01-03-01-01-01-01.wav', actor: 'Actor 01 (Male)', sample_url: '/api/audio/03-01-03-01-01-01-01.wav' },
      { emotion: 'sad', display_name: 'Sad', filename: '03-01-04-01-01-01-01.wav', actor: 'Actor 01 (Male)', sample_url: '/api/audio/03-01-04-01-01-01-01.wav' },
      { emotion: 'angry', display_name: 'Angry', filename: '03-01-05-01-01-01-01.wav', actor: 'Actor 01 (Male)', sample_url: '/api/audio/03-01-05-01-01-01-01.wav' },
      { emotion: 'fearful', display_name: 'Fearful', filename: '03-01-06-01-01-01-01.wav', actor: 'Actor 01 (Male)', sample_url: '/api/audio/03-01-06-01-01-01-01.wav' },
      { emotion: 'calm', display_name: 'Calm', filename: '03-01-02-01-01-01-01.wav', actor: 'Actor 01 (Male)', sample_url: '/api/audio/03-01-02-01-01-01-01.wav' },
      { emotion: 'neutral', display_name: 'Neutral', filename: '03-01-01-01-01-01-01.wav', actor: 'Actor 01 (Male)', sample_url: '/api/audio/03-01-01-01-01-01-01.wav' },
      { emotion: 'disgust', display_name: 'Disgust', filename: '03-01-07-01-01-01-01.wav', actor: 'Actor 01 (Male)', sample_url: '/api/audio/03-01-07-01-01-01-01.wav' },
      { emotion: 'surprised', display_name: 'Surprised', filename: '03-01-08-01-01-01-01.wav', actor: 'Actor 01 (Male)', sample_url: '/api/audio/03-01-08-01-01-01-01.wav' },
    ];
    setSamples(list);
  };

  const handlePlayToggle = (sample) => {
    if (playingAudio === sample.filename) {
      if (audioElement) {
        audioElement.pause();
      }
      setPlayingAudio(null);
    } else {
      if (audioElement) {
        audioElement.pause();
      }
      const audio = new Audio(`http://localhost:8000${sample.sample_url}`);
      audio.onended = () => setPlayingAudio(null);
      audio.play().catch((e) => console.log('Audio playback error:', e));
      setAudioElement(audio);
      setPlayingAudio(sample.filename);
    }
  };

  const handleTestSample = async (sample) => {
    try {
      const res = await fetch(`http://localhost:8000${sample.sample_url}`);
      const blob = await res.blob();
      const file = new File([blob], sample.filename, { type: 'audio/wav' });
      onSelectSample(file);
    } catch (e) {
      alert('Could not load sample audio: ' + e.message);
    }
  };

  return (
    <div className="bg-navy-900/90 border border-navy-800 rounded-2xl p-6 shadow-xl relative overflow-hidden backdrop-blur-sm">
      <div className="flex items-center justify-between pb-3 border-b border-navy-800 mb-4">
        <div className="flex items-center gap-2">
          <Disc3 className="w-4 h-4 text-violet-400" />
          <span className="text-xs font-semibold uppercase tracking-wider text-slate-400">
            Instant 1-Click Evaluation (RAVDESS Speech Benchmark)
          </span>
        </div>
        <span className="text-[11px] font-mono text-slate-500 hidden sm:inline">
          Actor 01 Ground Truth Recordings
        </span>
      </div>

      <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
        {samples.map((sample) => {
          const theme = EMOTION_THEMES[sample.emotion] || {
            color: '#38bdf8',
            bg: 'rgba(56, 189, 248, 0.1)',
            emoji: '🎙️',
          };
          const isCurrentPlaying = playingAudio === sample.filename;

          return (
            <div
              key={sample.emotion}
              className="group p-3 rounded-xl bg-navy-950/80 border border-navy-800/80 hover:border-navy-700 transition-all flex flex-col justify-between"
            >
              <div>
                <div className="flex items-center justify-between mb-2">
                  <span className="text-2xl">{theme.emoji}</span>
                  <button
                    onClick={() => handlePlayToggle(sample)}
                    className="p-1.5 rounded-lg bg-navy-900 hover:bg-navy-800 text-slate-300 transition-all"
                    title={isCurrentPlaying ? 'Pause Audio' : 'Listen to Audio'}
                  >
                    {isCurrentPlaying ? <Pause className="w-3.5 h-3.5 text-cyan-400" /> : <Play className="w-3.5 h-3.5 ml-0.5" />}
                  </button>
                </div>
                <div className="font-semibold text-xs text-white capitalize">{sample.display_name}</div>
                <div className="text-[10px] text-slate-500 font-mono mt-0.5">{sample.actor}</div>
              </div>

              <button
                onClick={() => handleTestSample(sample)}
                disabled={isLoading}
                className="mt-3 w-full py-1.5 px-2 rounded-lg text-[11px] font-medium flex items-center justify-center gap-1.5 border transition-all cursor-pointer disabled:opacity-50"
                style={{
                  backgroundColor: theme.bg,
                  color: theme.color,
                  borderColor: `${theme.color}30`,
                }}
              >
                <Zap className="w-3 h-3" />
                <span>Test Model</span>
              </button>
            </div>
          );
        })}
      </div>
    </div>
  );
}
