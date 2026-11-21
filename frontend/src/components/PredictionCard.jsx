import React from 'react';
import {
  Sparkles,
  Zap,
  Activity,
  Clock,
  Gauge,
  AudioWaveform,
  Volume2,
  Info
} from 'lucide-react';

const EMOTION_EMOJIS = {
  neutral: '😐',
  calm: '😌',
  happy: '😄',
  sad: '😢',
  angry: '😠',
  fearful: '😨',
  disgust: '🤢',
  surprised: '😲',
};

export default function PredictionCard({ result }) {
  if (!result) return null;

  const {
    predicted_emotion,
    display_name,
    confidence_percentage,
    color,
    diagnostics,
    insight,
    model_name,
    inference_time_ms,
  } = result;

  const emoji = EMOTION_EMOJIS[predicted_emotion] || '🎙️';

  const getArousalBadge = (arousal) => {
    switch (arousal?.toLowerCase()) {
      case 'high':
        return 'bg-red-500/10 text-red-400 border-red-500/30';
      case 'moderate':
        return 'bg-amber-500/10 text-amber-400 border-amber-500/30';
      default:
        return 'bg-blue-500/10 text-blue-400 border-blue-500/30';
    }
  };

  const getValenceBadge = (valence) => {
    if (valence?.toLowerCase().includes('positive')) {
      return 'bg-emerald-500/10 text-emerald-400 border-emerald-500/30';
    }
    if (valence?.toLowerCase().includes('negative')) {
      return 'bg-rose-500/10 text-rose-400 border-rose-500/30';
    }
    return 'bg-slate-500/10 text-slate-300 border-slate-500/30';
  };

  return (
    <div className="bg-navy-900/90 border border-navy-800 rounded-2xl p-6 shadow-xl relative overflow-hidden backdrop-blur-sm">
      {/* Decorative background glow */}
      <div
        className="absolute -top-12 -right-12 w-48 h-48 rounded-full blur-3xl opacity-20 pointer-events-none"
        style={{ backgroundColor: color }}
      />

      <div className="flex items-center justify-between pb-4 border-b border-navy-800">
        <div className="flex items-center gap-2">
          <Sparkles className="w-4 h-4 text-blue-400" />
          <span className="text-xs font-semibold uppercase tracking-wider text-slate-400">
            Emotion Recognition Result
          </span>
        </div>
        <div className="flex items-center gap-2 font-mono text-[11px] text-slate-400">
          <Zap className="w-3.5 h-3.5 text-amber-400" />
          <span>{inference_time_ms} ms</span>
        </div>
      </div>

      {/* Main Emotion Display */}
      <div className="py-6 flex flex-col sm:flex-row items-center sm:items-start gap-5">
        <div
          className="w-20 h-20 rounded-2xl flex items-center justify-center text-4xl shadow-xl shrink-0 border border-white/10"
          style={{
            backgroundColor: `${color}20`,
            boxShadow: `0 10px 25px -5px ${color}30`,
          }}
        >
          {emoji}
        </div>

        <div className="text-center sm:text-left flex-1">
          <div className="flex flex-wrap items-center justify-center sm:justify-start gap-2 mb-1.5">
            <h2 className="text-3xl font-extrabold tracking-tight text-white">
              {display_name}
            </h2>
            <span
              className="px-2.5 py-1 rounded-full text-xs font-bold font-mono tracking-wide border"
              style={{
                backgroundColor: `${color}15`,
                color: color,
                borderColor: `${color}40`,
              }}
            >
              {confidence_percentage}% Confidence
            </span>
          </div>

          <div className="flex flex-wrap items-center justify-center sm:justify-start gap-2 mt-2">
            <span
              className={`px-2 py-0.5 rounded text-[11px] font-medium border ${getValenceBadge(
                insight?.valence
              )}`}
            >
              Valence: {insight?.valence || 'Neutral'}
            </span>
            <span
              className={`px-2 py-0.5 rounded text-[11px] font-medium border ${getArousalBadge(
                insight?.arousal
              )}`}
            >
              Arousal: {insight?.arousal || 'Moderate'}
            </span>
            <span className="px-2 py-0.5 rounded text-[11px] font-mono text-slate-400 bg-navy-950 border border-navy-800">
              {insight?.confidence_tier || 'High'}
            </span>
          </div>

          <p className="mt-3 text-xs text-slate-300 leading-relaxed font-sans bg-navy-950/60 p-3 rounded-xl border border-navy-800/80">
            {insight?.description}
          </p>
        </div>
      </div>

      {/* Acoustic Diagnostics Grid */}
      {diagnostics && (
        <div className="mt-2 pt-4 border-t border-navy-800/80">
          <div className="text-[11px] font-semibold uppercase tracking-wider text-slate-500 mb-3 flex items-center gap-1.5">
            <Activity className="w-3.5 h-3.5 text-cyan-400" />
            <span>Extracted Acoustic Diagnostics</span>
          </div>

          <div className="grid grid-cols-2 sm:grid-cols-4 gap-2.5">
            <div className="bg-navy-950/80 p-2.5 rounded-xl border border-navy-800/60">
              <div className="flex items-center gap-1 text-[11px] text-slate-400">
                <Clock className="w-3 h-3 text-blue-400" />
                <span>Duration</span>
              </div>
              <div className="text-sm font-bold font-mono text-slate-200 mt-0.5">
                {diagnostics.duration_seconds}s
              </div>
            </div>

            <div className="bg-navy-950/80 p-2.5 rounded-xl border border-navy-800/60">
              <div className="flex items-center gap-1 text-[11px] text-slate-400">
                <Volume2 className="w-3 h-3 text-cyan-400" />
                <span>RMS Energy</span>
              </div>
              <div className="text-sm font-bold font-mono text-slate-200 mt-0.5">
                {diagnostics.mean_rms_energy}
              </div>
            </div>

            <div className="bg-navy-950/80 p-2.5 rounded-xl border border-navy-800/60">
              <div className="flex items-center gap-1 text-[11px] text-slate-400">
                <Gauge className="w-3 h-3 text-violet-400" />
                <span>Zero Crossing</span>
              </div>
              <div className="text-sm font-bold font-mono text-slate-200 mt-0.5">
                {diagnostics.mean_zero_crossing_rate}
              </div>
            </div>

            <div className="bg-navy-950/80 p-2.5 rounded-xl border border-navy-800/60">
              <div className="flex items-center gap-1 text-[11px] text-slate-400">
                <AudioWaveform className="w-3 h-3 text-emerald-400" />
                <span>Centroid</span>
              </div>
              <div className="text-sm font-bold font-mono text-slate-200 mt-0.5">
                {diagnostics.spectral_centroid_hz} Hz
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Model provenance info */}
      <div className="mt-4 pt-3 border-t border-navy-800/60 flex items-center justify-between text-[11px] text-slate-500 font-mono">
        <span>Model: {model_name}</span>
        <span>Target: 22,050 Hz • 40-MFCC</span>
      </div>
    </div>
  );
}
