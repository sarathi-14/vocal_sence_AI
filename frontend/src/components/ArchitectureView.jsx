import React from 'react';
import { Layers, Cpu, Radio, Sparkles, Sliders, Database, ArrowRight } from 'lucide-react';

export default function ArchitectureView() {
  const pipelineSteps = [
    {
      step: '01',
      title: 'Acoustic Input',
      icon: Radio,
      desc: 'Raw audio waveform captured via microphone or file upload (WAV, MP3, WebM, OGG).',
      badge: 'Multi-Format',
      color: 'border-blue-500/40 text-blue-400',
    },
    {
      step: '02',
      title: 'Signal Preprocessing',
      icon: Sliders,
      desc: 'Resampling to 22,050 Hz mono, energy-based silence trimming, peak normalization, fixed 3.0s (66,150 samples).',
      badge: 'DSP Cleanse',
      color: 'border-cyan-500/40 text-cyan-400',
    },
    {
      step: '03',
      title: 'Feature Extraction',
      icon: Sparkles,
      desc: 'Computes 40-dimensional Mel-Frequency Cepstral Coefficients (MFCC) across 126 frames, Mel spectrogram, and centroid.',
      badge: '40-MFCC Seq',
      color: 'border-violet-500/40 text-violet-400',
    },
    {
      step: '04',
      title: 'CNN-BiLSTM Backbone',
      icon: Cpu,
      desc: 'Conv1D layers extract local spectral cues followed by Bidirectional LSTM layers capturing temporal prosody and emotional cadence.',
      badge: 'SOTA Hybrid',
      color: 'border-indigo-500/40 text-indigo-400',
    },
    {
      step: '05',
      title: 'Emotion Inference',
      icon: Layers,
      desc: 'Softmax activation outputs probability distribution over 8 target classes with valence & arousal classification.',
      badge: '8 Classes',
      color: 'border-emerald-500/40 text-emerald-400',
    },
  ];

  const taxonomy = [
    { emotion: 'Neutral', valence: 'Neutral', arousal: 'Low', desc: 'Objective vocal delivery, level pitch, balanced cadence' },
    { emotion: 'Calm', valence: 'Positive', arousal: 'Low', desc: 'Gentle intonation, smooth melodic transitions, steady energy' },
    { emotion: 'Happy', valence: 'Positive', arousal: 'High', desc: 'Dynamic pitch peaks, bright vocal harmonics, elevated energy' },
    { emotion: 'Sad', valence: 'Negative', arousal: 'Low', desc: 'Downward pitch slope, subdued volume, elongated pauses' },
    { emotion: 'Angry', valence: 'Negative', arousal: 'High', desc: 'Abrasive spectral sharpness, high RMS power, fast attack' },
    { emotion: 'Fearful', valence: 'Negative', arousal: 'High', desc: 'Tremulous pitch jitter, rapid breath intervals, erratic contour' },
    { emotion: 'Disgust', valence: 'Negative', arousal: 'Moderate', desc: 'Pharyngeal vocal fry, lowered fundamentals, constrained range' },
    { emotion: 'Surprised', valence: 'Neutral / Pos', arousal: 'High', desc: 'Sudden high-frequency jump, brief wide dynamic expansion' },
  ];

  return (
    <div className="space-y-6">
      {/* End-to-End Pipeline */}
      <div className="bg-navy-900/90 border border-navy-800 rounded-2xl p-6 shadow-xl backdrop-blur-sm">
        <h3 className="text-sm font-semibold uppercase tracking-wider text-slate-400 mb-6 flex items-center gap-2">
          <Layers className="w-4 h-4 text-blue-400" />
          End-to-End Speech Emotion Recognition Pipeline
        </h3>

        <div className="grid grid-cols-1 md:grid-cols-5 gap-4 relative">
          {pipelineSteps.map((step, idx) => {
            const Icon = step.icon;
            return (
              <div
                key={step.step}
                className="p-4 rounded-xl bg-navy-950/80 border border-navy-800 flex flex-col justify-between relative"
              >
                <div>
                  <div className="flex items-center justify-between mb-3">
                    <span className="text-xs font-mono font-bold text-slate-500">{step.step}</span>
                    <span className={`px-1.5 py-0.5 rounded text-[10px] font-mono border ${step.color}`}>
                      {step.badge}
                    </span>
                  </div>

                  <div className="w-8 h-8 rounded-lg bg-navy-900 flex items-center justify-center mb-3 text-slate-200">
                    <Icon className="w-4 h-4" />
                  </div>

                  <h4 className="text-xs font-bold text-white mb-1.5">{step.title}</h4>
                  <p className="text-[11px] text-slate-400 leading-relaxed">{step.desc}</p>
                </div>
              </div>
            );
          })}
        </div>
      </div>

      {/* Emotion Taxonomy Table */}
      <div className="bg-navy-900/90 border border-navy-800 rounded-2xl p-6 shadow-xl backdrop-blur-sm">
        <h3 className="text-sm font-semibold uppercase tracking-wider text-slate-400 mb-4 flex items-center gap-2">
          <Database className="w-4 h-4 text-cyan-400" />
          RAVDESS Emotion Taxonomy & Acoustic Indicators
        </h3>

        <div className="overflow-x-auto">
          <table className="w-full text-left text-xs border-collapse">
            <thead>
              <tr className="border-b border-navy-800 text-slate-400 font-mono text-[11px]">
                <th className="py-2.5 px-3">Emotion</th>
                <th className="py-2.5 px-3">Valence</th>
                <th className="py-2.5 px-3">Arousal</th>
                <th className="py-2.5 px-3">Acoustic & Prosodic Characteristics</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-navy-800/60 font-sans">
              {taxonomy.map((item) => (
                <tr key={item.emotion} className="hover:bg-navy-950/40 transition-colors">
                  <td className="py-2.5 px-3 font-semibold text-white">{item.emotion}</td>
                  <td className="py-2.5 px-3">
                    <span
                      className={`px-2 py-0.5 rounded text-[10px] font-medium ${
                        item.valence.includes('Positive')
                          ? 'bg-emerald-500/10 text-emerald-400'
                          : item.valence.includes('Negative')
                          ? 'bg-rose-500/10 text-rose-400'
                          : 'bg-slate-500/10 text-slate-300'
                      }`}
                    >
                      {item.valence}
                    </span>
                  </td>
                  <td className="py-2.5 px-3">
                    <span
                      className={`px-2 py-0.5 rounded text-[10px] font-medium ${
                        item.arousal === 'High'
                          ? 'bg-red-500/10 text-red-400'
                          : item.arousal === 'Moderate'
                          ? 'bg-amber-500/10 text-amber-400'
                          : 'bg-blue-500/10 text-blue-400'
                      }`}
                    >
                      {item.arousal}
                    </span>
                  </td>
                  <td className="py-2.5 px-3 text-slate-300 text-[11px]">{item.desc}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
}
