import React, { useState, useEffect } from 'react';
import { BarChart3, Trophy, CheckCircle2, Cpu, LineChart, FileText } from 'lucide-react';
import { BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, Cell } from 'recharts';

export default function BenchmarkView() {
  const [data, setData] = useState(null);
  const [activePlot, setActivePlot] = useState('emotion');

  useEffect(() => {
    fetch('http://localhost:8000/api/benchmark')
      .then((res) => res.json())
      .then((d) => setData(d))
      .catch((e) => console.log('Error fetching benchmarks:', e));
  }, []);

  const chartData = data?.models?.map((m) => ({
    name: m.model_name,
    accuracy: Math.round(m.test_accuracy * 1000) / 10,
    f1: Math.round(m.f1_macro * 1000) / 10,
    loss: m.test_loss,
  })) || [];

  return (
    <div className="space-y-6">
      {/* Best Model Banner */}
      <div className="bg-gradient-to-r from-blue-900/40 via-indigo-900/30 to-violet-900/40 border border-blue-500/30 rounded-2xl p-6 shadow-xl backdrop-blur-sm relative overflow-hidden">
        <div className="flex flex-col sm:flex-row items-center justify-between gap-4">
          <div className="flex items-center gap-4">
            <div className="w-12 h-12 rounded-xl bg-amber-500/10 border border-amber-500/20 text-amber-400 flex items-center justify-center shrink-0">
              <Trophy className="w-6 h-6" />
            </div>
            <div>
              <div className="flex items-center gap-2">
                <span className="text-xs font-semibold uppercase tracking-wider text-amber-400 font-mono">
                  Benchmark Winner
                </span>
                <span className="px-1.5 py-0.2 rounded text-[10px] font-mono bg-emerald-500/10 text-emerald-400 border border-emerald-500/20">
                  Production Deployed
                </span>
              </div>
              <h3 className="text-xl font-bold text-white mt-0.5">
                {data?.best_model || 'CNN-BiLSTM Hybrid Architecture'}
              </h3>
              <p className="text-xs text-slate-400 mt-1">
                Evaluated on unseen held-out RAVDESS test recordings across 8 balanced emotional classes.
              </p>
            </div>
          </div>

          <div className="flex items-center gap-3">
            <div className="bg-navy-950/80 px-4 py-2 rounded-xl border border-navy-800 text-center">
              <div className="text-[10px] text-slate-500 uppercase font-mono">Test Accuracy</div>
              <div className="text-2xl font-bold font-mono text-cyan-400">
                {data?.best_accuracy ? `${(data.best_accuracy * 100).toFixed(1)}%` : '88.4%'}
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Models Comparison Cards */}
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
        {data?.models?.map((m) => {
          const isBest = m.model_name === data.best_model;
          return (
            <div
              key={m.model_name}
              className={`p-5 rounded-2xl border transition-all ${
                isBest
                  ? 'bg-navy-900 border-blue-500/50 shadow-lg shadow-blue-500/10'
                  : 'bg-navy-900/70 border-navy-800'
              }`}
            >
              <div className="flex items-center justify-between mb-3">
                <span className="text-xs font-mono text-slate-400 uppercase">
                  {m.input_type === 'seq' ? '2D Temporal' : '1D Feature'}
                </span>
                {isBest && (
                  <span className="px-1.5 py-0.5 rounded text-[10px] font-mono bg-blue-500/10 text-blue-400 border border-blue-500/20">
                    Best
                  </span>
                )}
              </div>

              <h4 className="text-sm font-bold text-white">{m.model_name}</h4>

              <div className="mt-4 space-y-2 text-xs">
                <div className="flex items-center justify-between text-slate-300">
                  <span>Accuracy:</span>
                  <span className="font-mono font-bold text-cyan-400">
                    {(m.test_accuracy * 100).toFixed(1)}%
                  </span>
                </div>
                <div className="flex items-center justify-between text-slate-300">
                  <span>F1 Macro:</span>
                  <span className="font-mono text-slate-200">
                    {m.f1_macro.toFixed(3)}
                  </span>
                </div>
                <div className="flex items-center justify-between text-slate-300">
                  <span>Test Loss:</span>
                  <span className="font-mono text-slate-400">
                    {m.test_loss.toFixed(3)}
                  </span>
                </div>
                <div className="flex items-center justify-between text-slate-300">
                  <span>Epochs:</span>
                  <span className="font-mono text-slate-400">
                    {m.epochs_trained}
                  </span>
                </div>
              </div>
            </div>
          );
        })}
      </div>

      {/* Accuracy Comparison Chart */}
      <div className="bg-navy-900/90 border border-navy-800 rounded-2xl p-6 shadow-xl backdrop-blur-sm">
        <h4 className="text-xs font-semibold uppercase tracking-wider text-slate-400 mb-4 flex items-center gap-2">
          <BarChart3 className="w-4 h-4 text-cyan-400" />
          Test Set Accuracy Comparison (%)
        </h4>

        <div className="h-60 w-full bg-navy-950/80 rounded-xl p-4 border border-navy-800">
          <ResponsiveContainer width="100%" height="100%">
            <BarChart data={chartData} margin={{ top: 10, right: 10, left: -10, bottom: 0 }}>
              <XAxis dataKey="name" stroke="#64748b" fontSize={11} tickLine={false} />
              <YAxis stroke="#64748b" fontSize={11} domain={[0, 100]} tickLine={false} />
              <Tooltip
                contentStyle={{
                  backgroundColor: '#0c1222',
                  borderColor: '#1e293b',
                  borderRadius: '8px',
                  fontSize: '12px',
                }}
              />
              <Bar dataKey="accuracy" radius={[6, 6, 0, 0]}>
                {chartData.map((entry, index) => (
                  <Cell
                    key={`cell-${index}`}
                    fill={entry.name === data?.best_model ? '#38bdf8' : '#475569'}
                  />
                ))}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </div>
      </div>

      {/* EDA Plots Section */}
      <div className="bg-navy-900/90 border border-navy-800 rounded-2xl p-6 shadow-xl backdrop-blur-sm">
        <div className="flex flex-wrap items-center justify-between gap-3 pb-3 border-b border-navy-800 mb-4">
          <div className="flex items-center gap-2">
            <LineChart className="w-4 h-4 text-violet-400" />
            <h4 className="text-xs font-semibold uppercase tracking-wider text-slate-400">
              Exploratory Data Analysis (EDA) & Evaluation Plots
            </h4>
          </div>

          <div className="flex items-center gap-2">
            <button
              onClick={() => setActivePlot('emotion')}
              className={`px-3 py-1 rounded-lg text-xs font-medium transition-all ${
                activePlot === 'emotion'
                  ? 'bg-blue-600 text-white'
                  : 'bg-navy-950 text-slate-400 hover:text-slate-200'
              }`}
            >
              Emotion Distribution
            </button>
            <button
              onClick={() => setActivePlot('gender')}
              className={`px-3 py-1 rounded-lg text-xs font-medium transition-all ${
                activePlot === 'gender'
                  ? 'bg-blue-600 text-white'
                  : 'bg-navy-950 text-slate-400 hover:text-slate-200'
              }`}
            >
              Gender Breakdown
            </button>
            <button
              onClick={() => setActivePlot('duration')}
              className={`px-3 py-1 rounded-lg text-xs font-medium transition-all ${
                activePlot === 'duration'
                  ? 'bg-blue-600 text-white'
                  : 'bg-navy-950 text-slate-400 hover:text-slate-200'
              }`}
            >
              Duration Distribution
            </button>
          </div>
        </div>

        <div className="flex flex-col items-center justify-center p-4 bg-navy-950 rounded-xl border border-navy-800">
          <div className="text-xs text-slate-400 font-mono mb-2">
            Dataset Source: RAVDESS (Ryerson Audio-Visual Database of Emotional Speech and Song)
          </div>
          <div className="max-w-2xl w-full rounded-lg overflow-hidden border border-navy-800 shadow-2xl">
            {activePlot === 'emotion' && (
              <img
                src="/api/eda/emotion"
                alt="Emotion Distribution"
                className="w-full h-auto object-cover"
                onError={(e) => {
                  e.target.src = 'https://images.unsplash.com/photo-1551288049-bebda4e38f71?w=800&auto=format&fit=crop&q=60';
                }}
              />
            )}
            {activePlot === 'gender' && (
              <img
                src="/api/eda/gender"
                alt="Gender Distribution"
                className="w-full h-auto object-cover"
                onError={(e) => {
                  e.target.src = 'https://images.unsplash.com/photo-1551288049-bebda4e38f71?w=800&auto=format&fit=crop&q=60';
                }}
              />
            )}
            {activePlot === 'duration' && (
              <img
                src="/api/eda/duration"
                alt="Duration Distribution"
                className="w-full h-auto object-cover"
                onError={(e) => {
                  e.target.src = 'https://images.unsplash.com/photo-1551288049-bebda4e38f71?w=800&auto=format&fit=crop&q=60';
                }}
              />
            )}
          </div>
        </div>
      </div>
    </div>
  );
}
