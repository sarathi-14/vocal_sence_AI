import React, { useState } from 'react';
import { Activity, UploadCloud, Clock, ArrowRight, Loader2, Sparkles, TrendingUp } from 'lucide-react';
import { AreaChart, Area, XAxis, YAxis, Tooltip, ResponsiveContainer } from 'recharts';

export default function TrajectoryViewer() {
  const [selectedFile, setSelectedFile] = useState(null);
  const [isLoading, setIsLoading] = useState(false);
  const [result, setResult] = useState(null);

  const handleFileChange = (e) => {
    if (e.target.files && e.target.files[0]) {
      setSelectedFile(e.target.files[0]);
    }
  };

  const handleAnalyze = async () => {
    if (!selectedFile) return;
    setIsLoading(true);
    const formData = new FormData();
    formData.append('file', selectedFile);

    try {
      const res = await fetch('http://localhost:8000/api/analyze?window_sec=2.5&step_sec=1.0', {
        method: 'POST',
        body: formData,
      });
      const data = await res.json();
      setResult(data);
    } catch (e) {
      alert('Error analyzing trajectory: ' + e.message);
    } finally {
      setIsLoading(false);
    }
  };

  const chartData = result?.trajectory?.map((seg) => ({
    time: `${seg.start_seconds}s - ${seg.end_seconds}s`,
    confidence: Math.round(seg.confidence * 100),
    emotion: seg.dominant_emotion,
  })) || [];

  return (
    <div className="space-y-6">
      <div className="bg-navy-900/90 border border-navy-800 rounded-2xl p-6 shadow-xl backdrop-blur-sm">
        <div className="flex items-center justify-between pb-3 border-b border-navy-800 mb-4">
          <div className="flex items-center gap-2">
            <Activity className="w-4 h-4 text-cyan-400" />
            <h3 className="text-sm font-semibold uppercase tracking-wider text-slate-400">
              Temporal Emotion Trajectory Analysis
            </h3>
          </div>
          <span className="text-[11px] font-mono text-slate-500">2.5s Sliding Window</span>
        </div>

        <p className="text-xs text-slate-400 mb-6 leading-relaxed">
          Upload multi-second speech to detect how emotional tone evolves, escalates, or stabilizes across time windows.
        </p>

        <div className="flex flex-col sm:flex-row items-center gap-4">
          <input
            type="file"
            id="traj-file"
            accept="audio/*"
            className="hidden"
            onChange={handleFileChange}
          />
          <label
            htmlFor="traj-file"
            className="w-full sm:w-auto px-4 py-2.5 rounded-xl bg-navy-950 border border-navy-700 hover:border-navy-600 text-xs font-medium text-slate-300 flex items-center justify-center gap-2 cursor-pointer transition-all"
          >
            <UploadCloud className="w-4 h-4 text-blue-400" />
            <span>{selectedFile ? selectedFile.name : 'Choose Audio File'}</span>
          </label>

          <button
            onClick={handleAnalyze}
            disabled={!selectedFile || isLoading}
            className="w-full sm:w-auto px-6 py-2.5 rounded-xl bg-blue-600 hover:bg-blue-500 text-white text-xs font-medium shadow-md shadow-blue-500/20 flex items-center justify-center gap-2 transition-all disabled:opacity-50 cursor-pointer"
          >
            {isLoading ? (
              <>
                <Loader2 className="w-3.5 h-3.5 animate-spin" />
                <span>Computing Trajectory...</span>
              </>
            ) : (
              <>
                <span>Analyze Emotional Dynamics</span>
                <ArrowRight className="w-3.5 h-3.5" />
              </>
            )}
          </button>
        </div>
      </div>

      {result && (
        <div className="bg-navy-900/90 border border-navy-800 rounded-2xl p-6 shadow-xl backdrop-blur-sm space-y-6">
          <div className="flex flex-wrap items-center justify-between gap-4 pb-4 border-b border-navy-800">
            <div>
              <div className="text-[11px] text-slate-400 uppercase tracking-wider font-mono">
                Overall Classification
              </div>
              <div className="text-2xl font-bold text-white capitalize mt-0.5">
                {result.overall_emotion} ({(result.overall_confidence * 100).toFixed(1)}%)
              </div>
            </div>

            <div className="flex items-center gap-4 text-xs font-mono text-slate-400">
              <div className="bg-navy-950 px-3 py-1.5 rounded-lg border border-navy-800">
                Total Duration: <span className="text-slate-200 font-bold">{result.total_duration}s</span>
              </div>
              <div className="bg-navy-950 px-3 py-1.5 rounded-lg border border-navy-800">
                Segments: <span className="text-slate-200 font-bold">{result.trajectory?.length}</span>
              </div>
            </div>
          </div>

          {/* Timeline visualization */}
          <div>
            <h4 className="text-xs font-semibold uppercase tracking-wider text-slate-400 mb-3 flex items-center gap-1.5">
              <TrendingUp className="w-3.5 h-3.5 text-blue-400" />
              Segment-by-Segment Timeline
            </h4>
            <div className="flex flex-wrap gap-2">
              {result.trajectory?.map((seg) => (
                <div
                  key={seg.segment_index}
                  className="p-2.5 rounded-xl bg-navy-950 border flex flex-col justify-between min-w-[120px]"
                  style={{ borderColor: `${seg.color}40` }}
                >
                  <div className="flex items-center justify-between text-[10px] text-slate-500 font-mono mb-1">
                    <span>#{seg.segment_index}</span>
                    <span>{seg.start_seconds}s - {seg.end_seconds}s</span>
                  </div>
                  <div className="font-bold text-xs capitalize text-white">
                    {seg.dominant_emotion}
                  </div>
                  <div
                    className="text-[11px] font-mono font-semibold mt-1"
                    style={{ color: seg.color }}
                  >
                    {(seg.confidence * 100).toFixed(0)}%
                  </div>
                </div>
              ))}
            </div>
          </div>

          {/* Trajectory Confidence Chart */}
          <div>
            <h4 className="text-xs font-semibold uppercase tracking-wider text-slate-400 mb-3">
              Confidence & Emotion Trajectory Curve
            </h4>
            <div className="h-48 w-full bg-navy-950/80 rounded-xl p-3 border border-navy-800">
              <ResponsiveContainer width="100%" height="100%">
                <AreaChart data={chartData}>
                  <defs>
                    <linearGradient id="confidenceGrad" x1="0" y1="0" x2="0" y2="1">
                      <stop offset="5%" stopColor="#38bdf8" stopOpacity={0.4} />
                      <stop offset="95%" stopColor="#38bdf8" stopOpacity={0.0} />
                    </linearGradient>
                  </defs>
                  <XAxis dataKey="time" stroke="#64748b" fontSize={10} tickLine={false} />
                  <YAxis stroke="#64748b" fontSize={10} domain={[0, 100]} tickLine={false} />
                  <Tooltip
                    contentStyle={{
                      backgroundColor: '#0c1222',
                      borderColor: '#1e293b',
                      borderRadius: '8px',
                      fontSize: '12px',
                    }}
                  />
                  <Area
                    type="monotone"
                    dataKey="confidence"
                    stroke="#38bdf8"
                    strokeWidth={2}
                    fillOpacity={1}
                    fill="url(#confidenceGrad)"
                  />
                </AreaChart>
              </ResponsiveContainer>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
