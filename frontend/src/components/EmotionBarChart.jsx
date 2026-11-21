import React from 'react';
import { BarChart3 } from 'lucide-react';

export default function EmotionBarChart({ probabilities }) {
  if (!probabilities || probabilities.length === 0) return null;

  return (
    <div className="bg-navy-900/90 border border-navy-800 rounded-2xl p-6 shadow-xl relative overflow-hidden backdrop-blur-sm">
      <div className="flex items-center justify-between pb-3 border-b border-navy-800 mb-4">
        <div className="flex items-center gap-2">
          <BarChart3 className="w-4 h-4 text-cyan-400" />
          <span className="text-xs font-semibold uppercase tracking-wider text-slate-400">
            Softmax Probability Distribution (8 Classes)
          </span>
        </div>
        <span className="text-[11px] font-mono text-slate-500">Sorted by Confidence</span>
      </div>

      <div className="space-y-3">
        {probabilities.map((item, index) => {
          const isWinner = index === 0;
          return (
            <div key={item.emotion} className="space-y-1">
              <div className="flex items-center justify-between text-xs">
                <div className="flex items-center gap-2 font-medium">
                  <span
                    className="w-2.5 h-2.5 rounded-full shrink-0"
                    style={{ backgroundColor: item.color }}
                  />
                  <span className={isWinner ? 'text-white font-bold' : 'text-slate-300'}>
                    {item.display_name}
                  </span>
                  {isWinner && (
                    <span className="px-1.5 py-0.2 rounded text-[10px] font-mono bg-blue-500/20 text-blue-300 border border-blue-500/30">
                      Top Match
                    </span>
                  )}
                </div>
                <div className="flex items-center gap-2 font-mono">
                  <span className="text-slate-400 text-[11px]">
                    (p = {item.probability.toFixed(3)})
                  </span>
                  <span className={`font-bold ${isWinner ? 'text-cyan-400' : 'text-slate-200'}`}>
                    {item.percentage}%
                  </span>
                </div>
              </div>

              {/* Progress Bar Container */}
              <div className="h-2.5 w-full bg-navy-950 rounded-full overflow-hidden border border-navy-800/80 p-0.5 flex items-center">
                <div
                  className="h-full rounded-full transition-all duration-700 ease-out"
                  style={{
                    width: `${Math.max(item.percentage, 1)}%`,
                    backgroundColor: item.color,
                    boxShadow: isWinner ? `0 0 10px ${item.color}80` : 'none',
                  }}
                />
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}
