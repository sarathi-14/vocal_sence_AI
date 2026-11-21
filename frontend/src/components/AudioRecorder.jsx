import React, { useState, useRef, useEffect } from 'react';
import { Mic, Square, Play, Pause, RotateCcw, ArrowRight, Loader2, Sparkles } from 'lucide-react';

export default function AudioRecorder({ onAnalyze, isLoading }) {
  const [isRecording, setIsRecording] = useState(false);
  const [audioBlob, setAudioBlob] = useState(null);
  const [audioUrl, setAudioUrl] = useState(null);
  const [recordingSeconds, setRecordingSeconds] = useState(0);
  const [isPlaying, setIsPlaying] = useState(false);
  
  const mediaRecorderRef = useRef(null);
  const audioChunksRef = useRef([]);
  const timerRef = useRef(null);
  const audioRef = useRef(null);

  useEffect(() => {
    return () => {
      if (timerRef.current) clearInterval(timerRef.current);
      if (audioUrl) URL.revokeObjectURL(audioUrl);
    };
  }, [audioUrl]);

  const startRecording = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      audioChunksRef.current = [];
      const mediaRecorder = new MediaRecorder(stream);
      mediaRecorderRef.current = mediaRecorder;

      mediaRecorder.ondataavailable = (event) => {
        if (event.data.size > 0) {
          audioChunksRef.current.push(event.data);
        }
      };

      mediaRecorder.onstop = () => {
        const blob = new Blob(audioChunksRef.current, { type: 'audio/wav' });
        setAudioBlob(blob);
        const url = URL.createObjectURL(blob);
        setAudioUrl(url);
        stream.getTracks().forEach((track) => track.stop());
      };

      mediaRecorder.start();
      setIsRecording(true);
      setRecordingSeconds(0);
      setAudioBlob(null);
      if (audioUrl) URL.revokeObjectURL(audioUrl);
      setAudioUrl(null);

      timerRef.current = setInterval(() => {
        setRecordingSeconds((prev) => prev + 1);
      }, 1000);
    } catch (err) {
      alert('Microphone access denied or not available: ' + err.message);
    }
  };

  const stopRecording = () => {
    if (mediaRecorderRef.current && isRecording) {
      mediaRecorderRef.current.stop();
      setIsRecording(false);
      if (timerRef.current) clearInterval(timerRef.current);
    }
  };

  const resetRecording = () => {
    setAudioBlob(null);
    if (audioUrl) URL.revokeObjectURL(audioUrl);
    setAudioUrl(null);
    setRecordingSeconds(0);
    setIsPlaying(false);
  };

  const togglePlay = () => {
    if (!audioRef.current) return;
    if (isPlaying) {
      audioRef.current.pause();
      setIsPlaying(false);
    } else {
      audioRef.current.play();
      setIsPlaying(true);
    }
  };

  const handleAnalyze = () => {
    if (!audioBlob) return;
    const file = new File([audioBlob], `microphone_capture_${Date.now()}.wav`, {
      type: 'audio/wav',
    });
    onAnalyze(file);
  };

  const formatTime = (secs) => {
    const mins = Math.floor(secs / 60);
    const remaining = secs % 60;
    return `${mins.toString().padStart(2, '0')}:${remaining.toString().padStart(2, '0')}`;
  };

  return (
    <div className="bg-navy-900/90 border border-navy-800 rounded-2xl p-6 shadow-xl relative overflow-hidden backdrop-blur-sm">
      <h3 className="text-sm font-semibold uppercase tracking-wider text-slate-400 mb-4 flex items-center gap-2">
        <Mic className="w-4 h-4 text-violet-400" />
        Live Microphone Capture
      </h3>

      <div className="flex flex-col items-center justify-center p-6 bg-navy-950/60 rounded-xl border border-navy-800">
        {!audioBlob ? (
          <>
            <div className="relative mb-4">
              {isRecording && (
                <div className="absolute inset-0 rounded-full bg-red-500/20 animate-ping" />
              )}
              <button
                onClick={isRecording ? stopRecording : startRecording}
                className={`relative w-16 h-16 rounded-full flex items-center justify-center shadow-lg transition-all ${
                  isRecording
                    ? 'bg-red-500 hover:bg-red-600 text-white shadow-red-500/30'
                    : 'bg-violet-600 hover:bg-violet-500 text-white shadow-violet-500/30 hover:scale-105'
                }`}
              >
                {isRecording ? <Square className="w-6 h-6" /> : <Mic className="w-7 h-7" />}
              </button>
            </div>

            <div className="text-center">
              <div className="text-xs font-mono font-bold text-slate-300">
                {isRecording ? (
                  <span className="text-red-400 flex items-center justify-center gap-2">
                    <span className="w-2 h-2 rounded-full bg-red-500 animate-pulse" />
                    Recording: {formatTime(recordingSeconds)}
                  </span>
                ) : (
                  'Click microphone to record your voice'
                )}
              </div>
              <p className="text-[11px] text-slate-500 mt-1 max-w-xs">
                Speak naturally for 2 to 4 seconds to capture emotion, pitch variations, and tone.
              </p>
            </div>
          </>
        ) : (
          <div className="w-full space-y-4">
            <div className="p-3 bg-navy-900 rounded-xl border border-navy-800 flex items-center justify-between">
              <div className="flex items-center gap-3">
                <button
                  onClick={togglePlay}
                  className="w-9 h-9 rounded-lg bg-violet-600 hover:bg-violet-500 text-white flex items-center justify-center shadow-md transition-all"
                >
                  {isPlaying ? <Pause className="w-4 h-4" /> : <Play className="w-4 h-4 ml-0.5" />}
                </button>
                <div>
                  <div className="text-xs font-medium text-slate-200">Captured Voice Sample</div>
                  <div className="text-[11px] text-slate-400 font-mono">
                    Duration: {formatTime(recordingSeconds)}
                  </div>
                </div>
              </div>

              <button
                onClick={resetRecording}
                className="p-1.5 text-slate-400 hover:text-slate-200 hover:bg-navy-800 rounded-lg text-xs flex items-center gap-1 transition-all"
                title="Discard & Record Again"
              >
                <RotateCcw className="w-3.5 h-3.5" />
                <span>Retake</span>
              </button>
            </div>

            <audio
              ref={audioRef}
              src={audioUrl}
              onEnded={() => setIsPlaying(false)}
              className="hidden"
            />

            <button
              onClick={handleAnalyze}
              disabled={isLoading}
              className="w-full py-2.5 px-4 rounded-xl bg-gradient-to-r from-violet-600 to-indigo-600 hover:from-violet-500 hover:to-indigo-500 text-white font-medium text-xs shadow-lg shadow-violet-600/25 flex items-center justify-center gap-2 transition-all disabled:opacity-50"
            >
              {isLoading ? (
                <>
                  <Loader2 className="w-4 h-4 animate-spin" />
                  <span>Classifying Speech Emotion...</span>
                </>
              ) : (
                <>
                  <span>Analyze Recorded Voice</span>
                  <ArrowRight className="w-3.5 h-3.5" />
                </>
              )}
            </button>
          </div>
        )}
      </div>
    </div>
  );
}
