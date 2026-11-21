import React, { useRef, useState } from 'react';
import { UploadCloud, FileAudio, Play, Pause, Trash2, ArrowRight, Loader2 } from 'lucide-react';

export default function AudioUploader({ onAnalyze, isLoading }) {
  const [dragActive, setDragActive] = useState(false);
  const [selectedFile, setSelectedFile] = useState(null);
  const [audioUrl, setAudioUrl] = useState(null);
  const [isPlaying, setIsPlaying] = useState(false);
  const audioRef = useRef(null);
  const inputRef = useRef(null);

  const handleDrag = (e) => {
    e.preventDefault();
    e.stopPropagation();
    if (e.type === 'dragenter' || e.type === 'dragover') {
      setDragActive(true);
    } else if (e.type === 'dragleave') {
      setDragActive(false);
    }
  };

  const handleDrop = (e) => {
    e.preventDefault();
    e.stopPropagation();
    setDragActive(false);
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      handleFile(e.dataTransfer.files[0]);
    }
  };

  const handleFileInput = (e) => {
    if (e.target.files && e.target.files[0]) {
      handleFile(e.target.files[0]);
    }
  };

  const handleFile = (file) => {
    if (!file.type.includes('audio') && !file.name.match(/\.(wav|mp3|ogg|webm|m4a|flac)$/i)) {
      alert('Please upload a valid audio file (WAV, MP3, WebM, OGG, M4A).');
      return;
    }
    setSelectedFile(file);
    const url = URL.createObjectURL(file);
    setAudioUrl(url);
    setIsPlaying(false);
  };

  const handleRemove = () => {
    setSelectedFile(null);
    if (audioUrl) {
      URL.revokeObjectURL(audioUrl);
      setAudioUrl(null);
    }
    setIsPlaying(false);
    if (inputRef.current) inputRef.current.value = '';
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

  const handleSubmit = () => {
    if (!selectedFile) return;
    onAnalyze(selectedFile);
  };

  return (
    <div className="bg-navy-900/90 border border-navy-800 rounded-2xl p-6 shadow-xl relative overflow-hidden backdrop-blur-sm">
      <h3 className="text-sm font-semibold uppercase tracking-wider text-slate-400 mb-4 flex items-center gap-2">
        <UploadCloud className="w-4 h-4 text-blue-400" />
        Audio File Upload
      </h3>

      {!selectedFile ? (
        <div
          onDragEnter={handleDrag}
          onDragLeave={handleDrag}
          onDragOver={handleDrag}
          onDrop={handleDrop}
          onClick={() => inputRef.current?.click()}
          className={`border-2 border-dashed rounded-xl p-8 text-center cursor-pointer transition-all ${
            dragActive
              ? 'border-blue-400 bg-blue-500/10 scale-[1.01]'
              : 'border-navy-700 hover:border-navy-600 bg-navy-950/50 hover:bg-navy-950/80'
          }`}
        >
          <input
            ref={inputRef}
            type="file"
            accept="audio/*,.wav,.mp3,.ogg,.webm,.m4a"
            className="hidden"
            onChange={handleFileInput}
          />
          <div className="flex flex-col items-center justify-center">
            <div className="w-14 h-14 rounded-2xl bg-blue-500/10 border border-blue-500/20 flex items-center justify-center mb-3 text-blue-400">
              <UploadCloud className="w-7 h-7" />
            </div>
            <p className="text-sm font-medium text-slate-200">
              Drop an audio recording here, or <span className="text-blue-400 underline underline-offset-2">browse</span>
            </p>
            <p className="text-xs text-slate-500 mt-1 font-mono">
              Supports WAV, MP3, WebM, OGG, M4A • Max 20MB
            </p>
          </div>
        </div>
      ) : (
        <div className="space-y-4">
          <div className="p-4 rounded-xl bg-navy-950 border border-navy-800 flex items-center justify-between">
            <div className="flex items-center gap-3 overflow-hidden">
              <button
                onClick={togglePlay}
                className="w-10 h-10 rounded-xl bg-blue-600 hover:bg-blue-500 text-white flex items-center justify-center shrink-0 shadow-lg shadow-blue-500/20 transition-all"
              >
                {isPlaying ? <Pause className="w-5 h-5" /> : <Play className="w-5 h-5 ml-0.5" />}
              </button>
              <div className="truncate">
                <div className="text-sm font-medium text-slate-200 truncate">{selectedFile.name}</div>
                <div className="text-xs text-slate-400 font-mono mt-0.5">
                  {(selectedFile.size / 1024).toFixed(1)} KB • {selectedFile.type || 'audio/wav'}
                </div>
              </div>
            </div>

            <button
              onClick={handleRemove}
              className="p-2 text-slate-500 hover:text-red-400 hover:bg-navy-900 rounded-lg transition-all"
              title="Remove file"
            >
              <Trash2 className="w-4 h-4" />
            </button>
          </div>

          <audio
            ref={audioRef}
            src={audioUrl}
            onEnded={() => setIsPlaying(false)}
            className="hidden"
          />

          <button
            onClick={handleSubmit}
            disabled={isLoading}
            className="w-full py-3 px-4 rounded-xl bg-gradient-to-r from-blue-600 to-indigo-600 hover:from-blue-500 hover:to-indigo-500 text-white font-medium text-sm shadow-lg shadow-blue-600/25 flex items-center justify-center gap-2 transition-all disabled:opacity-50 disabled:cursor-not-allowed cursor-pointer"
          >
            {isLoading ? (
              <>
                <Loader2 className="w-4 h-4 animate-spin" />
                <span>Extracting Acoustic Features & Classifying...</span>
              </>
            ) : (
              <>
                <span>Analyze Vocal Emotion</span>
                <ArrowRight className="w-4 h-4" />
              </>
            )}
          </button>
        </div>
      )}
    </div>
  );
}
