import React from "react";
import { Mic, Upload, Square } from "lucide-react";

export default function AudioInput({
  audioFile,
  recordedBlob,
  isRecording,
  fileInputRef,
  onFileClick,
  onFileChange,
  onRecordToggle,
}) {
  return (
    <div className="bg-white/10 backdrop-blur-md rounded-xl p-6 mb-6 border border-white/20">
      <h2 className="text-xl font-semibold text-white mb-4">Audio Input</h2>

      <div className="grid md:grid-cols-2 gap-4">
        {/* File Upload */}
        <div>
          <input
            type="file"
            ref={fileInputRef}
            onChange={onFileChange}
            accept="audio/*"
            className="hidden"
          />
          <button
            onClick={onFileClick}
            className="w-full bg-purple-600 hover:bg-purple-700 text-white rounded-lg p-4 flex items-center justify-center gap-3 transition-colors"
          >
            <Upload className="w-5 h-5" />
            <span>Upload Audio File</span>
          </button>
          {audioFile && (
            <p className="text-green-300 text-sm mt-2">✓ {audioFile.name}</p>
          )}
        </div>

        {/* Microphone Recording */}
        <div>
          <button
            onClick={onRecordToggle}
            className={`w-full ${
              isRecording
                ? "bg-red-600 hover:bg-red-700"
                : "bg-blue-600 hover:bg-blue-700"
            } text-white rounded-lg p-4 flex items-center justify-center gap-3 transition-colors`}
          >
            {isRecording ? (
              <>
                <Square className="w-5 h-5" />
                <span>Stop Recording</span>
              </>
            ) : (
              <>
                <Mic className="w-5 h-5" />
                <span>Record Audio</span>
              </>
            )}
          </button>
          {recordedBlob && (
            <p className="text-green-300 text-sm mt-2">✓ Recording ready</p>
          )}
        </div>
      </div>
    </div>
  );
}
