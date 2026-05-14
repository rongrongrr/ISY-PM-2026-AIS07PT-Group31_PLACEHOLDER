import React, { useState, useRef, useEffect } from "react";
import { analyzeEmotion, fetchAvailableModels } from "../utils/api";
import AudioInput from "./AudioInput";
import EmotionResults from "./EmotionResults";
import SpectrogramResults from "./SpectrogramResults";
import ModelSelection from "./ModelSelection";

export default function AudioEmotionAnalyzer() {
  const [audioFile, setAudioFile] = useState(null);
  const [isRecording, setIsRecording] = useState(false);
  const [recordedBlob, setRecordedBlob] = useState(null);
  const [emotionResults, setEmotionResults] = useState(null);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [availableModels, setAvailableModels] = useState([]);
  const [selectedModel, setSelectedModel] = useState(null);
  const [showSpectrogram, setShowSpectrogram] = useState(true);

  const mediaRecorderRef = useRef(null);
  const audioChunksRef = useRef([]);
  const fileInputRef = useRef(null);

  useEffect(() => {
    fetchAvailableModels()
      .then((models) => {
        setAvailableModels(models);
        if (models.length > 0) {
          setSelectedModel(models[0].id);
        }
      })
      .catch((err) => console.error("Failed to fetch models:", err));
  }, []);

  const handleFileUpload = (e) => {
    const file = e.target.files[0];
    if (file && file.type.startsWith("audio/")) {
      setAudioFile(file);
      setRecordedBlob(null);
      setEmotionResults(null);
    }
  };

  const startRecording = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      mediaRecorderRef.current = new MediaRecorder(stream);
      audioChunksRef.current = [];

      mediaRecorderRef.current.ondataavailable = (e) => {
        audioChunksRef.current.push(e.data);
      };

      mediaRecorderRef.current.onstop = () => {
        const blob = new Blob(audioChunksRef.current, {
          type: mediaRecorderRef.current.mimeType || "audio/webm",
        });
        setRecordedBlob(blob);
        setAudioFile(null);
        setEmotionResults(null);
        stream.getTracks().forEach((track) => track.stop());
      };

      mediaRecorderRef.current.start();
      setIsRecording(true);
    } catch (err) {
      console.error("Error accessing microphone:", err);
      alert("Could not access microphone. Please check permissions.");
    }
  };

  const stopRecording = () => {
    if (mediaRecorderRef.current && isRecording) {
      mediaRecorderRef.current.stop();
      setIsRecording(false);
    }
  };

  const analyzeEmotions = async () => {
    if (!audioFile && !recordedBlob) {
      alert("Please upload an audio file or record audio");
      return;
    }

    setIsAnalyzing(true);

    try {
      const results = await fetchEmotionAnalysisFromBackend(
        audioFile || recordedBlob,
      );
      setEmotionResults(results);
    } catch (error) {
      console.error("Error analyzing emotions:", error);
      alert("Failed to analyze emotions. Please try again.");
    } finally {
      setIsAnalyzing(false);
    }
  };

  const fetchEmotionAnalysisFromBackend = async (audio) => {
    let audioFile = audio;

    if (audio instanceof Blob && !(audio instanceof File)) {
      audioFile = await convertBlobToWavFile(audio);
    }

    const results = await analyzeEmotion(audioFile);
    return results;
  };

  const convertBlobToWavFile = async (blob) => {
    const arrayBuffer = await new Promise((resolve, reject) => {
      const reader = new FileReader();
      reader.onload = () => resolve(reader.result);
      reader.onerror = reject;
      reader.readAsArrayBuffer(blob);
    });

    const AudioContext = window.AudioContext || window.webkitAudioContext;
    const audioCtx = new AudioContext();
    const decoded = await audioCtx.decodeAudioData(arrayBuffer);
    const wavBuffer = audioBufferToWav(decoded);
    const wavBlob = new Blob([wavBuffer], { type: "audio/wav" });

    return new File([wavBlob], "recording.wav", { type: "audio/wav" });
  };

  const audioBufferToWav = (audioBuffer) => {
    const numChannels = audioBuffer.numberOfChannels;
    const sampleRate = audioBuffer.sampleRate;
    const numFrames = audioBuffer.length;
    const bytesPerSample = 2;
    const blockAlign = numChannels * bytesPerSample;
    const buffer = new ArrayBuffer(44 + numFrames * blockAlign);
    const view = new DataView(buffer);

    const writeString = (offset, string) => {
      for (let i = 0; i < string.length; i += 1) {
        view.setUint8(offset + i, string.charCodeAt(i));
      }
    };

    writeString(0, "RIFF");
    view.setUint32(4, 36 + numFrames * blockAlign, true);
    writeString(8, "WAVE");
    writeString(12, "fmt ");
    view.setUint32(16, 16, true);
    view.setUint16(20, 1, true);
    view.setUint16(22, numChannels, true);
    view.setUint32(24, sampleRate, true);
    view.setUint32(28, sampleRate * blockAlign, true);
    view.setUint16(32, blockAlign, true);
    view.setUint16(34, bytesPerSample * 8, true);
    writeString(36, "data");
    view.setUint32(40, numFrames * blockAlign, true);

    let offset = 44;
    for (let i = 0; i < numFrames; i += 1) {
      for (let channel = 0; channel < numChannels; channel += 1) {
        const sample = audioBuffer.getChannelData(channel)[i];
        const clamped = Math.max(-1, Math.min(1, sample));
        view.setInt16(
          offset,
          clamped < 0 ? clamped * 0x8000 : clamped * 0x7fff,
          true,
        );
        offset += bytesPerSample;
      }
    }

    return buffer;
  };

  return (
    <div className="min-h-screen bg-[#05040D] text-slate-100 p-8">
      <div className="max-w-7xl mx-auto">
        <h1 className="text-xl font-bold text-white mb-2">
          Echoes of Emotion (Audio Emotion Analyzer)
        </h1>
        <p className="text-purple-200 mb-2">
          Visualizing the Sound of Emotions through Spectrograms
        </p>

        <div className="grid gap-8 lg:grid-cols-[0.95fr_1.5fr]">
          <div className="space-y-8">
            <AudioInput
              audioFile={audioFile}
              recordedBlob={recordedBlob}
              isRecording={isRecording}
              fileInputRef={fileInputRef}
              onFileClick={() => fileInputRef.current?.click()}
              onFileChange={handleFileUpload}
              onRecordToggle={isRecording ? stopRecording : startRecording}
            />

            <ModelSelection
              availableModels={availableModels}
              selectedModel={selectedModel}
              onModelChange={setSelectedModel}
            />

            <button
              onClick={analyzeEmotions}
              disabled={isAnalyzing}
              className="w-full bg-gradient-to-r from-violet-600 to-fuchsia-600 hover:from-violet-700 hover:to-fuchsia-700 disabled:from-slate-700 disabled:to-slate-700 text-white rounded-3xl p-4 font-semibold text-lg mb-8 transition-all transform hover:scale-[1.01] disabled:scale-100"
            >
              {isAnalyzing ? "Analyzing..." : "Analyze Emotions"}
            </button>
          </div>

          <div className="space-y-8">
            {emotionResults ? (
              <div className="space-y-4">
                <div className="flex gap-3">
                  <button
                    onClick={() => setShowSpectrogram(true)}
                    className={`flex-1 py-2 px-4 rounded-xl font-semibold transition-all ${
                      showSpectrogram
                        ? "bg-violet-600 text-white shadow-lg"
                        : "bg-slate-800 text-slate-300 hover:bg-slate-700"
                    }`}
                  >
                    Spectrogram
                  </button>
                  <button
                    onClick={() => setShowSpectrogram(false)}
                    className={`flex-1 py-2 px-4 rounded-xl font-semibold transition-all ${
                      !showSpectrogram
                        ? "bg-violet-600 text-white shadow-lg"
                        : "bg-slate-800 text-slate-300 hover:bg-slate-700"
                    }`}
                  >
                    Analysis
                  </button>
                </div>

                {showSpectrogram ? (
                  <SpectrogramResults
                    spectrogram={emotionResults.spectrogramData}
                    duration={emotionResults.duration}
                    sampleRate={emotionResults.sampleRate}
                    segments={emotionResults.segments}
                  />
                ) : (
                  <EmotionResults emotionResults={emotionResults} />
                )}
              </div>
            ) : (
              <div className="rounded-3xl bg-slate-950/90 p-8 border border-violet-600/20 text-slate-200 shadow-2xl shadow-purple-950/20">
                <h2 className="text-2xl font-semibold mb-3">
                  Ready to analyze
                </h2>
                <p className="text-slate-300 leading-relaxed">
                  Select or record audio in the left panel, choose a model, then
                  click Analyze Emotions. Results appear in the right panel.
                </p>
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}
