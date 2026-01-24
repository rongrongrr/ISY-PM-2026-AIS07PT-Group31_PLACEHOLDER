import React, { useState, useRef } from "react";
import { analyzeAudio } from "../utils/api";
import AudioInput from "./AudioInput";
import ModelSelection from "./ModelSelection";
import SpectrogramResults from "./SpectrogramResults";

export default function AudioEmotionAnalyzer() {
  const [audioFile, setAudioFile] = useState(null);
  const [isRecording, setIsRecording] = useState(false);
  const [recordedBlob, setRecordedBlob] = useState(null);
  const [selectedModels, setSelectedModels] = useState({
    baseline: true,
    advanced: false,
    ensemble: false,
  });
  const [spectrograms, setSpectrograms] = useState([]);
  const [isGenerating, setIsGenerating] = useState(false);

  const mediaRecorderRef = useRef(null);
  const audioChunksRef = useRef([]);
  const fileInputRef = useRef(null);

  const models = [
    {
      id: "baseline",
      name: "Baseline Model",
      description: "Fast, basic emotion detection",
    },
    {
      id: "advanced",
      name: "Advanced Model",
      description: "Higher accuracy, slower processing",
    },
    {
      id: "ensemble",
      name: "Ensemble Model",
      description: "Multiple models combined",
    },
  ];

  const handleFileUpload = (e) => {
    const file = e.target.files[0];
    if (file && file.type.startsWith("audio/")) {
      setAudioFile(file);
      setRecordedBlob(null);
      setSpectrograms([]);
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
        const blob = new Blob(audioChunksRef.current, { type: "audio/wav" });
        setRecordedBlob(blob);
        setAudioFile(null);
        setSpectrograms([]);
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

  const toggleModel = (modelId) => {
    setSelectedModels((prev) => ({ ...prev, [modelId]: !prev[modelId] }));
  };

  const generateSpectrograms = async () => {
    const activeModels = Object.entries(selectedModels)
      .filter(([_, selected]) => selected)
      .map(([id]) => models.find((m) => m.id === id));

    if (activeModels.length === 0) {
      alert("Please select at least one model");
      return;
    }

    if (!audioFile && !recordedBlob) {
      alert("Please upload an audio file or record audio");
      return;
    }

    setIsGenerating(true);

    try {
      const results = await fetchSpectrogramsFromBackend(
        audioFile || recordedBlob,
        activeModels,
      );
      setSpectrograms(results);
    } catch (error) {
      console.error("Error generating spectrograms:", error);
      alert("Failed to generate spectrograms. Please try again.");
    } finally {
      setIsGenerating(false);
    }
  };

  const fetchSpectrogramsFromBackend = async (audio, models) => {
    const audioFile =
      audio instanceof Blob && !(audio instanceof File)
        ? new File([audio], "recording.wav", { type: "audio/wav" })
        : audio;
    const modelMap = {};
    models.forEach((m) => (modelMap[m.id] = true));
    const results = await analyzeAudio(audioFile, modelMap);
    return results;
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-900 via-purple-900 to-slate-900 p-8">
      <div className="max-w-6xl mx-auto">
        <h1 className="text-4xl font-bold text-white mb-2">
          Audio Emotion Analyzer
        </h1>
        <p className="text-purple-200 mb-8">
          Analyze emotional patterns in speech using AI models
        </p>

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
          models={models}
          selectedModels={selectedModels}
          toggleModel={toggleModel}
        />

        <button
          onClick={generateSpectrograms}
          disabled={isGenerating}
          className="w-full bg-gradient-to-r from-purple-600 to-pink-600 hover:from-purple-700 hover:to-pink-700 disabled:from-gray-600 disabled:to-gray-600 text-white rounded-xl p-4 font-semibold text-lg mb-8 transition-all transform hover:scale-[1.02] disabled:scale-100"
        >
          {isGenerating ? "Generating..." : "Generate Spectrograms"}
        </button>

        <SpectrogramResults spectrograms={spectrograms} />
      </div>
    </div>
  );
}
