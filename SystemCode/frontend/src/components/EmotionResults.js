import React from "react";

export default function EmotionResults({ emotionResults }) {
  if (!emotionResults) return null;

  const emotions =
    emotionResults.emotions ||
    Object.fromEntries(
      (emotionResults.emotionDistribution || []).map((item) => [
        item.label || item.emotion,
        item.confidence,
      ]),
    );
  const topEmotion =
    emotionResults.topEmotion || emotionResults.top_emotion || "";
  const confidence =
    emotionResults.confidence ?? emotionResults.confidence ?? 0;

  const emotionLabels = {
    ANG: "Angry",
    DIS: "Disgust",
    FEA: "Fear",
    HAP: "Happy",
    NEU: "Neutral",
    SAD: "Sad",
  };

  const emotionColors = {
    ANG: "#ef4444",
    DIS: "#10b981",
    FEA: "#8b5cf6",
    HAP: "#fbbf24",
    NEU: "#9ca3af",
    SAD: "#60a5fa",
  };

  const formatPercent = (value) => (value * 100).toFixed(1);

  return (
    <div className="space-y-3">
      {/* Row with Primary Emotion + Probabilities */}
      <div className="flex space-x-3">
        {/* Top Emotion Display */}
        <div className="bg-white/10 backdrop-blur-md rounded-xl p-4 border border-white/20 flex-1">
          <h3 className="text-xl font-semibold text-white mb-4">
            Primary Emotion
          </h3>
          <div className="text-center">
            <div
              className="inline-block px-4 py-2 rounded-lg text-white font-bold text-lg"
              style={{ backgroundColor: emotionColors[topEmotion] }}
            >
              {emotionLabels[topEmotion] || topEmotion}
            </div>
            <p className="text-purple-200 text-xs mt-2">
              Confidence: {formatPercent(confidence)}%
            </p>
          </div>
        </div>

        {/* Emotion Probabilities Chart */}
        <div className="bg-white/10 backdrop-blur-md rounded-xl p-4 border border-white/20 flex-1">
          <h3 className="text-sm font-semibold text-white mb-3">
            Probabilities
          </h3>
          <div className="space-y-2">
            {Object.entries(emotions).map(([emotion, probability]) => (
              <div key={emotion} className="flex items-center space-x-2">
                <div className="w-14 text-xs text-white font-medium">
                  {emotionLabels[emotion] || emotion}
                </div>
                <div className="flex-1 bg-slate-700 rounded-full h-3">
                  <div
                    className="h-3 rounded-full transition-all duration-500"
                    style={{
                      width: `${Math.min(100, probability * 100)}%`,
                      backgroundColor: emotionColors[emotion],
                    }}
                  ></div>
                </div>
                <div className="w-10 text-xs text-white text-right">
                  {formatPercent(probability)}%
                </div>
              </div>
            ))}
          </div>
        </div>
      </div>

      {/* Emotion Breakdown stays below */}
      <div className="bg-white/10 backdrop-blur-md rounded-xl p-4 border border-white/20">
        <h3 className="text-sm font-semibold text-white mb-3">Breakdown</h3>
        <div className="grid grid-cols-3 gap-2">
          {Object.entries(emotions).map(([emotion, probability]) => (
            <div key={emotion} className="text-center">
              <div
                className="w-12 h-12 rounded-full mx-auto mb-1 flex items-center justify-center text-white font-bold text-sm"
                style={{ backgroundColor: emotionColors[emotion] }}
              >
                {formatPercent(probability)}%
              </div>
              <p className="text-white text-xs">
                {emotionLabels[emotion] || emotion}
              </p>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}
