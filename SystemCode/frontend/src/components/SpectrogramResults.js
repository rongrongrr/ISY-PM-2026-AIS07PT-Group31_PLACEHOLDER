import React from "react";

export default function SpectrogramResults({ spectrograms }) {
  if (!spectrograms || spectrograms.length === 0) return null;

  return (
    <div className="space-y-6">
      <h2 className="text-2xl font-bold text-white">Results</h2>

      {spectrograms.map((spec, idx) => (
        <div
          key={idx}
          className="bg-white/10 backdrop-blur-md rounded-xl p-6 border border-white/20"
        >
          <h3 className="text-lg font-semibold text-white mb-4">
            {spec.modelName}
          </h3>

          <div className="bg-slate-900 rounded-lg p-4 mb-4">
            <svg width="100%" height="280" className="rounded">
              <defs>
                <linearGradient
                  id={`grad-${idx}`}
                  x1="0%"
                  y1="0%"
                  x2="0%"
                  y2="100%"
                >
                  <stop
                    offset="0%"
                    style={{ stopColor: "#8b5cf6", stopOpacity: 0.8 }}
                  />
                  <stop
                    offset="50%"
                    style={{ stopColor: "#ec4899", stopOpacity: 0.6 }}
                  />
                  <stop
                    offset="100%"
                    style={{ stopColor: "#3b82f6", stopOpacity: 0.4 }}
                  />
                </linearGradient>
              </defs>

              {(spec.spectrogramData || []).map((height, i) => {
                const x = (i / spec.spectrogramData.length) * 100;
                const normalizedHeight = height;
                const y = 100 - normalizedHeight / 2;

                return (
                  <rect
                    key={i}
                    x={`${x}%`}
                    y={y}
                    width={`${0.8 / (spec.spectrogramData.length / 100)}%`}
                    height={normalizedHeight}
                    fill={`url(#grad-${idx})`}
                    opacity={0.3 + Math.random() * 0.5}
                  />
                );
              })}

              {[0, 25, 50, 75, 100].map((y) => (
                <line
                  key={y}
                  x1="0"
                  y1={y * 2}
                  x2="100%"
                  y2={y * 2}
                  stroke="rgba(255,255,255,0.1)"
                  strokeWidth="1"
                />
              ))}

              <text x="5" y="15" fill="rgba(255,255,255,0.5)" fontSize="10">
                High
              </text>
              <text x="5" y="105" fill="rgba(255,255,255,0.5)" fontSize="10">
                Mid
              </text>
              <text x="5" y="195" fill="rgba(255,255,255,0.5)" fontSize="10">
                Low
              </text>

              <line
                x1="0"
                y1="210"
                x2="100%"
                y2="210"
                stroke="rgba(255,255,255,0.3)"
                strokeWidth="2"
              />

              {Array.from({ length: Math.ceil(spec.duration) + 1 }).map(
                (_, i) => {
                  const x = (i / spec.duration) * 100;
                  return (
                    <g key={i}>
                      <line
                        x1={`${x}%`}
                        y1="210"
                        x2={`${x}%`}
                        y2="215"
                        stroke="rgba(255,255,255,0.5)"
                        strokeWidth="1"
                      />
                      <text
                        x={`${x}%`}
                        y="227"
                        fill="rgba(255,255,255,0.6)"
                        fontSize="10"
                        textAnchor="middle"
                      >
                        {i}s
                      </text>
                    </g>
                  );
                },
              )}

              {spec.emotions.map((emotion, i) => {
                const startX = (emotion.start / spec.duration) * 100;
                const endX = (emotion.end / spec.duration) * 100;
                const centerX = (startX + endX) / 2;
                const width = endX - startX;
                const label =
                  emotion.label ?? emotion.emotion ?? emotion.name ?? "Unknown";
                const confidence =
                  typeof emotion.confidence === "number"
                    ? emotion.confidence
                    : (emotion.confidence ?? null);

                return (
                  <g key={i}>
                    <rect
                      x={`${startX}%`}
                      y="235"
                      width={`${width}%`}
                      height="12"
                      fill={getEmotionColor(label)}
                      opacity={0.35}
                    />
                    <text
                      x={`${centerX}%`}
                      y="244"
                      fill={getEmotionColor(label)}
                      fontSize="12"
                      textAnchor="middle"
                    >
                      {label}
                    </text>
                    {confidence != null && (
                      <text
                        x={`${centerX}%`}
                        y="258"
                        fill={hexToRgba(getEmotionColor(label), 0.85)}
                        fontSize="10"
                        textAnchor="middle"
                      >
                        {(confidence * 100).toFixed(0)}%
                      </text>
                    )}
                  </g>
                );
              })}
            </svg>
          </div>
        </div>
      ))}
    </div>
  );
}

// Helper (kept here for readability)
function getEmotionColor(emotion) {
  const colors = {
    Happy: "#fbbf24",
    Sad: "#60a5fa",
    Angry: "#ef4444",
    Neutral: "#9ca3af",
    Surprised: "#a78bfa",
    Fear: "#8b5cf6",
    Disgust: "#10b981",
  };
  return colors[emotion] || "#9ca3af";
}

// convert hex like #rrggbb to rgba() with given alpha
function hexToRgba(hex, alpha = 1) {
  const h = hex.replace("#", "");
  const bigint = parseInt(h, 16);
  const r = (bigint >> 16) & 255;
  const g = (bigint >> 8) & 255;
  const b = bigint & 255;
  return `rgba(${r}, ${g}, ${b}, ${alpha})`;
}
