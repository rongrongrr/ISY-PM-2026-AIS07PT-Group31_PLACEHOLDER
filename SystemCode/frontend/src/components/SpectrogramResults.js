import React from "react";

const MAX_DISPLAY_COLUMNS = 220;

const segmentLegend = {
  Happy: "#fbbf24",
  Sad: "#60a5fa",
  Angry: "#ef4444",
  Neutral: "#9ca3af",
  Surprised: "#a78bfa",
  Fear: "#8b5cf6",
  Disgust: "#10b981",
};

export default function SpectrogramResults({
  spectrogram,
  duration,
  sampleRate,
  segments = [],
}) {
  if (!Array.isArray(spectrogram) || spectrogram.length === 0) {
    return null;
  }

  const numRows = spectrogram.length;
  const numCols = spectrogram[0]?.length || 0;
  const displayCols = Math.min(MAX_DISPLAY_COLUMNS, numCols);
  const columnStep = Math.max(1, Math.floor(numCols / displayCols));

  const downsampledSpectrogram = spectrogram.map((row) => {
    const columns = [];
    for (let i = 0; i < displayCols; i += 1) {
      const index = Math.min(i * columnStep, numCols - 1);
      columns.push(row[index] ?? 0);
    }
    return columns;
  });

  const tickCount = 5;
  const ticks = Array.from({ length: tickCount }, (_, index) => {
    const time = (duration * index) / (tickCount - 1);
    const x = Math.round((displayCols * index) / (tickCount - 1));
    return { time, x };
  });

  return (
    <div className="bg-white/10 backdrop-blur-md rounded-xl p-5 border border-white/20">
      <div className="flex flex-col gap-2">
        <div>
          <p className="text-xs text-slate-400 mt-0">
            {numRows} bins × {numCols} frames @ {sampleRate} Hz
          </p>
        </div>

        <div className="rounded-lg overflow-hidden bg-slate-950 shadow-inner">
          <svg
            viewBox={`0 0 ${displayCols} ${numRows}`}
            preserveAspectRatio="xMidYMid slice"
            className="w-full h-[250px]"
          >
            {downsampledSpectrogram.map((row, rowIdx) =>
              row.map((value, colIdx) => (
                <rect
                  key={`${rowIdx}-${colIdx}`}
                  x={colIdx}
                  y={rowIdx}
                  width="1"
                  height="1"
                  fill={valueToColor(value)}
                />
              )),
            )}

            {segments.map((segment, index) => {
              const startX = Math.round(
                (segment.start / duration) * displayCols,
              );
              const endX = Math.round((segment.end / duration) * displayCols);
              return (
                <rect
                  key={`segment-${index}`}
                  x={Math.max(0, startX)}
                  y={numRows - 4}
                  width={Math.max(1, endX - startX)}
                  height="4"
                  fill={segmentColor(segment.emotion)}
                  opacity="0.7"
                />
              );
            })}
          </svg>

          {/* Time axis labels rendered as HTML */}
          <div className="flex justify-between px-1 pt-1 text-xs text-slate-400">
            {ticks.map((tick, index) => (
              <span key={`label-${index}`} className="flex-1 text-center">
                {tick.time.toFixed(1)}s
              </span>
            ))}
          </div>
        </div>

        <div className="grid gap-2 md:grid-cols-2">
          <div className="rounded-lg bg-slate-900/80 p-2 border border-violet-500/10">
            <h3 className="text-xs font-semibold text-white">Summary</h3>
            <p className="text-xs text-slate-300 mt-1">
              Duration: <strong>{formatSeconds(duration)}</strong>
            </p>
            <p className="text-xs text-slate-300">
              Rate: <strong>{sampleRate} Hz</strong>
            </p>
          </div>

          <div className="rounded-lg bg-slate-900/80 p-2 border border-violet-500/10">
            <h3 className="text-xs font-semibold text-white">Legend</h3>
            <div className="mt-1 grid gap-1 sm:grid-cols-2">
              {Object.entries(segmentLegend).map(([emotion, color]) => (
                <div key={emotion} className="flex items-center gap-1">
                  <span
                    className="h-2 w-2 rounded-full"
                    style={{ backgroundColor: color }}
                  />
                  <span className="text-xs text-slate-300">{emotion}</span>
                </div>
              ))}
            </div>
          </div>
        </div>

        <div className="rounded-lg bg-slate-900/80 p-2 border border-white/10">
          <h3 className="text-xs font-semibold text-white">Segments</h3>
          {segments.length > 0 ? (
            <div className="space-y-1 mt-1 max-h-[80px] overflow-y-auto">
              {segments.map((segment, index) => (
                <div
                  key={index}
                  className="rounded text-xs bg-slate-800/90 p-1 border border-white/10"
                >
                  <p className="text-slate-300">
                    <strong>{segment.emotion}</strong> {segment.start}-
                    {segment.end}s
                  </p>
                </div>
              ))}
            </div>
          ) : (
            <p className="text-xs text-slate-400 mt-1">No segments</p>
          )}
        </div>
      </div>
    </div>
  );
}

function valueToColor(value) {
  const clipped = Math.max(0, Math.min(1, value));
  const hue = 240 - clipped * 240;
  const lightness = 30 + clipped * 35;
  return `hsl(${hue}, 85%, ${lightness}%)`;
}

function segmentColor(emotion) {
  return segmentLegend[emotion] || "#7c3aed";
}

function formatSeconds(seconds) {
  if (typeof seconds !== "number" || Number.isNaN(seconds)) {
    return "0.00 s";
  }
  return `${seconds.toFixed(2)} s`;
}
