import React from "react";

export default function ModelSelection({
  availableModels,
  selectedModel,
  onModelChange,
}) {
  return (
    <div className="rounded-2xl bg-slate-900/80 p-6 border border-violet-500/20 shadow-lg">
      <label
        htmlFor="model-select"
        className="block text-lg font-semibold text-white mb-3"
      >
        Select Model
      </label>
      <select
        id="model-select"
        value={selectedModel || ""}
        onChange={(e) => onModelChange(e.target.value)}
        disabled={availableModels.length === 0}
        className="w-full bg-slate-950 text-white border border-violet-500/30 rounded-xl p-3 font-medium cursor-pointer hover:border-violet-400/50 transition-colors focus:outline-none focus:ring-2 focus:ring-violet-500/50 disabled:opacity-50 disabled:cursor-not-allowed"
      >
        {availableModels.length === 0 ? (
          <option value="">No models available</option>
        ) : (
          availableModels.map((model) => (
            <option key={model.id} value={model.id}>
              {model.name}
            </option>
          ))
        )}
      </select>
      {availableModels.length > 0 && selectedModel && (
        <p className="text-sm text-slate-400 mt-3">
          {availableModels.find((m) => m.id === selectedModel)?.description ||
            ""}
        </p>
      )}
      {availableModels.length > 0 && !selectedModel && (
        <p className="text-xs text-violet-400 mt-2">✓ Models loaded</p>
      )}
    </div>
  );
}
