import React from "react";

export default function ModelSelection({
  models,
  selectedModels,
  toggleModel,
}) {
  return (
    <div className="bg-white/10 backdrop-blur-md rounded-xl p-6 mb-6 border border-white/20">
      <h2 className="text-xl font-semibold text-white mb-4">Select Models</h2>

      <div className="space-y-3">
        {models.map((model) => (
          <label
            key={model.id}
            className="flex items-start gap-3 p-4 bg-white/5 rounded-lg cursor-pointer hover:bg-white/10 transition-colors"
          >
            <input
              type="checkbox"
              checked={selectedModels[model.id]}
              onChange={() => toggleModel(model.id)}
              className="mt-1 w-5 h-5 rounded border-white/30 text-purple-600 focus:ring-purple-500"
            />
            <div className="flex-1">
              <div className="text-white font-medium">{model.name}</div>
              <div className="text-purple-200 text-sm">{model.description}</div>
            </div>
          </label>
        ))}
      </div>
    </div>
  );
}
