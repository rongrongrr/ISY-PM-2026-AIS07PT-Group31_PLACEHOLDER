const API_BASE_URL = "http://localhost:8000/api";

export async function analyzeAudio(audioFile, selectedModels) {
  const formData = new FormData();
  formData.append("audio", audioFile);

  const modelIds = Object.entries(selectedModels)
    .filter(([_, selected]) => selected)
    .map(([id, _]) => id);

  formData.append("models", JSON.stringify(modelIds));

  const response = await fetch(`${API_BASE_URL}/analyze`, {
    method: "POST",
    body: formData,
  });

  if (!response.ok) {
    throw new Error("Backend request failed");
  }

  const data = await response.json();
  return data.spectrograms;
}
