const API_BASE_URL = "http://localhost:8000/api";

export async function fetchAvailableModels() {
  try {
    console.log("Fetching available models from:", `${API_BASE_URL}/models`);
    const response = await fetch(`${API_BASE_URL}/models`);

    if (!response.ok) {
      console.error("Models endpoint returned status:", response.status);
      throw new Error(`HTTP ${response.status}`);
    }

    const data = await response.json();
    console.log("Models fetched successfully:", data);

    if (!data.models || data.models.length === 0) {
      console.warn("No models returned from backend, using default model");
      return [
        {
          id: "model_resnet50.pth",
          name: "ResNet50 v1",
          description: "Pretrained ResNet50 on CREMA-D dataset",
        },
      ];
    }

    return data.models;
  } catch (error) {
    console.error("Error fetching models:", error);
    // Return a default model as fallback
    return [
      {
        id: "model_resnet_50.pth",
        name: "ResNet50 v1",
        description: "Pretrained ResNet50 on CREMA-D dataset",
      },
    ];
  }
}

export async function analyzeAudio(audioFile, selectedModels) {
  const formData = new FormData();
  formData.append("audio", audioFile);

  const modelIds = Object.entries(selectedModels)
    .filter(([_, selected]) => selected)
    .map(([id]) => id);

  formData.append("models", JSON.stringify(modelIds));

  const response = await fetch(`${API_BASE_URL}/analyze`, {
    method: "POST",
    body: formData,
  });

  if (!response.ok) {
    throw new Error("Backend request failed");
  }

  const data = await response.json();
  return data;
}

export async function analyzeEmotion(audioFile) {
  const formData = new FormData();
  formData.append("audio", audioFile);

  const response = await fetch(`${API_BASE_URL}/analyze-emotion`, {
    method: "POST",
    body: formData,
  });

  if (!response.ok) {
    throw new Error("Backend request failed");
  }

  const data = await response.json();
  return data;
}
