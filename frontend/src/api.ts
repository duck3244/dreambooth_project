import type {
  DatasetInfo,
  GPUInfo,
  InferJobState,
  InferRequest,
  JobState,
  ModelInfo,
  TrainStartRequest,
} from "./types";

const BASE = "/api";

class ApiError extends Error {
  status: number;
  constructor(status: number, message: string) {
    super(message);
    this.status = status;
  }
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(BASE + path, init);
  if (!res.ok) {
    let detail = res.statusText;
    try {
      const body = await res.json();
      detail = typeof body?.detail === "string" ? body.detail : JSON.stringify(body);
    } catch {
      // keep statusText
    }
    throw new ApiError(res.status, detail);
  }
  if (res.status === 204) return undefined as T;
  return (await res.json()) as T;
}

export const api = {
  gpu: () => request<GPUInfo>("/gpu"),
  health: () => request<{ ok: boolean }>("/health"),

  // Datasets
  listDatasets: () => request<{ datasets: DatasetInfo[] }>("/datasets"),
  getDataset: (id: string) => request<DatasetInfo>(`/datasets/${id}`),
  createDataset: async (name: string): Promise<{ id: string; name: string }> => {
    const form = new FormData();
    form.append("name", name);
    return request("/datasets", { method: "POST", body: form });
  },
  uploadImages: async (datasetId: string, files: File[]) => {
    const form = new FormData();
    for (const f of files) form.append("files", f, f.name);
    return request<{ id: string; added: string[]; skipped: string[]; image_count: number }>(
      `/datasets/${datasetId}/images`,
      { method: "POST", body: form },
    );
  },
  deleteDataset: (id: string) => request<{ ok: true }>(`/datasets/${id}`, { method: "DELETE" }),

  // Training
  listJobs: () => request<{ jobs: JobState[] }>("/train"),
  getJob: (id: string) => request<JobState>(`/train/${id}`),
  startTraining: (req: TrainStartRequest) =>
    request<JobState>("/train", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(req),
    }),
  stopJob: (id: string) =>
    request<JobState>(`/train/${id}/stop`, { method: "POST" }),
  deleteJob: (id: string) =>
    request<{ ok: true }>(`/train/${id}`, { method: "DELETE" }),
  eventsUrl: (id: string) => `${BASE}/train/${id}/events`,

  // Models
  listModels: () => request<{ models: ModelInfo[] }>("/models"),

  // Inference
  generate: (req: InferRequest) =>
    request<InferJobState>("/inference/generate", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(req),
    }),
  listInferenceJobs: () => request<{ jobs: InferJobState[] }>("/inference"),
  getInferenceJob: (id: string) => request<InferJobState>(`/inference/${id}`),
  stopInference: (id: string) =>
    request<InferJobState>(`/inference/${id}/stop`, { method: "POST" }),
  deleteInference: (id: string) =>
    request<{ ok: true }>(`/inference/${id}`, { method: "DELETE" }),
  inferenceEventsUrl: (id: string) => `${BASE}/inference/${id}/events`,
  inferenceImageUrl: (id: string, filename: string) =>
    `${BASE}/inference/${id}/images/${encodeURIComponent(filename)}`,
};

export { ApiError };
