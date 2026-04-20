// Mirror of backend/api/schemas.py — keep in sync.

export interface GPUInfo {
  available: boolean;
  name?: string | null;
  driver_version?: string | null;
  memory_total_mb?: number | null;
  memory_used_mb?: number | null;
  memory_free_mb?: number | null;
  utilization_percent?: number | null;
  temperature_c?: number | null;
  error?: string | null;
}

export interface DatasetInfo {
  id: string;
  name: string;
  image_count: number;
  total_bytes: number;
  created_at: number;
}

export type Preset = "person" | "object" | "style" | "fast" | "high_quality";

export type JobStatus = "pending" | "running" | "completed" | "failed" | "cancelled";

export interface JobState {
  id: string;
  status: JobStatus;
  created_at: number;
  started_at?: number | null;
  finished_at?: number | null;
  pid?: number | null;
  return_code?: number | null;
  error?: string | null;
  dataset_id?: string | null;
  output_dir?: string | null;
  event_log?: string | null;
  max_train_steps?: number | null;
  latest_step?: number | null;
  latest_loss?: number | null;
}

export interface ModelInfo {
  id: string;
  job_id?: string | null;
  path: string;
  kind: "lora" | "full";
  created_at: number;
}

export interface TrainStartRequest {
  dataset_id: string;
  preset?: Preset;
  instance_prompt: string;
  class_prompt?: string;
  max_train_steps?: number;
  learning_rate?: number;
  resolution?: number;
  use_lora?: boolean;
  lora_rank?: number;
  lora_alpha?: number;
  mixed_precision?: "no" | "fp16" | "bf16";
  enable_vae_slicing?: boolean;
  enable_vae_tiling?: boolean;
  cpu_offload_text_encoder?: boolean;
  with_prior_preservation?: boolean;
  seed?: number;
  deterministic?: boolean;
  pretrained_model_name_or_path?: string;
}

// Training events (JSONL → SSE)
export interface TrainEventBase {
  ts: number;
  type: string;
  step: number;
}

export interface StepEvent extends TrainEventBase {
  type: "step";
  loss: number;
  lr: number;
  elapsed: number;
  max_steps: number;
}

export interface StartedEvent extends TrainEventBase {
  type: "started";
  max_train_steps: number;
  output_dir: string;
  use_lora: boolean;
  mixed_precision: string;
  resolution: number;
  instance_prompt: string;
}

export interface ValidationEvent extends TrainEventBase {
  type: "validation";
  images: string[];
}

export interface CheckpointEvent extends TrainEventBase {
  type: "checkpoint";
  path: string;
}

export interface CompletedEvent extends TrainEventBase {
  type: "completed";
  elapsed: number;
  output_dir: string;
}

export interface ErrorEvent extends TrainEventBase {
  type: "error";
  error: string;
  error_type?: string;
}

export type TrainEvent =
  | StartedEvent
  | StepEvent
  | ValidationEvent
  | CheckpointEvent
  | CompletedEvent
  | ErrorEvent
  | TrainEventBase;

// ---------- Inference ----------

export interface InferRequest {
  model_id: string;
  prompts: string[];
  negative_prompt?: string | null;
  num_inference_steps?: number;
  guidance_scale?: number;
  height?: number;
  width?: number;
  num_images_per_prompt?: number;
  seed?: number | null;
}

export interface InferJobState {
  id: string;
  status: JobStatus;
  model_id: string;
  created_at: number;
  started_at?: number | null;
  finished_at?: number | null;
  pid?: number | null;
  return_code?: number | null;
  error?: string | null;
  output_dir?: string | null;
  event_log?: string | null;
  prompts: string[];
  images: string[];
  total_images: number;
}

// Inference events (JSONL → SSE)
export interface InferStartedEvent {
  ts: number;
  type: "started";
  model_id: string;
  prompts: string[];
  total: number;
}

export interface InferImageEvent {
  ts: number;
  type: "image";
  index: number;
  total: number;
  filename: string;
  prompt: string;
  prompt_index: number;
  seed: number;
  elapsed: number;
}

export interface InferCompletedEvent {
  ts: number;
  type: "completed";
  total: number;
  elapsed: number;
}

export interface InferErrorEvent {
  ts: number;
  type: "error";
  error: string;
  error_type?: string;
}

export type InferEvent =
  | InferStartedEvent
  | InferImageEvent
  | InferCompletedEvent
  | InferErrorEvent
  | { ts: number; type: string; [k: string]: any };
