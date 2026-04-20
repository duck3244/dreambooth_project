import { useMutation, useQuery } from "@tanstack/react-query";
import { useState } from "react";
import { useNavigate } from "react-router-dom";

import { api } from "../api";
import type { Preset, TrainStartRequest } from "../types";

const PRESETS: { value: Preset; label: string; steps: number }[] = [
  { value: "person", label: "Person (400 steps)", steps: 400 },
  { value: "object", label: "Object (600 steps)", steps: 600 },
  { value: "style", label: "Style (800 steps, lr=1e-6)", steps: 800 },
  { value: "fast", label: "Fast (200 steps, 256px)", steps: 200 },
  { value: "high_quality", label: "High quality (800 steps)", steps: 800 },
];

function Field({
  label,
  hint,
  children,
}: {
  label: string;
  hint?: string;
  children: React.ReactNode;
}) {
  return (
    <label className="flex flex-col gap-1">
      <span className="text-sm text-slate-300">{label}</span>
      {children}
      {hint && <span className="text-xs text-slate-500">{hint}</span>}
    </label>
  );
}

const inputCls =
  "rounded bg-slate-800 border border-slate-700 px-3 py-1.5 text-sm focus:outline-none focus:border-blue-500";

export default function TrainNew() {
  const nav = useNavigate();
  const { data: dsData } = useQuery({ queryKey: ["datasets"], queryFn: api.listDatasets });
  const datasets = dsData?.datasets ?? [];

  const [form, setForm] = useState<TrainStartRequest>({
    dataset_id: "",
    preset: "person",
    instance_prompt: "a photo of sks person",
    max_train_steps: 400,
    learning_rate: 1e-6,
    resolution: 512,
    use_lora: true,
    lora_rank: 4,
    lora_alpha: 4,
    mixed_precision: "fp16",
    enable_vae_slicing: true,
    enable_vae_tiling: false,
    cpu_offload_text_encoder: false,
    with_prior_preservation: false,
    seed: 42,
    deterministic: false,
  });

  const update = <K extends keyof TrainStartRequest>(k: K, v: TrainStartRequest[K]) =>
    setForm((f) => ({ ...f, [k]: v }));

  const start = useMutation({
    mutationFn: (req: TrainStartRequest) => api.startTraining(req),
    onSuccess: (state) => nav(`/train/${state.id}`),
  });

  const canSubmit = !!form.dataset_id && !!form.instance_prompt?.trim();

  return (
    <div className="max-w-3xl space-y-4">
      <h1 className="text-2xl font-semibold">Start new training</h1>

      {datasets.length === 0 && (
        <div className="rounded border border-amber-800 bg-amber-950/40 p-3 text-amber-200 text-sm">
          Dataset이 없습니다. 먼저{" "}
          <button
            className="underline"
            onClick={() => nav("/datasets")}
          >
            Datasets 페이지
          </button>
          에서 이미지를 업로드하세요.
        </div>
      )}

      <form
        className="grid grid-cols-1 md:grid-cols-2 gap-4 rounded-lg border border-slate-800 bg-slate-900 p-4"
        onSubmit={(e) => {
          e.preventDefault();
          if (canSubmit) start.mutate(form);
        }}
      >
        <Field label="Dataset">
          <select
            className={inputCls}
            value={form.dataset_id}
            onChange={(e) => update("dataset_id", e.target.value)}
            required
          >
            <option value="">-- select --</option>
            {datasets.map((d) => (
              <option key={d.id} value={d.id}>
                {d.name} ({d.image_count} imgs)
              </option>
            ))}
          </select>
        </Field>

        <Field label="Preset" hint="Preset adjusts default step count">
          <select
            className={inputCls}
            value={form.preset}
            onChange={(e) => {
              const p = e.target.value as Preset;
              const preset = PRESETS.find((x) => x.value === p);
              update("preset", p);
              if (preset) update("max_train_steps", preset.steps);
            }}
          >
            {PRESETS.map((p) => (
              <option key={p.value} value={p.value}>
                {p.label}
              </option>
            ))}
          </select>
        </Field>

        <Field label="Instance prompt" hint='예: "a photo of sks person"'>
          <input
            className={inputCls}
            value={form.instance_prompt ?? ""}
            onChange={(e) => update("instance_prompt", e.target.value)}
            required
          />
        </Field>

        <Field label="Class prompt (prior preservation 용, 선택)">
          <input
            className={inputCls}
            value={form.class_prompt ?? ""}
            onChange={(e) => update("class_prompt", e.target.value)}
            placeholder="a photo of person"
          />
        </Field>

        <Field label="Max train steps">
          <input
            type="number"
            min={1}
            max={5000}
            className={inputCls}
            value={form.max_train_steps}
            onChange={(e) => update("max_train_steps", Number(e.target.value))}
          />
        </Field>

        <Field label="Learning rate">
          <input
            type="number"
            step={1e-7}
            className={inputCls}
            value={form.learning_rate}
            onChange={(e) => update("learning_rate", Number(e.target.value))}
          />
        </Field>

        <Field label="Resolution">
          <select
            className={inputCls}
            value={form.resolution}
            onChange={(e) => update("resolution", Number(e.target.value))}
          >
            <option value={256}>256</option>
            <option value={512}>512</option>
            <option value={768}>768</option>
          </select>
        </Field>

        <Field label="Mixed precision">
          <select
            className={inputCls}
            value={form.mixed_precision}
            onChange={(e) =>
              update("mixed_precision", e.target.value as "no" | "fp16" | "bf16")
            }
          >
            <option value="fp16">fp16</option>
            <option value="bf16">bf16</option>
            <option value="no">no</option>
          </select>
        </Field>

        <Field label="LoRA">
          <div className="flex items-center gap-2 mt-1">
            <input
              type="checkbox"
              id="use_lora"
              checked={!!form.use_lora}
              onChange={(e) => update("use_lora", e.target.checked)}
            />
            <label htmlFor="use_lora" className="text-sm text-slate-300">
              Enable LoRA (권장, 메모리↓/속도↑)
            </label>
          </div>
        </Field>

        {form.use_lora && (
          <Field label="LoRA rank">
            <input
              type="number"
              min={1}
              max={128}
              className={inputCls}
              value={form.lora_rank}
              onChange={(e) => update("lora_rank", Number(e.target.value))}
            />
          </Field>
        )}

        <Field label="VAE slicing">
          <div className="flex items-center gap-2 mt-1">
            <input
              type="checkbox"
              id="vae_slicing"
              checked={!!form.enable_vae_slicing}
              onChange={(e) => update("enable_vae_slicing", e.target.checked)}
            />
            <label htmlFor="vae_slicing" className="text-sm text-slate-300">
              Enable (메모리 절약)
            </label>
          </div>
        </Field>

        <Field label="Text encoder CPU offload">
          <div className="flex items-center gap-2 mt-1">
            <input
              type="checkbox"
              id="cpu_offload"
              checked={!!form.cpu_offload_text_encoder}
              onChange={(e) => update("cpu_offload_text_encoder", e.target.checked)}
            />
            <label htmlFor="cpu_offload" className="text-sm text-slate-300">
              Offload (고정 프롬프트 사전 인코딩)
            </label>
          </div>
        </Field>

        <Field label="Seed">
          <input
            type="number"
            className={inputCls}
            value={form.seed}
            onChange={(e) => update("seed", Number(e.target.value))}
          />
        </Field>

        <div className="md:col-span-2 flex items-center justify-end gap-3 pt-2">
          {start.isError && (
            <span className="text-xs text-rose-400">
              {(start.error as Error).message}
            </span>
          )}
          <button
            type="submit"
            disabled={!canSubmit || start.isPending}
            className="px-4 py-2 rounded bg-blue-600 hover:bg-blue-500 disabled:opacity-50 text-sm font-medium"
          >
            {start.isPending ? "Starting…" : "Start training"}
          </button>
        </div>
      </form>
    </div>
  );
}
