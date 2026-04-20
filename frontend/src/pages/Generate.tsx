import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { useEffect, useMemo, useState } from "react";

import { api } from "../api";
import { useSSE } from "../hooks/useSSE";
import type { InferEvent, InferJobState, InferRequest } from "../types";

const inputCls =
  "rounded bg-slate-800 border border-slate-700 px-3 py-1.5 text-sm focus:outline-none focus:border-blue-500";

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

function ProgressBar({ value, max }: { value: number; max: number }) {
  const pct = max > 0 ? Math.min(100, (value / max) * 100) : 0;
  return (
    <div className="w-full bg-slate-800 rounded h-2 overflow-hidden">
      <div className="bg-blue-500 h-full transition-all" style={{ width: `${pct}%` }} />
    </div>
  );
}

export default function Generate() {
  const qc = useQueryClient();
  const { data: modelsData } = useQuery({ queryKey: ["models"], queryFn: api.listModels });
  const models = modelsData?.models ?? [];

  const [form, setForm] = useState<InferRequest>({
    model_id: "",
    prompts: ["a photo of sks person, highly detailed"],
    negative_prompt: "low quality, blurry, distorted",
    num_inference_steps: 30,
    guidance_scale: 7.5,
    height: 512,
    width: 512,
    num_images_per_prompt: 1,
    seed: 42,
  });
  const [activeJobId, setActiveJobId] = useState<string | null>(null);

  useEffect(() => {
    if (!form.model_id && models.length > 0) {
      setForm((f) => ({ ...f, model_id: models[0].id }));
    }
  }, [models, form.model_id]);

  const update = <K extends keyof InferRequest>(k: K, v: InferRequest[K]) =>
    setForm((f) => ({ ...f, [k]: v }));

  const start = useMutation({
    mutationFn: (req: InferRequest) => api.generate(req),
    onSuccess: (state) => {
      setActiveJobId(state.id);
      qc.invalidateQueries({ queryKey: ["inference-job", state.id] });
    },
  });

  const { data: job } = useQuery<InferJobState>({
    queryKey: ["inference-job", activeJobId],
    queryFn: () => api.getInferenceJob(activeJobId!),
    enabled: !!activeJobId,
    refetchInterval: (query) => {
      const s = query.state.data?.status;
      return s && ["completed", "failed", "cancelled"].includes(s) ? false : 2000;
    },
  });

  const sseUrl = useMemo(
    () =>
      activeJobId && job && (job.status === "running" || job.status === "pending")
        ? api.inferenceEventsUrl(activeJobId)
        : null,
    [activeJobId, job?.status], // eslint-disable-line react-hooks/exhaustive-deps
  );
  const { events, state: sseState, terminal } = useSSE<InferEvent>(sseUrl);

  useEffect(() => {
    if (terminal) qc.invalidateQueries({ queryKey: ["inference-job", activeJobId] });
  }, [terminal, qc, activeJobId]);

  const imageEvents = useMemo(
    () => events.filter((e): e is Extract<InferEvent, { type: "image" }> => e.type === "image"),
    [events],
  );
  const producedCount = imageEvents.length;
  const totalImages = job?.total_images ?? form.prompts.length * (form.num_images_per_prompt ?? 1);

  const canSubmit = !!form.model_id && form.prompts.every((p) => p.trim().length > 0) && !start.isPending;

  const currentImages: string[] = useMemo(() => {
    if (imageEvents.length > 0) return imageEvents.map((e) => e.filename);
    return job?.images ?? [];
  }, [imageEvents, job?.images]);

  return (
    <div className="max-w-5xl space-y-6">
      <h1 className="text-2xl font-semibold">Generate</h1>

      {models.length === 0 && (
        <div className="rounded border border-amber-800 bg-amber-950/40 p-3 text-amber-200 text-sm">
          완성된 모델이 없습니다. 먼저 학습을 완료하세요.
        </div>
      )}

      <div className="rounded border border-slate-800 bg-slate-900/50 p-3 text-xs text-slate-400">
        ⚠️ 생성된 이미지는 로컬에만 저장됩니다. 실존 인물의 얼굴이나 상표 등
        타인 권리를 침해하는 용도로 생성·배포하지 마세요.
      </div>

      <form
        className="grid grid-cols-1 md:grid-cols-2 gap-4 rounded-lg border border-slate-800 bg-slate-900 p-4"
        onSubmit={(e) => {
          e.preventDefault();
          if (canSubmit) start.mutate(form);
        }}
      >
        <Field label="Model">
          <select
            className={inputCls}
            value={form.model_id}
            onChange={(e) => update("model_id", e.target.value)}
            required
          >
            <option value="">-- select --</option>
            {models.map((m) => (
              <option key={m.id} value={m.id}>
                {m.kind.toUpperCase()} · {m.id}
              </option>
            ))}
          </select>
        </Field>

        <Field label="Negative prompt">
          <input
            className={inputCls}
            value={form.negative_prompt ?? ""}
            onChange={(e) => update("negative_prompt", e.target.value)}
          />
        </Field>

        <div className="md:col-span-2">
          <Field
            label={`Prompts (${form.prompts.length})`}
            hint="Each line is a separate prompt (최대 8개)"
          >
            <textarea
              className={inputCls + " font-mono min-h-[6rem]"}
              value={form.prompts.join("\n")}
              onChange={(e) => {
                const lines = e.target.value.split("\n").slice(0, 8);
                update("prompts", lines);
              }}
            />
          </Field>
        </div>

        <Field label="Inference steps">
          <input
            type="number"
            min={1}
            max={150}
            className={inputCls}
            value={form.num_inference_steps}
            onChange={(e) => update("num_inference_steps", Number(e.target.value))}
          />
        </Field>

        <Field label="Guidance scale (CFG)">
          <input
            type="number"
            step={0.5}
            min={0}
            max={30}
            className={inputCls}
            value={form.guidance_scale}
            onChange={(e) => update("guidance_scale", Number(e.target.value))}
          />
        </Field>

        <Field label="Height">
          <select
            className={inputCls}
            value={form.height}
            onChange={(e) => update("height", Number(e.target.value))}
          >
            <option value={256}>256</option>
            <option value={512}>512</option>
            <option value={768}>768</option>
          </select>
        </Field>

        <Field label="Width">
          <select
            className={inputCls}
            value={form.width}
            onChange={(e) => update("width", Number(e.target.value))}
          >
            <option value={256}>256</option>
            <option value={512}>512</option>
            <option value={768}>768</option>
          </select>
        </Field>

        <Field label="Images per prompt">
          <input
            type="number"
            min={1}
            max={4}
            className={inputCls}
            value={form.num_images_per_prompt}
            onChange={(e) => update("num_images_per_prompt", Number(e.target.value))}
          />
        </Field>

        <Field label="Seed" hint="비워두면 매번 다른 결과">
          <input
            type="number"
            className={inputCls}
            value={form.seed ?? ""}
            onChange={(e) =>
              update("seed", e.target.value === "" ? null : Number(e.target.value))
            }
          />
        </Field>

        <div className="md:col-span-2 flex items-center justify-end gap-3 pt-2">
          {start.isError && (
            <span className="text-xs text-rose-400">{(start.error as Error).message}</span>
          )}
          <button
            type="submit"
            disabled={!canSubmit}
            className="px-4 py-2 rounded bg-blue-600 hover:bg-blue-500 disabled:opacity-50 text-sm font-medium"
          >
            {start.isPending ? "Starting…" : "Generate"}
          </button>
        </div>
      </form>

      {activeJobId && job && (
        <section className="space-y-3">
          <div className="flex items-center justify-between text-sm">
            <div className="flex items-center gap-3">
              <span className="font-mono text-slate-300">{activeJobId}</span>
              <span className="text-slate-400">status: {job.status}</span>
              <span className="text-slate-500 text-xs">stream: {sseState}</span>
              {job.error && <span className="text-rose-400">error: {job.error}</span>}
            </div>
            <span className="text-slate-400">
              {producedCount} / {totalImages}
            </span>
          </div>
          <ProgressBar value={producedCount} max={totalImages} />

          <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
            {currentImages.map((fname) => (
              <a
                key={fname}
                href={api.inferenceImageUrl(activeJobId, fname)}
                target="_blank"
                rel="noreferrer"
                className="group block rounded overflow-hidden border border-slate-800 bg-slate-900"
              >
                <img
                  src={api.inferenceImageUrl(activeJobId, fname)}
                  alt={fname}
                  className="w-full aspect-square object-cover group-hover:opacity-90"
                />
                <div className="px-2 py-1 text-[10px] font-mono text-slate-500 truncate">
                  {fname}
                </div>
              </a>
            ))}
            {currentImages.length === 0 && (
              <div className="col-span-full text-slate-500 text-xs text-center py-8">
                {job.status === "running" || job.status === "pending"
                  ? "첫 이미지를 기다리는 중…"
                  : "생성된 이미지 없음"}
              </div>
            )}
          </div>
        </section>
      )}
    </div>
  );
}
