import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { useEffect, useMemo } from "react";
import { useParams } from "react-router-dom";

import { api } from "../api";
import JobStatusBadge from "../components/JobStatusBadge";
import LossChart from "../components/LossChart";
import { useSSE } from "../hooks/useSSE";
import type { JobState, TrainEvent } from "../types";

function ProgressBar({ value, max }: { value: number; max: number }) {
  const pct = max > 0 ? Math.min(100, (value / max) * 100) : 0;
  return (
    <div className="w-full bg-slate-800 rounded h-2 overflow-hidden">
      <div className="bg-blue-500 h-full transition-all" style={{ width: `${pct}%` }} />
    </div>
  );
}

function fmtDuration(sec: number): string {
  if (!Number.isFinite(sec)) return "—";
  const h = Math.floor(sec / 3600);
  const m = Math.floor((sec % 3600) / 60);
  const s = Math.floor(sec % 60);
  return h > 0 ? `${h}h ${m}m ${s}s` : m > 0 ? `${m}m ${s}s` : `${s}s`;
}

function EventRow({ ev }: { ev: TrainEvent }) {
  const ts = new Date(ev.ts * 1000).toLocaleTimeString();
  const payload = { ...ev } as any;
  delete payload.type;
  delete payload.ts;
  return (
    <div className="grid grid-cols-[auto_auto_1fr] gap-3 text-xs py-1 border-b border-slate-800/60">
      <span className="text-slate-500 font-mono">{ts}</span>
      <span className="text-slate-300 font-medium">{ev.type}</span>
      <span className="text-slate-400 font-mono truncate">{JSON.stringify(payload)}</span>
    </div>
  );
}

export default function TrainLive() {
  const { jobId } = useParams<{ jobId: string }>();
  const qc = useQueryClient();
  const { data: job } = useQuery<JobState>({
    queryKey: ["job", jobId],
    queryFn: () => api.getJob(jobId!),
    enabled: !!jobId,
    refetchInterval: (query) => {
      const s = query.state.data?.status;
      return s && ["completed", "failed", "cancelled"].includes(s) ? false : 3000;
    },
  });

  const sseUrl = useMemo(
    () =>
      jobId && job && (job.status === "running" || job.status === "pending")
        ? api.eventsUrl(jobId)
        : null,
    [jobId, job?.status], // eslint-disable-line react-hooks/exhaustive-deps
  );

  const { events, state, terminal } = useSSE<TrainEvent>(sseUrl);

  // When the stream produces a terminal event, refresh the job state (so
  // status/return_code update immediately without waiting on the poll).
  useEffect(() => {
    if (terminal) qc.invalidateQueries({ queryKey: ["job", jobId] });
  }, [terminal, qc, jobId]);

  const stop = useMutation({
    mutationFn: () => api.stopJob(jobId!),
    onSuccess: () => qc.invalidateQueries({ queryKey: ["job", jobId] }),
  });

  if (!job) {
    return <div className="text-slate-400">Loading…</div>;
  }

  // Pull latest step/loss from events if fresher than stored state.
  const latestStep =
    events.find((e) => e.type === "step" && e === events[events.length - 1])?.step ??
    job.latest_step ??
    0;
  const lastStepEvent = [...events].reverse().find((e) => e.type === "step") as
    | (TrainEvent & { loss: number; elapsed: number; max_steps: number })
    | undefined;
  const currentStep = lastStepEvent?.step ?? latestStep;
  const currentLoss = lastStepEvent?.loss ?? job.latest_loss ?? null;
  const maxSteps = lastStepEvent?.max_steps ?? job.max_train_steps ?? 1;
  const elapsed = lastStepEvent?.elapsed ?? 0;
  const stepsPerSec = currentStep > 0 && elapsed > 0 ? currentStep / elapsed : 0;
  const eta = stepsPerSec > 0 ? (maxSteps - currentStep) / stepsPerSec : Infinity;

  const isRunning = job.status === "running";

  return (
    <div className="max-w-5xl space-y-6">
      <div className="flex items-start justify-between gap-4">
        <div>
          <h1 className="text-2xl font-semibold font-mono">{job.id}</h1>
          <div className="mt-1 flex items-center gap-3 text-sm text-slate-400">
            <JobStatusBadge status={job.status} />
            <span>stream: {state}</span>
            {job.error && <span className="text-rose-400">error: {job.error}</span>}
          </div>
        </div>
        {isRunning && (
          <button
            onClick={() => {
              if (confirm("이 학습을 중단하시겠습니까?")) stop.mutate();
            }}
            disabled={stop.isPending}
            className="px-4 py-2 rounded bg-rose-600 hover:bg-rose-500 disabled:opacity-50 text-sm font-medium"
          >
            {stop.isPending ? "Stopping…" : "Stop"}
          </button>
        )}
      </div>

      <section className="rounded-lg border border-slate-800 bg-slate-900 p-4 space-y-3">
        <div className="flex justify-between text-sm text-slate-400">
          <span>
            Step <span className="font-mono text-slate-200">{currentStep}</span> / {maxSteps}
          </span>
          <span>
            Loss{" "}
            <span className="font-mono text-slate-200">
              {currentLoss != null ? currentLoss.toFixed(4) : "—"}
            </span>
          </span>
          <span>
            Elapsed <span className="font-mono text-slate-200">{fmtDuration(elapsed)}</span>
          </span>
          <span>
            ETA{" "}
            <span className="font-mono text-slate-200">
              {Number.isFinite(eta) ? fmtDuration(eta) : "—"}
            </span>
          </span>
        </div>
        <ProgressBar value={currentStep} max={maxSteps} />
      </section>

      <LossChart events={events} />

      <section>
        <h2 className="text-sm text-slate-400 mb-2">Event log ({events.length})</h2>
        <div className="rounded-lg border border-slate-800 bg-slate-900 p-3 max-h-80 overflow-auto">
          {events.length === 0 ? (
            <div className="text-slate-500 text-xs">
              {job.status === "running" || job.status === "pending"
                ? "연결 중…"
                : "No events recorded."}
            </div>
          ) : (
            events.slice().reverse().map((ev, i) => <EventRow key={i} ev={ev} />)
          )}
        </div>
      </section>
    </div>
  );
}
