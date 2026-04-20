import { useQuery } from "@tanstack/react-query";

import { api } from "../api";

function MemoryBar({ used, total }: { used: number; total: number }) {
  const pct = total > 0 ? Math.min(100, (used / total) * 100) : 0;
  const over75 = pct > 75;
  return (
    <div className="w-full bg-slate-800 rounded h-2 overflow-hidden">
      <div
        className={over75 ? "bg-amber-500 h-full" : "bg-emerald-500 h-full"}
        style={{ width: `${pct}%` }}
      />
    </div>
  );
}

export default function GpuStatusCard() {
  const { data, isLoading, error } = useQuery({
    queryKey: ["gpu"],
    queryFn: api.gpu,
    refetchInterval: 3000,
  });

  if (isLoading) {
    return <div className="rounded-lg border border-slate-800 bg-slate-900 p-4 text-slate-400">Loading GPU…</div>;
  }
  if (error || !data) {
    return (
      <div className="rounded-lg border border-rose-900 bg-rose-950/40 p-4 text-rose-300 text-sm">
        GPU 정보를 불러오지 못했습니다: {String((error as Error)?.message ?? "unknown")}
      </div>
    );
  }

  if (!data.available) {
    return (
      <div className="rounded-lg border border-amber-900 bg-amber-950/40 p-4 text-amber-300 text-sm">
        GPU not available{data.error ? `: ${data.error}` : ""}
      </div>
    );
  }

  const usedMb = data.memory_used_mb ?? 0;
  const totalMb = data.memory_total_mb ?? 0;

  return (
    <div className="rounded-lg border border-slate-800 bg-slate-900 p-4">
      <div className="flex items-start justify-between">
        <div>
          <div className="text-sm text-slate-400">GPU</div>
          <div className="font-semibold">{data.name}</div>
          <div className="text-xs text-slate-500 mt-1">
            Driver {data.driver_version ?? "?"}
          </div>
        </div>
        <div className="text-right">
          <div className="text-2xl font-mono">{data.utilization_percent ?? 0}%</div>
          <div className="text-xs text-slate-500">utilization</div>
        </div>
      </div>
      <div className="mt-4">
        <div className="flex justify-between text-xs text-slate-400 mb-1">
          <span>VRAM</span>
          <span>
            {usedMb.toLocaleString()} / {totalMb.toLocaleString()} MB
          </span>
        </div>
        <MemoryBar used={usedMb} total={totalMb} />
      </div>
      {data.temperature_c != null && (
        <div className="mt-3 text-xs text-slate-500">Temp: {data.temperature_c}°C</div>
      )}
    </div>
  );
}
