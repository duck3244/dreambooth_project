import { useMemo } from "react";
import {
  CartesianGrid,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";

import type { TrainEvent } from "../types";

export default function LossChart({ events }: { events: TrainEvent[] }) {
  const data = useMemo(
    () =>
      events
        .filter((e): e is TrainEvent & { loss: number } => e.type === "step" && typeof (e as any).loss === "number")
        .map((e) => ({
          step: e.step,
          loss: (e as any).loss as number,
        })),
    [events],
  );

  if (data.length === 0) {
    return (
      <div className="h-64 flex items-center justify-center text-slate-500 text-sm border border-slate-800 rounded-lg bg-slate-900">
        Waiting for first step…
      </div>
    );
  }

  return (
    <div className="h-64 border border-slate-800 rounded-lg bg-slate-900 p-2">
      <ResponsiveContainer width="100%" height="100%">
        <LineChart data={data}>
          <CartesianGrid stroke="#1e293b" strokeDasharray="3 3" />
          <XAxis dataKey="step" stroke="#64748b" fontSize={12} />
          <YAxis stroke="#64748b" fontSize={12} domain={["auto", "auto"]} />
          <Tooltip
            contentStyle={{ background: "#0f172a", border: "1px solid #334155" }}
            labelStyle={{ color: "#94a3b8" }}
          />
          <Line
            type="monotone"
            dataKey="loss"
            stroke="#38bdf8"
            dot={false}
            strokeWidth={2}
            isAnimationActive={false}
          />
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
}
