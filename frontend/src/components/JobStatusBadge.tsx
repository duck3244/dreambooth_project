import clsx from "clsx";

import type { JobStatus } from "../types";

const COLORS: Record<JobStatus, string> = {
  pending: "bg-slate-700 text-slate-300",
  running: "bg-blue-900 text-blue-200",
  completed: "bg-emerald-900 text-emerald-200",
  failed: "bg-rose-900 text-rose-200",
  cancelled: "bg-amber-900 text-amber-200",
};

export default function JobStatusBadge({ status }: { status: JobStatus }) {
  return (
    <span
      className={clsx(
        "inline-flex items-center px-2 py-0.5 rounded text-xs font-medium uppercase tracking-wide",
        COLORS[status],
      )}
    >
      {status}
    </span>
  );
}
