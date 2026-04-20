import { useQuery } from "@tanstack/react-query";
import { Link } from "react-router-dom";

import { api } from "../api";
import GpuStatusCard from "../components/GpuStatusCard";
import JobStatusBadge from "../components/JobStatusBadge";

function formatDate(ts: number): string {
  return new Date(ts * 1000).toLocaleString();
}

export default function Dashboard() {
  const { data: jobs } = useQuery({
    queryKey: ["jobs"],
    queryFn: api.listJobs,
    refetchInterval: 3000,
  });

  const recent = (jobs?.jobs ?? []).slice(0, 8);

  return (
    <div className="space-y-6 max-w-5xl">
      <h1 className="text-2xl font-semibold">Dashboard</h1>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        <GpuStatusCard />
        <div className="rounded-lg border border-slate-800 bg-slate-900 p-4 flex flex-col justify-between">
          <div>
            <div className="text-sm text-slate-400">Quick actions</div>
            <div className="mt-2 flex flex-col gap-2">
              <Link to="/datasets" className="text-blue-400 hover:underline text-sm">
                → Manage datasets
              </Link>
              <Link to="/train/new" className="text-blue-400 hover:underline text-sm">
                → Start new training
              </Link>
              <Link to="/models" className="text-blue-400 hover:underline text-sm">
                → View trained models
              </Link>
            </div>
          </div>
        </div>
      </div>

      <section>
        <div className="flex items-center justify-between mb-2">
          <h2 className="text-lg font-semibold">Recent jobs</h2>
          <Link to="/train/new" className="text-sm text-blue-400 hover:underline">
            + New
          </Link>
        </div>

        {recent.length === 0 ? (
          <div className="rounded-lg border border-slate-800 bg-slate-900 p-6 text-slate-400 text-sm">
            아직 학습 job이 없습니다. <Link to="/train/new" className="text-blue-400 underline">첫 학습을 시작</Link>해보세요.
          </div>
        ) : (
          <div className="rounded-lg border border-slate-800 bg-slate-900 overflow-hidden">
            <table className="w-full text-sm">
              <thead className="bg-slate-800/50 text-slate-400">
                <tr>
                  <th className="text-left px-4 py-2 font-normal">Job</th>
                  <th className="text-left px-4 py-2 font-normal">Status</th>
                  <th className="text-left px-4 py-2 font-normal">Step</th>
                  <th className="text-left px-4 py-2 font-normal">Loss</th>
                  <th className="text-left px-4 py-2 font-normal">Started</th>
                </tr>
              </thead>
              <tbody>
                {recent.map((j) => (
                  <tr key={j.id} className="border-t border-slate-800">
                    <td className="px-4 py-2">
                      <Link to={`/train/${j.id}`} className="text-blue-400 hover:underline font-mono">
                        {j.id}
                      </Link>
                    </td>
                    <td className="px-4 py-2">
                      <JobStatusBadge status={j.status} />
                    </td>
                    <td className="px-4 py-2 font-mono">
                      {j.latest_step ?? 0}/{j.max_train_steps ?? "?"}
                    </td>
                    <td className="px-4 py-2 font-mono">
                      {j.latest_loss != null ? j.latest_loss.toFixed(4) : "—"}
                    </td>
                    <td className="px-4 py-2 text-slate-400">
                      {j.started_at ? formatDate(j.started_at) : "—"}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </section>
    </div>
  );
}
