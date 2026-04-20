import { useQuery } from "@tanstack/react-query";
import { Link } from "react-router-dom";

import { api } from "../api";

function formatDate(ts: number): string {
  return new Date(ts * 1000).toLocaleString();
}

export default function Models() {
  const { data } = useQuery({ queryKey: ["models"], queryFn: api.listModels });
  const models = data?.models ?? [];

  return (
    <div className="max-w-5xl space-y-4">
      <h1 className="text-2xl font-semibold">Trained models</h1>

      {models.length === 0 ? (
        <div className="rounded-lg border border-slate-800 bg-slate-900 p-6 text-slate-400 text-sm">
          아직 학습된 모델이 없습니다.{" "}
          <Link to="/train/new" className="text-blue-400 underline">
            학습을 시작
          </Link>
          하세요.
        </div>
      ) : (
        <div className="rounded-lg border border-slate-800 bg-slate-900 overflow-hidden">
          <table className="w-full text-sm">
            <thead className="bg-slate-800/50 text-slate-400">
              <tr>
                <th className="text-left px-4 py-2 font-normal">Model ID</th>
                <th className="text-left px-4 py-2 font-normal">Kind</th>
                <th className="text-left px-4 py-2 font-normal">Job</th>
                <th className="text-left px-4 py-2 font-normal">Created</th>
                <th className="text-left px-4 py-2 font-normal">Path</th>
              </tr>
            </thead>
            <tbody>
              {models.map((m) => (
                <tr key={m.id} className="border-t border-slate-800">
                  <td className="px-4 py-2 font-mono text-xs">{m.id}</td>
                  <td className="px-4 py-2">
                    <span className="inline-block px-2 py-0.5 rounded bg-slate-800 text-xs uppercase">
                      {m.kind}
                    </span>
                  </td>
                  <td className="px-4 py-2">
                    {m.job_id ? (
                      <Link to={`/train/${m.job_id}`} className="text-blue-400 hover:underline font-mono">
                        {m.job_id}
                      </Link>
                    ) : (
                      "—"
                    )}
                  </td>
                  <td className="px-4 py-2 text-slate-400">{formatDate(m.created_at)}</td>
                  <td className="px-4 py-2 font-mono text-xs text-slate-500 truncate max-w-md">
                    {m.path}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
}
