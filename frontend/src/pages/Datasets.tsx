import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { useRef, useState } from "react";

import { api } from "../api";
import type { DatasetInfo } from "../types";

function formatBytes(n: number): string {
  if (n < 1024) return `${n} B`;
  if (n < 1024 * 1024) return `${(n / 1024).toFixed(1)} KB`;
  return `${(n / 1024 / 1024).toFixed(1)} MB`;
}

export default function Datasets() {
  const qc = useQueryClient();
  const { data } = useQuery({ queryKey: ["datasets"], queryFn: api.listDatasets });
  const datasets: DatasetInfo[] = data?.datasets ?? [];

  const [newName, setNewName] = useState("");
  const [selectedId, setSelectedId] = useState<string | null>(null);

  const create = useMutation({
    mutationFn: (name: string) => api.createDataset(name),
    onSuccess: (res) => {
      setNewName("");
      setSelectedId(res.id);
      qc.invalidateQueries({ queryKey: ["datasets"] });
    },
  });

  const upload = useMutation({
    mutationFn: ({ id, files }: { id: string; files: File[] }) => api.uploadImages(id, files),
    onSuccess: () => qc.invalidateQueries({ queryKey: ["datasets"] }),
  });

  const remove = useMutation({
    mutationFn: (id: string) => api.deleteDataset(id),
    onSuccess: () => {
      setSelectedId(null);
      qc.invalidateQueries({ queryKey: ["datasets"] });
    },
  });

  const fileInputRef = useRef<HTMLInputElement>(null);

  const handleFilesSelected = (files: FileList | null) => {
    if (!selectedId || !files || files.length === 0) return;
    upload.mutate({ id: selectedId, files: Array.from(files) });
  };

  return (
    <div className="max-w-5xl space-y-6">
      <h1 className="text-2xl font-semibold">Datasets</h1>

      <section className="rounded-lg border border-slate-800 bg-slate-900 p-4">
        <h2 className="font-medium mb-2">Create new dataset</h2>
        <form
          className="flex gap-2"
          onSubmit={(e) => {
            e.preventDefault();
            if (newName.trim()) create.mutate(newName.trim());
          }}
        >
          <input
            className="flex-1 rounded bg-slate-800 border border-slate-700 px-3 py-1.5 text-sm focus:outline-none focus:border-blue-500"
            placeholder="my_person_dataset"
            value={newName}
            onChange={(e) => setNewName(e.target.value)}
          />
          <button
            type="submit"
            disabled={create.isPending || !newName.trim()}
            className="px-4 py-1.5 rounded bg-blue-600 hover:bg-blue-500 disabled:opacity-50 text-sm font-medium"
          >
            {create.isPending ? "Creating…" : "Create"}
          </button>
        </form>
        {create.isError && (
          <p className="text-xs text-rose-400 mt-2">{(create.error as Error).message}</p>
        )}
      </section>

      <section>
        <h2 className="font-medium mb-2">Existing datasets</h2>
        {datasets.length === 0 ? (
          <div className="rounded-lg border border-slate-800 bg-slate-900 p-6 text-slate-400 text-sm">
            아직 dataset이 없습니다.
          </div>
        ) : (
          <div className="rounded-lg border border-slate-800 bg-slate-900 overflow-hidden">
            <table className="w-full text-sm">
              <thead className="bg-slate-800/50 text-slate-400">
                <tr>
                  <th className="text-left px-4 py-2 font-normal">Name</th>
                  <th className="text-left px-4 py-2 font-normal">ID</th>
                  <th className="text-left px-4 py-2 font-normal">Images</th>
                  <th className="text-left px-4 py-2 font-normal">Size</th>
                  <th className="text-right px-4 py-2 font-normal">Actions</th>
                </tr>
              </thead>
              <tbody>
                {datasets.map((d) => (
                  <tr
                    key={d.id}
                    className={
                      "border-t border-slate-800 cursor-pointer " +
                      (selectedId === d.id ? "bg-slate-800/40" : "hover:bg-slate-800/20")
                    }
                    onClick={() => setSelectedId(d.id)}
                  >
                    <td className="px-4 py-2">{d.name}</td>
                    <td className="px-4 py-2 font-mono text-xs">{d.id}</td>
                    <td className="px-4 py-2 font-mono">{d.image_count}</td>
                    <td className="px-4 py-2 text-slate-400">{formatBytes(d.total_bytes)}</td>
                    <td className="px-4 py-2 text-right">
                      <button
                        onClick={(e) => {
                          e.stopPropagation();
                          if (confirm(`Delete dataset "${d.name}"? This cannot be undone.`)) {
                            remove.mutate(d.id);
                          }
                        }}
                        className="text-rose-400 hover:underline text-xs"
                      >
                        Delete
                      </button>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </section>

      {selectedId && (
        <section className="rounded-lg border border-slate-800 bg-slate-900 p-4">
          <h2 className="font-medium mb-2">
            Upload images to{" "}
            <span className="font-mono text-sm text-slate-400">{selectedId}</span>
          </h2>
          <input
            ref={fileInputRef}
            type="file"
            accept="image/*"
            multiple
            className="hidden"
            onChange={(e) => handleFilesSelected(e.target.files)}
          />
          <div
            onClick={() => fileInputRef.current?.click()}
            onDragOver={(e) => e.preventDefault()}
            onDrop={(e) => {
              e.preventDefault();
              handleFilesSelected(e.dataTransfer.files);
            }}
            className="border-2 border-dashed border-slate-700 rounded p-6 text-center cursor-pointer hover:border-blue-500 text-slate-400 text-sm"
          >
            Drop images here or click to select
          </div>
          {upload.isPending && <p className="text-xs text-slate-400 mt-2">Uploading…</p>}
          {upload.data && (
            <p className="text-xs text-emerald-400 mt-2">
              Added {upload.data.added.length} · Skipped {upload.data.skipped.length} · Total {upload.data.image_count}
            </p>
          )}
          {upload.isError && (
            <p className="text-xs text-rose-400 mt-2">{(upload.error as Error).message}</p>
          )}
        </section>
      )}
    </div>
  );
}
