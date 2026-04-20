import { NavLink, Route, Routes } from "react-router-dom";
import clsx from "clsx";

import Dashboard from "./pages/Dashboard";
import Datasets from "./pages/Datasets";
import Generate from "./pages/Generate";
import TrainNew from "./pages/TrainNew";
import TrainLive from "./pages/TrainLive";
import Models from "./pages/Models";

const NAV = [
  { to: "/", label: "Dashboard", end: true },
  { to: "/datasets", label: "Datasets" },
  { to: "/train/new", label: "Train" },
  { to: "/models", label: "Models" },
  { to: "/generate", label: "Generate" },
];

export default function App() {
  return (
    <div className="min-h-screen flex">
      <aside className="w-56 border-r border-slate-800 bg-slate-900/50 p-4 flex flex-col gap-1">
        <div className="text-lg font-semibold mb-4">DreamBooth</div>
        {NAV.map((n) => (
          <NavLink
            key={n.to}
            to={n.to}
            end={n.end}
            className={({ isActive }) =>
              clsx(
                "px-3 py-2 rounded text-sm",
                isActive
                  ? "bg-slate-800 text-white"
                  : "text-slate-400 hover:bg-slate-800/60 hover:text-slate-200",
              )
            }
          >
            {n.label}
          </NavLink>
        ))}
      </aside>
      <main className="flex-1 p-6 overflow-auto">
        <Routes>
          <Route path="/" element={<Dashboard />} />
          <Route path="/datasets" element={<Datasets />} />
          <Route path="/train/new" element={<TrainNew />} />
          <Route path="/train/:jobId" element={<TrainLive />} />
          <Route path="/models" element={<Models />} />
          <Route path="/generate" element={<Generate />} />
        </Routes>
      </main>
    </div>
  );
}
