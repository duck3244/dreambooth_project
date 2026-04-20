import { useEffect, useRef, useState } from "react";

export type SSEState = "idle" | "connecting" | "open" | "closed" | "error";

export interface SSEEventLike {
  type: string;
  ts: number;
}

const DEFAULT_TERMINAL = new Set(["completed", "error", "cancelled"]);
const DEFAULT_TYPES = [
  "started",
  "step",
  "validation",
  "validation_started",
  "validation_error",
  "checkpoint",
  "completed",
  "error",
  "cancelled",
  "image",
];

/**
 * Subscribe to an SSE stream emitting JSON-encoded event records.
 *
 * EventSource auto-reconnects by default; once we see a terminal event we
 * close the connection explicitly so the browser does not reconnect forever.
 */
export function useSSE<T extends SSEEventLike = SSEEventLike>(
  url: string | null,
  options?: {
    terminalTypes?: Set<string>;
    eventTypes?: string[];
  },
) {
  const [events, setEvents] = useState<T[]>([]);
  const [state, setState] = useState<SSEState>(url ? "connecting" : "idle");
  const [terminal, setTerminal] = useState<T | null>(null);
  const esRef = useRef<EventSource | null>(null);

  const terminalTypes = options?.terminalTypes ?? DEFAULT_TERMINAL;
  const typesKey = (options?.eventTypes ?? DEFAULT_TYPES).join(",");

  useEffect(() => {
    if (!url) {
      setState("idle");
      return;
    }
    setEvents([]);
    setTerminal(null);
    setState("connecting");

    const es = new EventSource(url);
    esRef.current = es;

    es.onopen = () => setState("open");
    es.onerror = () => setState((prev) => (prev === "closed" ? prev : "error"));

    const handle = (raw: MessageEvent) => {
      try {
        const parsed = JSON.parse(raw.data) as T;
        setEvents((prev) => [...prev, parsed]);
        if (terminalTypes.has(parsed.type)) {
          setTerminal(parsed);
          es.close();
          setState("closed");
        }
      } catch {
        setEvents((prev) => [
          ...prev,
          { ts: Date.now() / 1000, type: "parse_error" } as unknown as T,
        ]);
      }
    };

    es.onmessage = handle;
    const types = options?.eventTypes ?? DEFAULT_TYPES;
    for (const t of types) es.addEventListener(t, handle as EventListener);

    return () => {
      for (const t of types) es.removeEventListener(t, handle as EventListener);
      es.close();
      esRef.current = null;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [url, typesKey]);

  return { events, state, terminal };
}
