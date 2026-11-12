"use client";

import { CaretDown } from "@phosphor-icons/react";
import { useEffect, useState } from "react";
import { API, type SourcesResponse } from "@/lib/api";

export interface Scope {
  id: string; // "all" | "meva" | an upload's camera id
  label: string;
  cameras: string[]; // empty = everything
}

const ALL: Scope = { id: "all", label: "All footage", cameras: [] };

/** Which footage to search: everything, the MEVA library, or one uploaded video. */
export function useScopes(): Scope[] {
  const [scopes, setScopes] = useState<Scope[]>([ALL]);
  useEffect(() => {
    Promise.all([
      fetch(`${API}/sources`).then((r) => r.json() as Promise<SourcesResponse>),
      fetch(`${API}/cameras`).then((r) => r.json() as Promise<{ id: string; kind: string }[]>),
    ])
      .then(([src, cams]) => {
        const meva = cams.filter((c) => c.kind === "library").map((c) => c.id);
        setScopes([
          ALL,
          ...(meva.length ? [{ id: "meva", label: src.library.name, cameras: meva }] : []),
          ...src.uploads.filter((u) => u.status === "ready").map((u) => ({ id: u.camera, label: u.name, cameras: [u.camera] })),
          ...cams.filter((c) => c.kind === "live").map((c) => ({ id: c.id, label: `Live: ${c.id}`, cameras: [c.id] })),
        ]);
      })
      .catch(() => {});
  }, []);
  return scopes;
}

export function ScopePicker({ scopes, value, onChange }: { scopes: Scope[]; value: string; onChange: (id: string) => void }) {
  return (
    <label className="relative flex items-center gap-2 text-xs text-muted">
      <span className="text-faint">Search in</span>
      <select
        value={value}
        onChange={(e) => onChange(e.target.value)}
        className="appearance-none rounded-[var(--radius-ui)] border border-line bg-surface py-1 pl-2.5 pr-7 text-xs text-text focus:border-accent focus:outline-none"
      >
        {scopes.map((s) => (
          <option key={s.id} value={s.id}>
            {s.label}
          </option>
        ))}
        {!scopes.some((s) => s.id === value) && <option value={value}>Camera {value}</option>}
      </select>
      <CaretDown size={12} className="pointer-events-none absolute right-2 text-faint" />
    </label>
  );
}
