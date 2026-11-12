"use client";

import { ArrowRight } from "@phosphor-icons/react";
import { clock, type SearchEvent, type SearchState } from "@/lib/api";

/** Photo search: where the person was seen, in time order. Verified sightings only. */
export function Trail({
  state,
  selected,
  onSelect,
}: {
  state: SearchState;
  selected: string | null;
  onSelect: (e: SearchEvent) => void;
}) {
  const seen = state.events
    .filter((e) => state.verdicts[e.id]?.match)
    .sort((a, b) => a.moment.localeCompare(b.moment));
  if (!seen.length) {
    return state.phase === "done" ? (
      <p className="mb-6 border-y border-line py-3 text-sm text-muted">
        Not found with confidence in this footage. Possible matches, if any, are below.
      </p>
    ) : null;
  }
  const cameras = new Set(seen.map((e) => e.camera));
  return (
    <section className="mb-6 flex flex-col gap-2 border-y border-line py-3">
      <h2 className="text-sm text-text">
        Seen {seen.length} {seen.length === 1 ? "time" : "times"} on {cameras.size} {cameras.size === 1 ? "camera" : "cameras"}
      </h2>
      <ol className="flex flex-wrap items-center gap-1.5">
        {seen.map((e, i) => (
          <li key={e.id} className="flex items-center gap-1.5">
            {i > 0 && <ArrowRight size={12} className="text-faint" />}
            <button
              onClick={() => onSelect(e)}
              className={`rounded-[var(--radius-ui)] border px-2 py-1 font-mono text-xs ${
                selected === e.id ? "border-accent text-text" : "border-line text-muted hover:border-line-strong hover:text-text"
              }`}
            >
              <span className="text-accent">{clock(e.moment).slice(0, 5)}</span> {e.camera}
            </button>
          </li>
        ))}
      </ol>
    </section>
  );
}
