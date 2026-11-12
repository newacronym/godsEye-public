"use client";

import { CaretDown, CheckCircle } from "@phosphor-icons/react";
import { motion, useReducedMotion } from "motion/react";
import { useState } from "react";
import { API, clock, partLabel, type SearchEvent, type SearchState, type Verdict } from "@/lib/api";

type Status = "provisional" | "checking" | "verified" | "possible" | "unconfirmed" | "rejected" | "unchecked";

function statusOf(e: SearchEvent, rank: number, s: SearchState): Status {
  if (s.phase === "provisional" || s.phase === "searching") return "provisional";
  const v = s.verdicts[e.id];
  if (v) {
    if (v.match) return "verified";
    if (v.unsure) return "possible";
    // Nothing ruled it out, but the specific thing asked couldn't be seen (a logo on a faraway shirt).
    return v.checks.length && !v.checks.some((c) => c.answer === "no") ? "unconfirmed" : "rejected";
  }
  return rank < s.verifyTop && s.phase !== "done" ? "checking" : "unchecked";
}

export function Results({
  state,
  selected,
  onSelect,
}: {
  state: SearchState;
  selected: string | null;
  onSelect: (e: SearchEvent) => void;
}) {
  const [showRejected, setShowRejected] = useState(false);
  const [showUnconfirmed, setShowUnconfirmed] = useState(false);
  const rows = state.events.map((e, rank) => ({ e, status: statusOf(e, rank, state), v: state.verdicts[e.id] }));
  const verified = rows
    .filter((r) => r.status === "verified")
    .sort((a, b) => (b.v?.confidence ?? 0) - (a.v?.confidence ?? 0));
  // Possible matches: the most conditions confirmed first.
  const support = (v?: Verdict) =>
    v?.checks.length ? v.checks.reduce((n, c) => n + (c.answer === "yes" ? 1 : c.answer === "maybe" ? 0.5 : 0), 0) / v.checks.length : 0;
  const possible = rows
    .filter((r) => r.status === "possible")
    .sort((a, b) => support(b.v) - support(a.v) || (b.v?.confidence ?? 0) - (a.v?.confidence ?? 0));
  const open = rows.filter((r) => r.status === "provisional" || r.status === "checking");
  const unchecked = rows.filter((r) => r.status === "unchecked");
  const rejected = rows.filter((r) => r.status === "rejected");
  const unconfirmed = rows.filter((r) => r.status === "unconfirmed");
  const done = state.phase === "done";

  return (
    <div className="flex flex-col gap-8">
      {(verified.length > 0 || done) && (
        <Section title="Verified" count={verified.length}>
          {verified.length ? (
            <Grid rows={verified} selected={selected} onSelect={onSelect} />
          ) : (
            <p className="max-w-[70ch] text-sm text-muted">
              {unconfirmed.length
                ? `Nothing confirmed. ${unconfirmed.length} candidates match part of the question, but the rest could not be seen in the footage (too far away or facing away); they are listed below.`
                : "No confident matches. The closest candidates are below; try describing what is visible in a single frame."}
            </p>
          )}
        </Section>
      )}
      {possible.length > 0 && (
        <Section title="Possible" count={possible.length} note="Nothing contradicts the question, but part of it could not be seen.">
          <Grid rows={possible} selected={selected} onSelect={onSelect} />
        </Section>
      )}
      {open.length > 0 && (
        <Section title={state.phase === "provisional" ? "First results" : "Checking"} count={open.length}>
          <Grid rows={open} selected={selected} onSelect={onSelect} />
        </Section>
      )}
      {unchecked.length > 0 && done && (
        <Section title="Not checked" count={unchecked.length}>
          <Grid rows={unchecked} selected={selected} onSelect={onSelect} />
        </Section>
      )}
      {unconfirmed.length > 0 && (
        <section>
          <button onClick={() => setShowUnconfirmed((x) => !x)} className="flex items-center gap-2 text-sm text-muted hover:text-text">
            <CaretDown size={14} className={showUnconfirmed ? "" : "-rotate-90"} />
            Couldn&apos;t be confirmed
            <span className="font-mono text-faint">{unconfirmed.length}</span>
            <span className="text-xs text-faint">part of the question wasn&apos;t visible</span>
          </button>
          {showUnconfirmed && (
            <div className="mt-3">
              <Grid rows={unconfirmed} selected={selected} onSelect={onSelect} />
            </div>
          )}
        </section>
      )}
      {rejected.length > 0 && (
        <section>
          <button
            onClick={() => setShowRejected((x) => !x)}
            className="flex items-center gap-2 text-sm text-muted hover:text-text"
          >
            <CaretDown size={14} className={showRejected ? "" : "-rotate-90"} />
            Rejected by the verifier
            <span className="font-mono text-faint">{rejected.length}</span>
          </button>
          {showRejected && (
            <div className="mt-3">
              <Grid rows={rejected} selected={selected} onSelect={onSelect} />
            </div>
          )}
        </section>
      )}
    </div>
  );
}

function Section({ title, count, note, children }: { title: string; count: number; note?: string; children: React.ReactNode }) {
  return (
    <section className="flex flex-col gap-3">
      <h2 className="flex items-baseline gap-2 text-sm font-medium text-text">
        {title}
        <span className="font-mono text-xs text-faint">{count}</span>
        {note && <span className="text-xs font-normal text-faint">{note}</span>}
      </h2>
      {children}
    </section>
  );
}

function Grid({
  rows,
  selected,
  onSelect,
}: {
  rows: { e: SearchEvent; status: Status; v?: Verdict }[];
  selected: string | null;
  onSelect: (e: SearchEvent) => void;
}) {
  return (
    <div className="grid grid-cols-[repeat(auto-fill,minmax(176px,1fr))] gap-3">
      {rows.map(({ e, status, v }, i) => (
        <Tile key={e.id} e={e} status={status} v={v} index={i} active={selected === e.id} onSelect={onSelect} />
      ))}
    </div>
  );
}

function Tile({
  e,
  status,
  v,
  index,
  active,
  onSelect,
}: {
  e: SearchEvent;
  status: Status;
  v?: Verdict;
  index: number;
  active: boolean;
  onSelect: (e: SearchEvent) => void;
}) {
  const reduce = useReducedMotion();
  const border = active
    ? "border-accent"
    : status === "verified"
      ? "border-accent/50"
      : status === "possible"
        ? "border-dashed border-accent/40"
      : status === "provisional"
        ? "border-dashed border-line-strong"
        : "border-line";
  return (
    <motion.button
      layout={!reduce}
      initial={reduce ? false : { opacity: 0, y: 8 }}
      animate={{ opacity: status === "provisional" || status === "unchecked" ? 0.7 : 1, y: 0 }}
      transition={{ duration: 0.35, delay: Math.min(index, 12) * 0.025, ease: [0.16, 1, 0.3, 1] }}
      onClick={() => onSelect(e)}
      className={`group flex flex-col overflow-hidden rounded-[var(--radius-ui)] border bg-surface text-left transition-colors hover:border-line-strong ${border}`}
    >
      {/* overflow-hidden keeps tall crops from growing the box past its aspect ratio */}
      <div className="relative aspect-[4/3] w-full overflow-hidden bg-surface-2">
        {/* eslint-disable-next-line @next/next/no-img-element */}
        <img
          src={`${API}${e.image}`}
          alt={e.detail ? `${partLabel(e.detail.part, e.detail.side)} close-up on ${e.camera}` : `${e.class ?? "scene"} on ${e.camera}`}
          className="absolute inset-0 h-full w-full object-contain"
          loading="lazy"
        />
        {e.kind === "window" && (
          <span
            className="pointer-events-none absolute border-2 border-accent"
            style={{
              left: `${e.bbox[0] * 100}%`,
              top: `${e.bbox[1] * 100}%`,
              width: `${(e.bbox[2] - e.bbox[0]) * 100}%`,
              height: `${(e.bbox[3] - e.bbox[1]) * 100}%`,
            }}
          />
        )}
        {e.detail && (
          <span className="absolute left-1.5 top-1.5 rounded-[var(--radius-ui)] bg-black/70 px-1.5 py-0.5 font-mono text-[10px] text-text">
            {partLabel(e.detail.part, e.detail.side)}
          </span>
        )}
        {e.sightings && (
          <span className="absolute bottom-1.5 right-1.5 rounded-[var(--radius-ui)] bg-black/70 px-1.5 py-0.5 font-mono text-[10px] text-text">
            {e.sightings} sightings
          </span>
        )}
        {status === "checking" && <span className="absolute inset-x-0 bottom-0 h-0.5 animate-pulse bg-accent/60" />}
      </div>
      <div className="flex items-center justify-between gap-2 px-2.5 py-2">
        <span className="truncate font-mono text-xs text-muted">
          {e.camera} <span className="text-text">{clock(e.moment)}</span>
        </span>
        {status === "verified" && v ? (
          <motion.span
            initial={reduce ? false : { scale: 0.6, opacity: 0 }}
            animate={{ scale: 1, opacity: 1 }}
            className="flex items-center gap-1 font-mono text-xs text-accent"
          >
            <CheckCircle size={14} weight="fill" />
            {(v.confidence ?? 0).toFixed(2)}
          </motion.span>
        ) : (
          <span className="font-mono text-[11px] text-faint">{e.score.toFixed(3)}</span>
        )}
      </div>
    </motion.button>
  );
}
