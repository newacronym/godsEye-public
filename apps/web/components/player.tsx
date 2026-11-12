"use client";

import { ArrowCounterClockwise, Check as CheckIcon, CheckCircle, Minus, Question, X, XCircle } from "@phosphor-icons/react";
import { useEffect, useRef, useState } from "react";
import { API, type Box, clock, partLabel, type PartView, type PhotoRef, photoUrl, type SearchEvent, type Verdict } from "@/lib/api";

interface Playback {
  url: string;
  offset: number;
  clip_start: string;
  duration: number;
}

const LEAD_S = 3; // start playback a little before the moment
const BOX_MATCH_S = 0.35;
const WINDOW_BOX_S = 1.5;

export function Player({
  event,
  verdict,
  reference,
  onClose,
}: {
  event: SearchEvent;
  verdict?: Verdict;
  reference?: PhotoRef | null; // photo search: shown next to the close-up for comparison
  onClose: () => void;
}) {
  const video = useRef<HTMLVideoElement>(null);
  const [playback, setPlayback] = useState<Playback | null>(null);
  const [track, setTrack] = useState<{ t: number; box: Box }[]>([]);
  const [box, setBox] = useState<Box | null>(null);
  const [views, setViews] = useState<PartView[]>([]);
  const [focus, setFocus] = useState<PartView | null>(null); // the close-up shown large, and outlined in the video
  const [partBox, setPartBox] = useState<Box | null>(null);
  const [aspect, setAspect] = useState(16 / 9);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    setPlayback(null);
    setTrack([]);
    setBox(null);
    setViews([]);
    setPartBox(null);
    setError(null);
    const d = event.detail;
    setFocus(
      d ? { id: Number(event.image.match(/(\d+)\.jpg$/)?.[1]), ts: event.moment, side: d.side, box: d.box, px: 0, image: event.image } : null,
    );
    const ctrl = new AbortController();
    fetch(`${API}/playback?camera=${event.camera}&ts=${encodeURIComponent(event.moment)}`, { signal: ctrl.signal })
      .then((r) => (r.ok ? r.json() : Promise.reject(new Error(`${r.status}`))))
      .then(setPlayback)
      .catch((e) => e.name !== "AbortError" && setError("No viewing copy of this clip on the Studio."));
    if (event.kind === "track") {
      fetch(`${API}/events/${event.id}/boxes`, { signal: ctrl.signal })
        .then((r) => (r.ok ? r.json() : []))
        .then((pts: { ts: string; box: Box }[]) => setTrack(pts.map((p) => ({ t: Date.parse(p.ts), box: p.box }))))
        .catch(() => {});
    }
    if (d) {
      fetch(`${API}/events/${event.id}/details?part=${d.part}`, { signal: ctrl.signal })
        .then((r) => (r.ok ? r.json() : []))
        .then(setViews)
        .catch(() => {});
    }
    return () => ctrl.abort();
  }, [event]);

  const onTime = () => {
    const v = video.current;
    if (!v || !playback) return;
    const now = Date.parse(playback.clip_start) + v.currentTime * 1000;
    if (track.length) {
      let best = track[0];
      for (const p of track) if (Math.abs(p.t - now) < Math.abs(best.t - now)) best = p;
      setBox(Math.abs(best.t - now) <= BOX_MATCH_S * 1000 ? best.box : null);
    } else {
      setBox(Math.abs(now - Date.parse(event.moment)) <= WINDOW_BOX_S * 1000 ? event.bbox : null);
    }
    setPartBox(focus && Math.abs(now - Date.parse(focus.ts)) <= BOX_MATCH_S * 1000 ? focus.box : null);
  };

  const seek = (iso: string, lead: number, play: boolean) => {
    const v = video.current;
    if (!v || !playback) return;
    v.currentTime = Math.max(0, (Date.parse(iso) - Date.parse(playback.clip_start)) / 1000 - lead);
    if (play) v.play().catch(() => {});
    else v.pause();
  };

  const showView = (view: PartView) => {
    setFocus(view);
    seek(view.ts, 0, false); // pause on the exact frame, with the part outlined
  };

  return (
    <aside className="flex flex-col gap-4">
      <header className="flex items-start justify-between gap-3">
        <div>
          <p className="font-mono text-sm text-text">
            {event.camera} <span className="text-accent">{clock(focus?.ts ?? event.moment)}</span>
          </p>
          <p className="mt-1 text-xs text-muted">
            {event.kind === "track" ? `Tracked ${event.class ?? "object"}` : "Scene region"}, {clock(event.start)} to{" "}
            {clock(event.end)}
          </p>
        </div>
        <button onClick={onClose} aria-label="Close player" className="rounded-[var(--radius-ui)] p-1.5 text-muted hover:bg-surface-2 hover:text-text">
          <X size={18} />
        </button>
      </header>

      <div className="relative w-full overflow-hidden rounded-[var(--radius-ui)] border border-line bg-black" style={{ aspectRatio: aspect }}>
        {playback && (
          <video
            ref={video}
            key={playback.url}
            src={`${API}${playback.url}`}
            className="absolute inset-0 h-full w-full"
            muted
            playsInline
            controls
            onLoadedMetadata={(e) => {
              const v = e.currentTarget;
              if (v.videoWidth && v.videoHeight) setAspect(v.videoWidth / v.videoHeight);
              v.currentTime = Math.max(0, playback.offset - LEAD_S);
              v.play().catch(() => {}); // autoplay can be blocked or paused by the browser; the user can press play
            }}
            onTimeUpdate={onTime}
            onSeeked={onTime}
          />
        )}
        {box && <Outline box={box} className="border-accent" />}
        {partBox && <Outline box={partBox} className="border-text shadow-[0_0_0_1px_rgba(0,0,0,0.6)]" />}
        {!playback && !error && <div className="absolute inset-0 animate-pulse bg-surface-2" />}
        {error && <p className="absolute inset-0 flex items-center justify-center p-6 text-center text-sm text-muted">{error}</p>}
      </div>

      <div className="flex items-center justify-between">
        <button
          onClick={() => seek(event.moment, LEAD_S, true)}
          disabled={!playback}
          className="flex items-center gap-2 text-sm text-muted hover:text-text disabled:opacity-40"
        >
          <ArrowCounterClockwise size={16} />
          Replay the moment
        </button>
        <span className="font-mono text-xs text-faint">score {event.score.toFixed(3)}</span>
      </div>

      {reference && (
        <section className="grid grid-cols-2 gap-3 border-t border-line pt-4">
          <figure className="flex flex-col gap-1.5">
            {/* eslint-disable-next-line @next/next/no-img-element */}
            <img src={photoUrl(reference)} alt="Your photo" className="aspect-[3/4] w-full rounded-[var(--radius-ui)] border border-line bg-black object-contain" />
            <figcaption className="text-xs text-faint">Your photo</figcaption>
          </figure>
          <figure className="flex flex-col gap-1.5">
            {/* eslint-disable-next-line @next/next/no-img-element */}
            <img src={`${API}${event.image}`} alt="This sighting" className="aspect-[3/4] w-full rounded-[var(--radius-ui)] border border-line bg-black object-contain" />
            <figcaption className="text-xs text-faint">
              {event.camera} at {clock(event.moment)}
            </figcaption>
          </figure>
        </section>
      )}

      {event.detail && focus && (
        <section className="flex flex-col gap-3 border-t border-line pt-4">
          <div className="flex items-baseline justify-between">
            <h3 className="text-sm text-text">Close-up, {partLabel(event.detail.part, focus.side)}</h3>
            <span className="font-mono text-xs text-faint">{clock(focus.ts)}</span>
          </div>
          {/* eslint-disable-next-line @next/next/no-img-element */}
          <img
            src={`${API}${focus.image}`}
            alt={`${partLabel(event.detail.part, focus.side)} at ${clock(focus.ts)}`}
            className="aspect-square max-h-[360px] w-full rounded-[var(--radius-ui)] border border-line bg-black object-contain"
          />
          {views.length > 1 && (
            <div className="flex gap-1.5 overflow-x-auto pb-1">
              {views.map((v) => (
                <button
                  key={v.id}
                  onClick={() => showView(v)}
                  title={`${partLabel(event.detail!.part, v.side)} at ${clock(v.ts)}, ${v.px}px`}
                  className={`relative size-14 shrink-0 overflow-hidden rounded-[var(--radius-ui)] border ${
                    v.id === focus.id ? "border-accent" : "border-line hover:border-line-strong"
                  }`}
                >
                  {/* eslint-disable-next-line @next/next/no-img-element */}
                  <img src={`${API}${v.image}`} alt="" loading="lazy" className="h-full w-full object-cover" />
                  {v.side && (
                    <span className="absolute bottom-0 right-0 bg-black/70 px-1 font-mono text-[9px] text-text">{v.side}</span>
                  )}
                </button>
              ))}
            </div>
          )}
          <p className="text-xs text-faint">
            Every view of this person&apos;s {event.detail.part === "hand" ? "hands" : event.detail.part} kept by the index, one per second. Pick
            one to jump to that frame.
          </p>
        </section>
      )}

      {verdict && (
        <div className="flex gap-3 border-t border-line pt-4">
          {verdict.match ? (
            <CheckCircle size={20} weight="fill" className="mt-0.5 shrink-0 text-accent" />
          ) : verdict.unsure ? (
            <Question size={20} className="mt-0.5 shrink-0 text-accent" />
          ) : (
            <XCircle size={20} className="mt-0.5 shrink-0 text-faint" />
          )}
          <div className="flex flex-col gap-2">
            <p className="text-sm text-text">{verdict.reason}</p>
            {verdict.checks.length > 0 && (
              <ul className="flex flex-col gap-1">
                {verdict.checks.map((c) => (
                  <li key={c.condition} className="flex items-center gap-2 text-xs">
                    {c.answer === "yes" ? (
                      <CheckIcon size={14} className="shrink-0 text-accent" />
                    ) : c.answer === "no" ? (
                      <X size={14} className="shrink-0 text-faint" />
                    ) : (
                      <Minus size={14} className="shrink-0 text-muted" />
                    )}
                    <span className={c.answer === "no" ? "text-faint line-through" : "text-muted"}>{c.condition}</span>
                    {c.answer === "not_visible" && <span className="text-faint">not visible</span>}
                    {c.answer === "maybe" && <span className="text-faint">maybe</span>}
                  </li>
                ))}
              </ul>
            )}
            <p className="font-mono text-xs text-faint">
              {verdict.match ? "verified" : verdict.unsure ? "unsure" : "rejected"} by {verdict.by ?? "Haiku 5.5"}, confidence{" "}
              {(verdict.confidence ?? 0).toFixed(2)}
            </p>
          </div>
        </div>
      )}
    </aside>
  );
}

function Outline({ box, className }: { box: Box; className: string }) {
  return (
    <span
      className={`pointer-events-none absolute border-2 transition-all duration-150 ${className}`}
      style={{
        left: `${box[0] * 100}%`,
        top: `${box[1] * 100}%`,
        width: `${(box[2] - box[0]) * 100}%`,
        height: `${(box[3] - box[1]) * 100}%`,
      }}
    />
  );
}
