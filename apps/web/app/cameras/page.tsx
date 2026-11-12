"use client";

import { Broadcast, MagnifyingGlass, SpinnerGap, Stop } from "@phosphor-icons/react";
import Link from "next/link";
import { useCallback, useEffect, useState } from "react";
import { API, WEBRTC } from "@/lib/api";

interface Indexing {
  state: string; // "starting the GPU" | "indexing" | "stopped" | "stream ended" | "failed"
  error: string | null;
  seconds: number;
  frames_indexed: number;
  frames_dropped: number;
  lag_s: number | null;
  rows: { tracks_updated?: number; crops?: number; close_ups?: number; scene_tiles?: number };
}

interface Stream {
  name: string;
  ready: boolean;
  readers: number;
  indexing: Indexing | null;
}

const RUNNING = ["starting the GPU", "indexing"];

export default function Cameras() {
  const [streams, setStreams] = useState<Stream[] | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [hd, setHd] = useState(false);

  const load = useCallback(() => {
    fetch(`${API}/live`)
      .then((r) => (r.ok ? r.json() : Promise.reject(new Error(String(r.status)))))
      .then((s: Stream[]) => {
        setStreams(s);
        setError(null);
      })
      .catch(() => setError("Can't reach the Studio. Is `make tunnel` running?"));
  }, []);

  useEffect(() => {
    load();
    const t = setInterval(load, 3000); // streams come and go; indexing status changes every second
    return () => clearInterval(t);
  }, [load]);

  const toggle = (s: Stream) => {
    const running = s.indexing && RUNNING.includes(s.indexing.state);
    fetch(`${API}/live/${encodeURIComponent(s.name)}/index`, { method: running ? "DELETE" : "POST" }).then(load);
  };

  // The 720p MEVA copies fit through the tunnel four at a time; the 1080p originals (-hd) only one at a time.
  const shown = (streams ?? []).filter((s) => s.ready && s.name.endsWith("-hd") === hd);
  const phones = shown.filter((s) => !s.name.startsWith("meva-"));
  const replays = shown.filter((s) => s.name.startsWith("meva-"));

  return (
    <div className="mx-auto flex max-w-[1600px] flex-col gap-6 px-4 py-6 md:px-8">
      <header className="flex flex-wrap items-end justify-between gap-4">
        <div>
          <h1 className="text-lg font-medium tracking-tight">Live cameras</h1>
          <p className="mt-1 max-w-[70ch] text-sm text-muted">
            Index a live stream to search it as it happens: a cloud GPU starts in about half a minute, then results are a
            second or so behind live. It costs about $0.80 an hour of Modal credit while it runs and stops by itself after
            two hours.
          </p>
        </div>
        <label className="flex items-center gap-2 text-sm text-muted">
          <input type="checkbox" checked={hd} onChange={(e) => setHd(e.target.checked)} className="accent-[var(--accent)]" />
          1080p MEVA originals
        </label>
      </header>

      {error && <p className="text-sm text-muted">{error}</p>}

      {streams && phones.length === 0 && <PhoneSetup />}

      {[...phones, ...replays].length > 0 && (
        <div className="grid grid-cols-1 gap-3 md:grid-cols-2">
          {[...phones, ...replays].map((s) => (
            <StreamCard key={s.name} stream={s} onToggle={() => toggle(s)} />
          ))}
        </div>
      )}
    </div>
  );
}

function StreamCard({ stream: s, onToggle }: { stream: Stream; onToggle: () => void }) {
  const ix = s.indexing;
  const running = ix && RUNNING.includes(ix.state);
  return (
    <figure className="overflow-hidden rounded-[var(--radius-ui)] border border-line bg-surface">
      <iframe
        title={s.name}
        src={`${WEBRTC}/${s.name}/?controls=false&muted=true&autoplay=true`}
        allow="autoplay"
        className="aspect-video w-full bg-black"
      />
      <figcaption className="flex flex-col gap-2 px-3 py-2.5">
        <div className="flex items-center justify-between gap-3">
          <span className="flex items-center gap-2 font-mono text-xs text-text">
            {!s.name.startsWith("meva-") && <Broadcast size={14} className="text-accent" />}
            {s.name}
            <span className="text-faint">{s.readers} watching</span>
          </span>
          <span className="flex items-center gap-2">
            {ix && ix.rows.tracks_updated ? (
              <Link
                href={`/?scope=${encodeURIComponent(s.name)}`}
                className="flex items-center gap-1.5 rounded-[var(--radius-ui)] border border-line-strong px-2.5 py-1 text-xs text-text hover:border-accent"
              >
                <MagnifyingGlass size={12} />
                Search
              </Link>
            ) : null}
            <button
              onClick={onToggle}
              className={`flex items-center gap-1.5 rounded-[var(--radius-ui)] px-2.5 py-1 text-xs font-medium transition-transform active:scale-[0.98] ${
                running ? "border border-line-strong text-text hover:border-alert" : "bg-accent text-accent-ink"
              }`}
            >
              {running ? <Stop size={12} weight="fill" /> : null}
              {running ? "Stop" : "Index live"}
            </button>
          </span>
        </div>
        {ix && <IndexingLine ix={ix} />}
      </figcaption>
    </figure>
  );
}

function IndexingLine({ ix }: { ix: Indexing }) {
  if (ix.state === "starting the GPU") {
    return (
      <p className="flex items-center gap-2 text-xs text-muted">
        <SpinnerGap size={12} className="animate-spin" />
        Starting a GPU ({ix.seconds} s)
      </p>
    );
  }
  if (ix.state === "failed") {
    return <p className="text-xs text-muted"><span className="text-alert">Indexing failed.</span> {ix.error}</p>;
  }
  const people = ix.rows.tracks_updated ?? 0;
  return (
    <p className="font-mono text-xs text-faint">
      {ix.state === "indexing" ? (
        <span className="text-accent">indexing</span>
      ) : (
        <span>{ix.state}</span>
      )}
      {ix.lag_s !== null && ix.state === "indexing" ? `, ${ix.lag_s.toFixed(1)} s behind live` : ""}
      {`, ${ix.frames_indexed} frames, ${people} track updates, ${ix.rows.close_ups ?? 0} close-ups`}
      {ix.frames_dropped ? `, ${ix.frames_dropped} frames skipped` : ""}
    </p>
  );
}

function PhoneSetup() {
  return (
    <section className="max-w-2xl border-y border-line py-5 text-sm text-muted">
      <h2 className="text-sm font-medium text-text">Stream from your phone</h2>
      <ol className="mt-3 flex list-decimal flex-col gap-2 pl-5">
        <li>
          Keep <code className="font-mono text-text">make tunnel</code> running on the Mac, with the phone on the same Wi-Fi.
        </li>
        <li>
          Copy the stream password: <code className="font-mono text-text">make publish-pass</code>.
        </li>
        <li>
          In Larix Broadcaster, add a connection with the URL{" "}
          <code className="break-all font-mono text-text">
            rtmp://&lt;Mac Wi-Fi IP&gt;:1935/live/phone1?user=publisher&amp;pass=&lt;password&gt;
          </code>{" "}
          and start streaming. The Mac&apos;s IP is in System Settings, Wi-Fi, Details.
        </li>
        <li>The phone appears here within a few seconds. Press Index live to make it searchable.</li>
      </ol>
    </section>
  );
}
