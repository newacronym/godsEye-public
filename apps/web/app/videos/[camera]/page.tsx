"use client";

import { ArrowLeft, MagnifyingGlass } from "@phosphor-icons/react";
import Link from "next/link";
import { useParams } from "next/navigation";
import { useEffect, useRef, useState } from "react";
import { API, clock } from "@/lib/api";

interface Clip {
  start: string;
  end: string;
  url: string;
}

interface Footage {
  camera: string;
  name: string;
  kind: "upload" | "library";
  clips: Clip[];
}

/** Watch a camera or an uploaded video from start to end; clips play back to back. */
export default function Watch() {
  const { camera } = useParams<{ camera: string }>();
  const [footage, setFootage] = useState<Footage | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [index, setIndex] = useState(0);
  const [now, setNow] = useState<string | null>(null); // wall-clock time at the playhead
  const [aspect, setAspect] = useState(16 / 9);
  const video = useRef<HTMLVideoElement>(null);
  const autoplay = useRef(false); // play the next clip on its own only after one was playing

  useEffect(() => {
    fetch(`${API}/footage/${encodeURIComponent(camera)}`)
      .then((r) => (r.ok ? r.json() : Promise.reject(new Error(String(r.status)))))
      .then((f: Footage) => (f.clips.length ? setFootage(f) : setError("This footage has no viewing copy yet.")))
      .catch(() => setError("Couldn't load this footage. Is `make tunnel` running?"));
  }, [camera]);

  const clip = footage?.clips[index];
  const first = footage?.clips[0];
  const last = footage?.clips[footage.clips.length - 1];

  const onTime = () => {
    const v = video.current;
    if (v && clip) setNow(new Date(Date.parse(clip.start) + v.currentTime * 1000).toISOString());
  };

  return (
    <div className="mx-auto flex max-w-[1200px] flex-col gap-5 px-4 py-6 md:px-8">
      <Link href="/videos" className="flex w-fit items-center gap-1.5 text-sm text-muted hover:text-text">
        <ArrowLeft size={14} />
        Videos
      </Link>

      {error && <p className="text-sm text-muted">{error}</p>}

      {footage && clip && first && last && (
        <>
          <header className="flex flex-wrap items-end justify-between gap-4">
            <div className="min-w-0">
              <h1 className="truncate text-lg font-medium tracking-tight text-text">{footage.name}</h1>
              <p className="mt-1 font-mono text-xs text-faint">
                {footage.camera}, {first.start.slice(0, 10)} {clock(first.start)} to {clock(last.end)} UTC
                {footage.clips.length > 1 ? `, ${footage.clips.length} clips` : ""}
              </p>
            </div>
            <Link
              href={`/?scope=${encodeURIComponent(footage.camera)}`}
              className="flex items-center gap-2 rounded-[var(--radius-ui)] bg-accent px-4 py-2 text-sm font-medium text-accent-ink transition-transform active:scale-[0.98]"
            >
              <MagnifyingGlass size={16} weight="bold" />
              Search this footage
            </Link>
          </header>

          <div
            className="relative mx-auto w-full overflow-hidden rounded-[var(--radius-ui)] border border-line bg-black"
            style={{ aspectRatio: aspect, maxHeight: "72dvh", maxWidth: `calc(72dvh * ${aspect})` }}
          >
            <video
              ref={video}
              key={clip.url}
              src={`${API}${clip.url}`}
              className="absolute inset-0 h-full w-full"
              controls
              muted
              playsInline
              preload="metadata"
              onLoadedMetadata={(e) => {
                const v = e.currentTarget;
                if (v.videoWidth && v.videoHeight) setAspect(v.videoWidth / v.videoHeight);
                onTime();
                if (autoplay.current) v.play().catch(() => {});
              }}
              onPlay={() => (autoplay.current = true)}
              onTimeUpdate={onTime}
              onSeeked={onTime}
              onEnded={() => index + 1 < footage.clips.length && setIndex(index + 1)}
            />
            {now && (
              <span className="pointer-events-none absolute left-3 top-3 rounded-[var(--radius-ui)] bg-black/70 px-2 py-1 font-mono text-sm text-accent">
                {clock(now)}
              </span>
            )}
          </div>

          {footage.clips.length > 1 && (
            <ol className="flex flex-wrap gap-1.5">
              {footage.clips.map((c, i) => (
                <li key={c.url}>
                  <button
                    onClick={() => setIndex(i)}
                    className={`rounded-[var(--radius-ui)] border px-2.5 py-1 font-mono text-xs ${
                      i === index ? "border-accent text-text" : "border-line text-muted hover:border-line-strong hover:text-text"
                    }`}
                  >
                    {clock(c.start).slice(0, 5)} to {clock(c.end).slice(0, 5)}
                  </button>
                </li>
              ))}
            </ol>
          )}
        </>
      )}

      {!footage && !error && <div className="aspect-video w-full animate-pulse rounded-[var(--radius-ui)] bg-surface-2" />}
    </div>
  );
}
