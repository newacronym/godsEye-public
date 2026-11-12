"use client";

import { ArrowRight, FilmStrip, Play, Trash, UploadSimple, WarningCircle } from "@phosphor-icons/react";
import Link from "next/link";
import { useCallback, useEffect, useRef, useState } from "react";
import { API, clock, type Source, type SourcesResponse } from "@/lib/api";

interface Camera {
  id: string;
  start: string;
  end: string;
  tracks: number;
  kind: "library" | "upload" | "live";
}

const ACTIVE = ["uploading", "preparing", "indexing", "loading"];
const STATUS: Record<string, string> = {
  uploading: "Uploading",
  preparing: "Preparing",
  indexing: "Indexing on GPU",
  loading: "Loading",
  ready: "Ready",
  failed: "Failed",
};

function minutes(s?: number | null): string {
  if (!s) return "";
  const m = Math.floor(s / 60);
  const r = Math.round(s % 60);
  return m ? `${m}:${String(r).padStart(2, "0")} min` : `${r} s`;
}

export default function Videos() {
  const [data, setData] = useState<SourcesResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [sending, setSending] = useState<{ name: string; fraction: number } | null>(null);
  const [drag, setDrag] = useState(false);
  const [cameras, setCameras] = useState<Camera[]>([]);
  const picker = useRef<HTMLInputElement>(null);

  useEffect(() => {
    fetch(`${API}/cameras`)
      .then((r) => (r.ok ? r.json() : []))
      .then((cs: Camera[]) => setCameras(cs.filter((c) => c.kind === "library")))
      .catch(() => {});
  }, []);

  const load = useCallback(() => {
    fetch(`${API}/sources`)
      .then((r) => (r.ok ? r.json() : Promise.reject(new Error(String(r.status)))))
      .then((d: SourcesResponse) => {
        setData(d);
        setError(null);
      })
      .catch(() => setError("Can't reach the Studio. Is `make tunnel` running?"));
  }, []);

  useEffect(load, [load]);
  // Poll while anything is in flight; the pipeline reports progress on each source.
  const busy = sending !== null || (data?.uploads ?? []).some((s) => ACTIVE.includes(s.status));
  useEffect(() => {
    if (!busy) return;
    const t = setInterval(load, 2000);
    return () => clearInterval(t);
  }, [busy, load]);

  const upload = (file: File) => {
    // XHR, not fetch: only XHR reports upload progress. The body goes through the SSH tunnel (~1 MB/s).
    const xhr = new XMLHttpRequest();
    xhr.open("POST", `${API}/sources?name=${encodeURIComponent(file.name)}`);
    xhr.upload.onprogress = (e) => e.lengthComputable && setSending({ name: file.name, fraction: e.loaded / e.total });
    xhr.onload = () => {
      setSending(null);
      if (xhr.status >= 400) setError(`Upload failed (${xhr.status}).`);
      load();
    };
    xhr.onerror = () => {
      setSending(null);
      setError("Upload interrupted. Is `make tunnel` running?");
    };
    setSending({ name: file.name, fraction: 0 });
    xhr.send(file);
    setTimeout(load, 500);
  };

  const remove = (s: Source) => {
    if (!window.confirm(`Remove "${s.name}" and its index?`)) return;
    fetch(`${API}/sources/${s.camera}`, { method: "DELETE" }).then(load);
  };

  const files = (list: FileList | null) => {
    const f = list?.[0];
    if (f && f.type.startsWith("video/")) upload(f);
    else if (f) setError("That doesn't look like a video file.");
  };

  return (
    <div className="mx-auto flex max-w-[1100px] flex-col gap-8 px-4 py-6 md:px-8">
      <header className="flex flex-col gap-1">
        <h1 className="text-lg font-medium tracking-tight">Videos</h1>
        <p className="max-w-[65ch] text-sm text-muted">
          Upload a video to search it. It is prepared on the Studio, indexed on a cloud GPU, and searchable in a few
          minutes for a short clip.
        </p>
      </header>

      <button
        type="button"
        onClick={() => picker.current?.click()}
        onDragOver={(e) => {
          e.preventDefault();
          setDrag(true);
        }}
        onDragLeave={() => setDrag(false)}
        onDrop={(e) => {
          e.preventDefault();
          setDrag(false);
          files(e.dataTransfer.files);
        }}
        disabled={sending !== null}
        className={`flex flex-col items-center gap-3 rounded-[var(--radius-ui)] border border-dashed px-6 py-10 text-center transition-colors disabled:cursor-wait ${
          drag ? "border-accent bg-accent-soft" : "border-line-strong hover:border-accent hover:bg-surface"
        }`}
      >
        <UploadSimple size={28} className="text-accent" />
        {sending ? (
          <div className="flex w-full max-w-sm flex-col gap-2">
            <p className="truncate text-sm text-text">Uploading {sending.name}</p>
            <Progress fraction={sending.fraction} />
            <p className="font-mono text-xs text-faint">{Math.round(sending.fraction * 100)}%, through the tunnel</p>
          </div>
        ) : (
          <>
            <p className="text-sm text-text">Drop a video here, or choose a file</p>
            <p className="text-xs text-faint">MP4, MOV or AVI. Phone videos work; a minute or two uploads in about a minute.</p>
          </>
        )}
      </button>
      <input ref={picker} type="file" accept="video/*" className="hidden" onChange={(e) => files(e.target.files)} />

      {error && (
        <p className="flex items-center gap-2 text-sm text-muted">
          <WarningCircle size={16} className="text-alert" />
          {error}
        </p>
      )}

      <section className="flex flex-col gap-3">
        <h2 className="text-sm font-medium text-text">Your uploads</h2>
        {data && data.uploads.length === 0 && <p className="text-sm text-faint">Nothing uploaded yet.</p>}
        <ul className="flex flex-col divide-y divide-line border-y border-line">
          {data?.uploads.map((s) => (
            <li key={s.camera} className="flex items-center gap-4 py-3">
              <Thumb
                href={s.status === "ready" ? `/videos/${s.camera}` : null}
                src={s.duration_s ? `${API}/uploads/${s.camera}/thumb.jpg` : null}
                label={`Watch ${s.name}`}
              />
              <div className="flex min-w-0 flex-1 flex-col gap-1.5">
                <div className="flex items-baseline gap-2">
                  {s.status === "ready" ? (
                    <Link href={`/videos/${s.camera}`} className="truncate text-sm text-text hover:text-accent">
                      {s.name}
                    </Link>
                  ) : (
                    <p className="truncate text-sm text-text">{s.name}</p>
                  )}
                  <span className="font-mono text-xs text-faint">{s.camera}</span>
                </div>
                {ACTIVE.includes(s.status) ? (
                  <div className="flex max-w-md flex-col gap-1">
                    <Progress fraction={s.fraction ?? 0} />
                    <p className="text-xs text-muted">
                      {STATUS[s.status]}: {s.stage}
                    </p>
                  </div>
                ) : s.status === "failed" ? (
                  <p className="text-xs text-muted">
                    <span className="text-alert">Failed.</span> {s.message}
                  </p>
                ) : (
                  <p className="font-mono text-xs text-faint">
                    {minutes(s.duration_s)}, {s.width}x{s.height}
                    {s.loaded ? `, ${s.loaded.tracks} tracked objects` : ""}
                    {s.index_stats?.total_s ? `, indexed in ${Math.round(s.index_stats.total_s)} s` : ""}
                  </p>
                )}
              </div>
              {s.status === "ready" && (
                <Link
                  href={`/?scope=${s.camera}`}
                  className="flex items-center gap-1.5 rounded-[var(--radius-ui)] border border-line-strong px-3 py-1.5 text-sm text-text hover:border-accent"
                >
                  Search
                  <ArrowRight size={14} />
                </Link>
              )}
              {!ACTIVE.includes(s.status) && (
                <button onClick={() => remove(s)} aria-label={`Remove ${s.name}`} className="p-1.5 text-faint hover:text-alert">
                  <Trash size={16} />
                </button>
              )}
            </li>
          ))}
        </ul>
      </section>

      {data?.library.cameras ? (
        <section className="flex flex-col gap-3">
          <div className="flex items-end justify-between gap-4">
            <div>
              <h2 className="text-sm font-medium text-text">{data.library.name}</h2>
              <p className="mt-1 font-mono text-xs text-faint">
                {data.library.cameras} cameras, {data.library.start?.slice(0, 10)} {data.library.start?.slice(11, 16)} to{" "}
                {data.library.end?.slice(11, 16)} UTC, {data.library.tracks.toLocaleString()} tracked objects
              </p>
            </div>
            <Link
              href="/?scope=meva"
              className="flex items-center gap-1.5 rounded-[var(--radius-ui)] border border-line-strong px-3 py-1.5 text-sm text-text hover:border-accent"
            >
              Search
              <ArrowRight size={14} />
            </Link>
          </div>
          <ul className="grid grid-cols-[repeat(auto-fill,minmax(200px,1fr))] gap-3">
            {cameras.map((c) => (
              <li key={c.id}>
                <Link href={`/videos/${c.id}`} className="group flex flex-col gap-1.5">
                  <span className="relative aspect-video overflow-hidden rounded-[var(--radius-ui)] border border-line bg-surface-2 group-hover:border-accent">
                    {/* eslint-disable-next-line @next/next/no-img-element */}
                    <img src={`${API}/footage/${c.id}/thumb.jpg`} alt="" loading="lazy" className="absolute inset-0 h-full w-full object-cover" />
                    <PlayBadge />
                  </span>
                  <span className="flex items-baseline justify-between gap-2 font-mono text-xs">
                    <span className="text-text">{c.id}</span>
                    <span className="text-faint">
                      {clock(c.start).slice(0, 5)} to {clock(c.end).slice(0, 5)}
                    </span>
                  </span>
                </Link>
              </li>
            ))}
          </ul>
        </section>
      ) : null}
    </div>
  );
}

function PlayBadge() {
  return (
    <span className="absolute inset-0 flex items-center justify-center bg-black/0 opacity-0 transition-opacity group-hover:bg-black/30 group-hover:opacity-100">
      <Play size={28} weight="fill" className="text-text" />
    </span>
  );
}

function Thumb({ href, src, label }: { href: string | null; src: string | null; label: string }) {
  const inner = (
    <>
      {src ? (
        // eslint-disable-next-line @next/next/no-img-element
        <img src={src} alt="" className="absolute inset-0 h-full w-full object-cover" />
      ) : (
        <FilmStrip size={20} className="absolute inset-0 m-auto text-faint" />
      )}
      {href && <PlayBadge />}
    </>
  );
  const cls = "group relative aspect-video w-32 shrink-0 overflow-hidden rounded-[var(--radius-ui)] bg-surface-2";
  return href ? (
    <Link href={href} aria-label={label} className={`${cls} border border-line hover:border-accent`}>
      {inner}
    </Link>
  ) : (
    <div className={cls}>{inner}</div>
  );
}

function Progress({ fraction }: { fraction: number }) {
  return (
    <div className="h-1 w-full overflow-hidden rounded-full bg-surface-2">
      <div className="h-full bg-accent transition-[width] duration-500" style={{ width: `${Math.max(2, fraction * 100)}%` }} />
    </div>
  );
}
