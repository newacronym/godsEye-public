"use client";

import { ArrowRight, ImageSquare, MagnifyingGlass, SpinnerGap, X } from "@phosphor-icons/react";
import { useEffect, useRef, useState } from "react";
import { API, type PhotoRef, photoUrl, type UploadedPhoto } from "@/lib/api";

/** The question bar: words, a photo of a person, or both. A photo can be chosen, pasted or dropped. */
export function QueryBar({
  initial,
  initialPhoto,
  busy,
  onSubmit,
}: {
  initial: string;
  initialPhoto: PhotoRef | null;
  busy: boolean;
  onSubmit: (q: string, photo: PhotoRef | null) => void;
}) {
  const [value, setValue] = useState(initial);
  const [photo, setPhoto] = useState<PhotoRef | null>(initialPhoto);
  const [people, setPeople] = useState<UploadedPhoto | null>(null);
  const [sending, setSending] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const input = useRef<HTMLInputElement>(null);
  const picker = useRef<HTMLInputElement>(null);

  useEffect(() => setValue(initial), [initial]);
  useEffect(() => setPhoto(initialPhoto), [initialPhoto]);
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if ((e.metaKey || e.ctrlKey) && e.key.toLowerCase() === "k") {
        e.preventDefault();
        input.current?.focus();
        input.current?.select();
      }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, []);

  const attach = (file: File | undefined | null) => {
    if (!file) return;
    if (!file.type.startsWith("image/")) {
      setError("That isn't an image.");
      return;
    }
    setSending(true);
    setError(null);
    fetch(`${API}/photos`, { method: "POST", body: file })
      .then((r) => (r.ok ? r.json() : Promise.reject(new Error(String(r.status)))))
      .then((p: UploadedPhoto) => {
        setPeople(p);
        setPhoto({ id: p.id, person: 0 });
        if (!p.people.length) setError("No one was found in this photo; the whole photo will be used.");
      })
      .catch(() => setError("Couldn't upload the photo. Is `make tunnel` running?"))
      .finally(() => setSending(false));
  };

  const submit = () => {
    if (value.trim() || photo) onSubmit(value.trim(), photo);
  };

  return (
    <div className="flex flex-col gap-2">
      <form
        className="flex items-center gap-3 rounded-[var(--radius-ui)] border border-line-strong bg-surface px-4 focus-within:border-accent"
        onSubmit={(e) => {
          e.preventDefault();
          submit();
        }}
        onDragOver={(e) => e.preventDefault()}
        onDrop={(e) => {
          e.preventDefault();
          attach(e.dataTransfer.files[0]);
        }}
      >
        <MagnifyingGlass size={20} className="shrink-0 text-muted" />
        {photo && (
          <span className="flex shrink-0 items-center gap-1.5 rounded-[var(--radius-ui)] border border-line bg-surface-2 py-1 pl-1 pr-1.5">
            {/* eslint-disable-next-line @next/next/no-img-element */}
            <img src={photoUrl(photo)} alt="The person to find" className="h-9 w-7 rounded-[3px] object-cover" />
            <span className="text-xs text-muted">this person</span>
            <button type="button" onClick={() => (setPhoto(null), setPeople(null))} aria-label="Remove the photo" className="text-faint hover:text-text">
              <X size={14} />
            </button>
          </span>
        )}
        <label htmlFor="q" className="sr-only">
          Describe what you're looking for
        </label>
        <input
          id="q"
          ref={input}
          autoFocus
          value={value}
          onChange={(e) => setValue(e.target.value)}
          onPaste={(e) => {
            const file = Array.from(e.clipboardData.files).find((f) => f.type.startsWith("image/"));
            if (file) {
              e.preventDefault();
              attach(file);
            }
          }}
          placeholder={photo ? "Anything to add? (optional)" : "Describe what you're looking for, or add a photo of a person"}
          className="h-14 min-w-0 flex-1 bg-transparent text-base text-text placeholder:text-faint focus:outline-none"
        />
        <button
          type="button"
          onClick={() => picker.current?.click()}
          aria-label="Search by a photo of a person"
          title="Search by a photo of a person"
          className="shrink-0 rounded-[var(--radius-ui)] p-1.5 text-muted hover:bg-surface-2 hover:text-text"
        >
          {sending ? <SpinnerGap size={20} className="animate-spin" /> : <ImageSquare size={20} />}
        </button>
        <input ref={picker} type="file" accept="image/*" className="hidden" onChange={(e) => attach(e.target.files?.[0])} />
        <kbd className="hidden font-mono text-xs text-faint sm:block">⌘K</kbd>
        <button
          type="submit"
          disabled={(!value.trim() && !photo) || busy || sending}
          className="flex h-9 items-center gap-2 rounded-[var(--radius-ui)] bg-accent px-4 text-sm font-medium text-accent-ink transition-transform active:scale-[0.98] disabled:opacity-40"
        >
          Search
          <ArrowRight size={16} weight="bold" />
        </button>
      </form>
      {people && people.people.length > 1 && photo && (
        <div className="flex items-center gap-2 text-xs text-muted">
          <span>{people.people.length} people in this photo. Find:</span>
          {people.people.map((_, i) => (
            <button
              key={i}
              type="button"
              onClick={() => setPhoto({ id: people.id, person: i })}
              className={`overflow-hidden rounded-[3px] border ${photo.person === i ? "border-accent" : "border-line hover:border-line-strong"}`}
              aria-label={`Person ${i + 1}`}
            >
              {/* eslint-disable-next-line @next/next/no-img-element */}
              <img src={photoUrl({ id: people.id, person: i })} alt="" className="h-12 w-9 object-cover" />
            </button>
          ))}
        </div>
      )}
      {error && <p className="text-xs text-muted">{error}</p>}
    </div>
  );
}
