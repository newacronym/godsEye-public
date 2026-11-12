"use client";

import { ArrowRight } from "@phosphor-icons/react";
import { AnimatePresence, motion, useReducedMotion } from "motion/react";
import { useEffect, useState } from "react";
import { Interpretation } from "@/components/interpretation";
import { Player } from "@/components/player";
import { QueryBar } from "@/components/query-bar";
import { Results } from "@/components/results";
import { ScopePicker, useScopes } from "@/components/scope";
import { Trail } from "@/components/trail";
import { type PhotoRef, type SearchEvent, useSearch } from "@/lib/api";

const EXAMPLES = [
  "find a person with a tattoo on his right hand",
  "someone texting on a phone",
  "a man wearing glasses",
  "a person getting out of a car",
  "a person with a logo on their shirt",
];

export default function Ask() {
  const { state, run } = useSearch();
  const [selected, setSelected] = useState<SearchEvent | null>(null);
  const [scope, setScope] = useState("all");
  const scopes = useScopes();
  const reduce = useReducedMotion();
  const busy = state.phase !== "idle" && state.phase !== "done" && state.phase !== "error";
  // A scope from the list, or any single camera id (from a video's "Search this footage").
  const camerasOf = (id: string) => scopes.find((s) => s.id === id)?.cameras ?? (id === "all" ? [] : [id]);

  const search = (q: string, within = scope, photo: PhotoRef | null = null) => {
    setSelected(null);
    const params = new URLSearchParams({ q, ...(within !== "all" ? { scope: within } : {}) });
    if (photo) {
      params.set("photo", photo.id);
      params.set("person", String(photo.person));
    }
    window.history.replaceState(null, "", `?${params}`); // shareable, survives reloads
    run(q, camerasOf(within), photo);
  };

  // The URL drives the first search; "meva" needs the camera list, so wait for the scopes.
  const [started, setStarted] = useState(false);
  useEffect(() => {
    const params = new URLSearchParams(window.location.search);
    const within = params.get("scope") ?? "all";
    setScope(within);
    const q = params.get("q") ?? "";
    const photo = params.get("photo") ? { id: params.get("photo")!, person: Number(params.get("person") ?? 0) } : null;
    if (started || (!q && !photo) || (within === "meva" && scopes.length < 2)) return;
    setStarted(true);
    run(q, camerasOf(within), photo);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [run, scopes]);

  return (
    <div className="mx-auto flex max-w-[1600px] flex-col gap-6 px-4 py-6 md:px-8">
      <div className="flex flex-col gap-3">
        <QueryBar initial={state.query} initialPhoto={state.photo} busy={busy} onSubmit={(q, photo) => search(q, scope, photo)} />
        <div className="flex flex-wrap items-center justify-between gap-3">
          <ScopePicker
            scopes={scopes}
            value={scope}
            onChange={(id) => {
              setScope(id);
              if (state.query || state.photo) search(state.query, id, state.photo);
            }}
          />
        </div>
        <Interpretation state={state} />
      </div>

      {state.phase === "idle" ? (
        <section className="flex max-w-2xl flex-col gap-4 pt-10">
          <h1 className="text-2xl font-medium tracking-tight text-text">Ask your cameras what happened.</h1>
          <p className="max-w-[60ch] text-sm leading-relaxed text-muted">
            Searches the MEVA school-site cameras and the videos you upload. Describe a person, a vehicle, an object, an action or a small
            detail like a tattoo or glasses, in plain words.
          </p>
          <ul className="mt-2 flex flex-col divide-y divide-line border-y border-line">
            {EXAMPLES.map((q) => (
              <li key={q}>
                <button
                  onClick={() => search(q)}
                  className="group flex w-full items-center justify-between py-3 text-left text-sm text-muted hover:text-text"
                >
                  {q}
                  <ArrowRight size={16} className="text-faint transition-transform group-hover:translate-x-0.5 group-hover:text-accent" />
                </button>
              </li>
            ))}
          </ul>
        </section>
      ) : (
        <div className="grid grid-cols-1 gap-8 lg:grid-cols-[minmax(0,1fr)_440px]">
          <div>
            {state.error && (
              <p className="mb-6 rounded-[var(--radius-ui)] border border-alert/40 bg-alert/10 px-4 py-3 text-sm text-text">{state.error}</p>
            )}
            {state.mode === "photo" && <Trail state={state} selected={selected?.id ?? null} onSelect={setSelected} />}
            {state.phase === "searching" ? <Skeleton /> : <Results state={state} selected={selected?.id ?? null} onSelect={setSelected} />}
          </div>
          <AnimatePresence mode="wait">
            {selected && (
              <motion.div
                key={selected.id}
                initial={reduce ? false : { opacity: 0, x: 12 }}
                animate={{ opacity: 1, x: 0 }}
                exit={reduce ? undefined : { opacity: 0, x: 12 }}
                transition={{ duration: 0.25, ease: [0.16, 1, 0.3, 1] }}
                className="lg:sticky lg:top-6 lg:self-start"
              >
                <Player event={selected} verdict={state.verdicts[selected.id]} reference={state.photo} onClose={() => setSelected(null)} />
              </motion.div>
            )}
          </AnimatePresence>
        </div>
      )}
    </div>
  );
}

function Skeleton() {
  return (
    <div className="grid grid-cols-[repeat(auto-fill,minmax(176px,1fr))] gap-3">
      {Array.from({ length: 10 }, (_, i) => (
        <div key={i} className="overflow-hidden rounded-[var(--radius-ui)] border border-line bg-surface">
          <div className="aspect-[4/3] animate-pulse bg-surface-2" />
          <div className="h-8" />
        </div>
      ))}
    </div>
  );
}
