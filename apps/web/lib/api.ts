"use client";

import { useCallback, useEffect, useRef, useState } from "react";

// The API runs on the Lightning Studio; `make tunnel` forwards it (and MediaMTX) to localhost.
export const API = process.env.NEXT_PUBLIC_API ?? "http://localhost:8000";
export const WEBRTC = process.env.NEXT_PUBLIC_WEBRTC ?? "http://localhost:8889";

export type Box = [number, number, number, number];

export interface SearchEvent {
  id: string;
  kind: "track" | "window";
  camera: string;
  class: string | null;
  start: string;
  end: string;
  moment: string;
  score: number;
  bbox: Box;
  image: string;
  detail?: Detail;
  sightings?: number; // the same object tracked in several pieces, merged into this result
}

export type Part = "hand" | "head" | "torso" | "legs" | "feet";
export type Scale = "whole" | Part;

/** Events found through a body part: the close-up where the condition shows best. */
export interface Detail {
  part: Part;
  side: "L" | "R" | null; // the person's own side (pose estimate), hands only
  box: Box;
}

export interface PartView {
  id: number;
  ts: string;
  side: "L" | "R" | null;
  box: Box;
  px: number;
  image: string;
}

export interface Attribute {
  phrase: string;
  scale: Scale;
  side: "left" | "right" | "either";
}

export interface Plan {
  subject: "person" | "vehicle" | "bag" | "other";
  object_phrase: string;
  attributes: Attribute[];
  activity: string;
  intent: "appearance" | "state" | "action" | "count";
  cameras: string[];
  time_from: string | null;
  time_to: string | null;
}

export interface Check {
  condition: string;
  answer: "yes" | "maybe" | "no" | "not_visible";
}

export interface Verdict {
  id: string;
  match: boolean | null;
  unsure: boolean | null; // nothing contradicts the question but something couldn't be seen: a possible match
  checks: Check[]; // one answer per condition
  by?: string; // which model checked it
  confidence: number | null;
  reason: string;
  ms: number;
}

export interface Source {
  camera: string; // "U-xxxxxx": an upload is searched like a camera
  name: string;
  status: "uploading" | "preparing" | "indexing" | "loading" | "ready" | "failed";
  stage?: string;
  fraction?: number;
  message?: string | null;
  created: string;
  start?: string;
  duration_s?: number;
  width?: number;
  height?: number;
  loaded?: { tracks: number; crops: number; frames: number; parts: number };
  index_stats?: { total_s?: number; realtime_x?: number };
}

export interface SourcesResponse {
  uploads: Source[];
  library: { name: string; cameras: number; tracks: number; start: string | null; end: string | null };
}

export type Phase = "idle" | "searching" | "provisional" | "matched" | "done" | "error";

/** A photo to search by: its id on the Studio and which detected person in it. */
export interface PhotoRef {
  id: string;
  person: number;
}

export interface UploadedPhoto {
  id: string;
  width: number;
  height: number;
  people: { box: Box; conf: number }[];
}

export function photoUrl(p: PhotoRef): string {
  return `${API}/photos/${p.id}/${p.person}.jpg`;
}

export interface SearchState {
  query: string;
  photo: PhotoRef | null;
  phase: Phase;
  events: SearchEvent[];
  plan: Plan | null;
  intent: string | null;
  verdicts: Record<string, Verdict>;
  mode: "attributes" | "scene" | "photo" | null; // attributes: every tracked person or object scored per condition
  verifyTop: number;
  timings: { provisional?: number; plan?: number; matched?: number; firstVerified?: number; done?: number };
  error: string | null;
}

const EMPTY: SearchState = {
  query: "",
  photo: null,
  phase: "idle",
  events: [],
  plan: null,
  intent: null,
  verdicts: {},
  mode: null,
  verifyTop: 20,
  timings: {},
  error: null,
};

export function useSearch() {
  const [state, setState] = useState<SearchState>(EMPTY);
  const source = useRef<EventSource | null>(null);

  const stop = useCallback(() => {
    source.current?.close();
    source.current = null;
  }, []);

  const run = useCallback(
    (query: string, cameras: string[] = [], photo: PhotoRef | null = null, verifyTop = 20) => {
      stop();
      setState({ ...EMPTY, query, photo, phase: "searching", verifyTop });
      const params = new URLSearchParams({ q: query, verify_top: String(verifyTop) });
      if (cameras.length) params.set("cameras", cameras.join(","));
      if (photo) {
        params.set("photo", photo.id);
        params.set("person", String(photo.person));
      }
      const es = new EventSource(`${API}/search?${params}`);
      source.current = es;
      const on = <T,>(type: string, fn: (data: T, s: SearchState) => Partial<SearchState>) =>
        es.addEventListener(type, (e) => {
          const data = JSON.parse((e as MessageEvent).data) as T;
          setState((s) => ({ ...s, ...fn(data, s) }));
        });

      on<{ ms: number; events: SearchEvent[] }>("provisional", (d, s) => ({
        phase: "provisional",
        events: d.events,
        timings: { ...s.timings, provisional: d.ms },
      }));
      on<{ ms: number; plan: Plan | null }>("plan", (d, s) => ({ plan: d.plan, timings: { ...s.timings, plan: d.ms } }));
      on<{ ms: number; events: SearchEvent[]; mode: "attributes" | "scene" | "photo" }>("matched", (d, s) => ({
        phase: "matched",
        events: d.events,
        mode: d.mode,
        timings: { ...s.timings, matched: d.ms },
      }));
      on<Verdict>("verdict", (d, s) => ({
        verdicts: { ...s.verdicts, [d.id]: d },
        timings: d.match && s.timings.firstVerified === undefined ? { ...s.timings, firstVerified: d.ms } : s.timings,
      }));
      on<{ ms: number; intent: string }>("done", (d, s) => {
        stop();
        return { phase: "done", intent: d.intent, timings: { ...s.timings, done: d.ms } };
      });
      // EventSource retries forever on its own; a search is one-shot, so stop and report instead.
      es.onerror = () => {
        stop();
        setState((s) =>
          s.phase === "done" ? s : { ...s, phase: "error", error: "Lost the connection to the search API. Is `make tunnel` running?" },
        );
      };
    },
    [stop],
  );

  useEffect(() => stop, [stop]);
  return { state, run };
}

export function clock(iso: string): string {
  return iso.slice(11, 19);
}

export function seconds(ms?: number): string {
  return ms === undefined ? "-" : `${(ms / 1000).toFixed(1)}s`;
}

export const SIDE: Record<"L" | "R", string> = { L: "left", R: "right" };

export function partLabel(part: Part, side?: "L" | "R" | null): string {
  return part === "hand" && side ? `${SIDE[side]} hand` : part;
}

export const SCALE_LABEL: Record<Scale, string> = {
  whole: "whole view",
  head: "head close-up",
  torso: "torso close-up",
  hand: "hand close-up",
  legs: "legs close-up",
  feet: "feet close-up",
};
