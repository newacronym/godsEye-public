import { SCALE_LABEL, type SearchState, seconds } from "@/lib/api";

const INTENT: Record<string, string> = {
  appearance: "Appearance",
  state: "Ongoing activity",
  action: "Action",
  count: "Count",
};

function Chip({ label, value }: { label: string; value: string }) {
  return (
    <span className="inline-flex items-center gap-1.5 rounded-[var(--radius-ui)] border border-line bg-surface px-2.5 py-1 text-xs">
      <span className="text-faint">{label}</span>
      <span className="text-text">{value}</span>
    </span>
  );
}

export function Interpretation({ state }: { state: SearchState }) {
  const { plan, timings, phase } = state;
  const window =
    plan?.time_from || plan?.time_to ? `${plan.time_from ?? "start"} to ${plan.time_to ?? "end"}` : "All footage";
  return (
    <div className="flex flex-wrap items-center justify-between gap-3">
      <div className="flex min-h-7 flex-wrap items-center gap-2">
        {plan ? (
          <>
            <Chip label="Looking for" value={plan.object_phrase} />
            {plan.attributes.map((a) => (
              <Chip
                key={a.phrase + a.scale}
                label={a.scale === "hand" && a.side !== "either" ? `${a.side} ${SCALE_LABEL[a.scale]}` : SCALE_LABEL[a.scale]}
                value={a.phrase}
              />
            ))}
            {plan.activity && <Chip label={INTENT[plan.intent] ?? "Activity"} value={plan.activity} />}
            <Chip label="Cameras" value={plan.cameras.length ? plan.cameras.join(", ") : "All"} />
            <Chip label="When" value={window} />
          </>
        ) : phase !== "idle" ? (
          <span className="text-xs text-faint">Interpreting your question</span>
        ) : null}
      </div>
      {phase !== "idle" && (
        <dl className="flex gap-4 font-mono text-[11px] text-faint">
          <Timing label="first results" ms={timings.provisional} />
          <Timing label="plan" ms={timings.plan} />
          <Timing label="first verified" ms={timings.firstVerified} />
          <Timing label="all checked" ms={timings.done} />
        </dl>
      )}
    </div>
  );
}

function Timing({ label, ms }: { label: string; ms?: number }) {
  return (
    <div className="flex gap-1.5">
      <dt>{label}</dt>
      <dd className={ms === undefined ? "text-faint" : "text-muted"}>{seconds(ms)}</dd>
    </div>
  );
}
